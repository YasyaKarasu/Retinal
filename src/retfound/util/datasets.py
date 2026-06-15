import csv
import os
from pathlib import Path

import torch
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset, Sampler, Subset
from torchvision import datasets, transforms


RFMID_SPLITS = {
    "train": (
        "Training_Set/Training_Set/Training",
        "Training_Set/Training_Set/RFMiD_Training_Labels.csv",
    ),
    "val": (
        "Evaluation_Set/Evaluation_Set/Validation",
        "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv",
    ),
    "test": (
        "Test_Set/Test_Set/Test",
        "Test_Set/Test_Set/RFMiD_Testing_Labels.csv",
    ),
}

# RFMiD challenge label space: 27 conditions with at least 10 examples
# across the official splits, plus one merged class for all rarer conditions.
RFMID_CHALLENGE_RETAINED_CLASSES = (
    "DR",
    "ARMD",
    "MH",
    "DN",
    "MYA",
    "BRVO",
    "TSLN",
    "ERM",
    "LS",
    "MS",
    "CSR",
    "ODC",
    "CRVO",
    "TV",
    "AH",
    "ODP",
    "ODE",
    "ST",
    "AION",
    "PT",
    "RT",
    "RS",
    "CRS",
    "EDN",
    "RPEC",
    "MHL",
    "RP",
)
RFMID_CHALLENGE_CLASS_NAMES = (
    *RFMID_CHALLENGE_RETAINED_CLASSES,
    "OTHER",
)


def get_rfmid_challenge_indices(class_names):
    """Return retained and rare column indices for the challenge schema."""
    class_indices = {name: index for index, name in enumerate(class_names)}
    missing = [
        name
        for name in RFMID_CHALLENGE_RETAINED_CLASSES
        if name not in class_indices
    ]
    if missing:
        raise ValueError(
            "RFMiD challenge classes are missing: " + ", ".join(missing)
        )

    retained_indices = [
        class_indices[name] for name in RFMID_CHALLENGE_RETAINED_CLASSES
    ]
    retained_set = set(RFMID_CHALLENGE_RETAINED_CLASSES)
    rare_indices = [
        index
        for index, name in enumerate(class_names)
        if name not in retained_set
    ]
    if not rare_indices:
        raise ValueError(
            "RFMiD challenge conversion requires rare classes for OTHER"
        )
    return retained_indices, rare_indices


def project_rfmid_challenge_targets(targets, class_names):
    """Convert 45 disease targets to 27 retained targets plus OTHER."""
    retained_indices, rare_indices = get_rfmid_challenge_indices(class_names)
    retained_targets = targets[..., retained_indices]
    other_target = targets[..., rare_indices].amax(dim=-1, keepdim=True)
    return torch.cat([retained_targets, other_target], dim=-1)


class RFMiDDataset(Dataset):
    """RFMiD image dataset backed by the official CSV multi-label annotations."""

    def __init__(
        self,
        root,
        split,
        transform=None,
        include_id=False,
        image_cache_dir=None,
    ):
        if split not in RFMID_SPLITS:
            raise ValueError(f"Unknown RFMiD split: {split}")

        image_dir, csv_file = RFMID_SPLITS[split]
        self.image_dir = Path(root) / image_dir
        self.csv_path = Path(root) / csv_file
        self.transform = transform
        self.include_id = include_id

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"RFMiD image directory not found: {self.image_dir}")
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"RFMiD label file not found: {self.csv_path}")

        with self.csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if fieldnames[:2] != ["ID", "Disease_Risk"]:
                raise ValueError(f"Unexpected RFMiD CSV columns in {self.csv_path}")

            self.class_names = fieldnames[2:]
            records = list(reader)

        self.image_ids = [record["ID"] for record in records]
        original_image_paths = [
            self.image_dir / f"{image_id}.png" for image_id in self.image_ids
        ]
        if image_cache_dir:
            cache_directory = Path(image_cache_dir) / split
            cached_image_paths = [
                cache_directory / f"{image_id}.png"
                for image_id in self.image_ids
            ]
            missing_cache = [
                path for path in cached_image_paths if not path.is_file()
            ]
            if missing_cache:
                raise FileNotFoundError(
                    f"{len(missing_cache)} cached RFMiD images are missing; "
                    "run `python -m retfound.prepare_cache` first"
                )
            self.image_paths = cached_image_paths
        else:
            self.image_paths = original_image_paths

        missing = [path for path in self.image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} RFMiD images are missing; first missing file: {missing[0]}"
            )

        self.targets = torch.tensor(
            [
                [float(record[class_name]) for class_name in self.class_names]
                for record in records
            ],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        if self.include_id:
            return image, self.targets[index], self.image_ids[index]
        return image, self.targets[index]


class DistributedEvalSampler(Sampler):
    """Shard evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == "rfmid":
        dataset = RFMiDDataset(
            args.data_path,
            is_train,
            transform=transform,
            include_id=is_train != "train",
            image_cache_dir=getattr(args, "image_cache_dir", ""),
        )
        if args.nb_classes != len(dataset.class_names):
            raise ValueError(
                f"RFMiD has {len(dataset.class_names)} disease labels, "
                f"but --nb_classes={args.nb_classes}"
            )
    else:
        root = os.path.join(args.data_path, is_train)
        dataset = datasets.ImageFolder(root, transform=transform)

    if is_train == "train":
        ratio = float(getattr(args, "dataratio", 1.0))
        seed = int(getattr(args, "seed", 0))
        stratified = bool(getattr(args, "stratified", False))

        if 0.0 < ratio < 1.0:
            if stratified and args.dataset == "rfmid":
                raise ValueError(
                    "--stratified is not implemented for multi-label RFMiD data"
                )
            if stratified:
                indices = _stratified_indices(dataset.targets, ratio, seed)
            else:
                generator = torch.Generator().manual_seed(seed)
                count = max(1, int(len(dataset) * ratio))
                indices = torch.randperm(
                    len(dataset), generator=generator
                )[:count].tolist()
            dataset = Subset(dataset, indices)

    return dataset


def get_target_matrix(dataset):
    """Return a float target matrix for a dataset or nested Subset."""
    if isinstance(dataset, Subset):
        targets = get_target_matrix(dataset.dataset)
        return targets[torch.as_tensor(dataset.indices)]
    targets = torch.as_tensor(dataset.targets)
    if targets.ndim != 2:
        raise ValueError("Expected a multi-label target matrix")
    return targets.float()


def get_class_names(dataset):
    """Return class names from a dataset or nested Subset."""
    if isinstance(dataset, Subset):
        return get_class_names(dataset.dataset)
    return list(dataset.class_names)


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == "train":
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _stratified_indices(targets, ratio: float, seed: int):
    """Maintain single-label class proportions."""
    targets = torch.as_tensor(targets)
    classes = torch.unique(targets)
    generator = torch.Generator().manual_seed(seed)

    keep = []
    for class_id in classes.tolist():
        class_indices = torch.nonzero(
            targets == class_id, as_tuple=False
        ).view(-1)
        if len(class_indices) == 0:
            continue
        count = max(1, int(round(len(class_indices) * ratio)))
        selected = class_indices[
            torch.randperm(len(class_indices), generator=generator)[:count]
        ]
        keep.extend(selected.tolist())

    shuffle_generator = torch.Generator().manual_seed(seed + 1)
    return torch.tensor(keep)[
        torch.randperm(len(keep), generator=shuffle_generator)
    ].tolist()
