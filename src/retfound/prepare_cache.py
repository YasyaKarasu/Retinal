"""Create a resized, lossless RFMiD image cache for faster training."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .util.datasets import RFMID_SPLITS


def _resize_short_side(image, short_side):
    width, height = image.size
    scale = short_side / min(width, height)
    if scale >= 1:
        return image
    output_size = (round(width * scale), round(height * scale))
    return image.resize(output_size, Image.Resampling.LANCZOS)


def _cache_image(source, destination, short_side):
    if destination.is_file():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    with Image.open(source) as image:
        image = _resize_short_side(image.convert("RGB"), short_side)
        image.save(temporary, format="PNG", compress_level=1)
    temporary.replace(destination)
    return True


def prepare_cache(data_path, cache_dir, short_side, workers):
    jobs = []
    data_path = Path(data_path)
    cache_dir = Path(cache_dir)

    for split, (image_directory, _) in RFMID_SPLITS.items():
        source_directory = data_path / image_directory
        if not source_directory.is_dir():
            raise FileNotFoundError(
                f"RFMiD source directory not found: {source_directory}"
            )
        split_cache = cache_dir / split
        for source in source_directory.glob("*.png"):
            jobs.append(
                (source, split_cache / source.name, short_side)
            )

    created = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_cache_image, *job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            created += int(future.result())
            if completed % 250 == 0 or completed == len(futures):
                print(
                    f"RFMiD cache progress: {completed}/{len(futures)}",
                    flush=True,
                )

    print(
        f"RFMiD cache ready: {cache_dir} "
        f"({created} created, {len(jobs) - created} reused)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="dataset")
    parser.add_argument(
        "--cache-dir", default="dataset/.cache/rfmid_768"
    )
    parser.add_argument("--short-side", type=int, default=768)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    prepare_cache(
        args.data_path,
        args.cache_dir,
        args.short_side,
        args.workers,
    )


if __name__ == "__main__":
    main()
