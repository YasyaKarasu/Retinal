#!/usr/bin/env python3

# =========================
import argparse
import datetime
import json
import os
import time
from pathlib import Path
import warnings
import faulthandler

# =========================
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup

# =========================
from . import models_vit as models
from . import engine_finetune
from .checkpoints import load_pretrained_weights
from .util import lr_decay as lrd
from .util import misc
from .util.datasets import (
    DistributedEvalSampler,
    build_dataset,
    get_class_names,
    get_target_matrix,
)
from .util.misc import NativeScalerWithGradNormCount as NativeScaler

# =========================
faulthandler.enable()
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "RETFound fine-tuning / linear probing for image classification"
    )

    # ---- Core training
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size per GPU (effective batch size = batch_size * accum_iter * #gpus)")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=1, type=int,
                        help="Gradient accumulation steps")

    # ---- Model parameters
    parser.add_argument("--model", default="vit_large_patch16", type=str, metavar="MODEL",
                        help="Model entry in models_vit.py")
    parser.add_argument("--model_arch", default="dinov3_vits16", type=str, metavar="MODEL_ARCH",
                        help="Backbone architecture key (e.g., dinov2_vitl14, convnext_base, etc.)")
    parser.add_argument("--input_size", default=256, type=int, help="Image size")
    parser.add_argument("--drop_path", type=float, default=0.2, metavar="PCT", help="Drop path rate")
    parser.add_argument("--global_pool", action="store_true"); parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool",
                        help="Use class token instead of global pool for classification")

    # ---- Optimizer parameters
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM", help="Clip grad norm")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="Absolute LR (overrides blr)")
    parser.add_argument("--blr", type=float, default=5e-3, metavar="LR",
                        help="Base LR: lr = blr * total_batch_size / 256")
    parser.add_argument("--layer_decay", type=float, default=0.65, help="Layer-wise LR decay (ViT)")
    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="LR", help="Lower LR bound")
    parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N", help="Warmup epochs")

    # ---- Augmentation
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", metavar="NAME")
    parser.add_argument("--smoothing", type=float, default=0.1)

    # ---- Random erase
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT")
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", action="store_true", default=False)

    # ---- Mixup/Cutmix
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")

    # ---- Finetuning & adaptation
    parser.add_argument(
        "--finetune",
        default="",
        type=str,
        help="Local checkpoint path or official Hugging Face RETFound model id",
    )
    parser.add_argument("--task", default="", type=str, help="Task name for logging/output grouping")
    parser.add_argument("--adaptation", default="finetune", choices=["finetune", "lp"],
                        help="Adaptation strategy: finetune=full fine-tune, lp=linear probe (train head only)")

    # ---- Dataset & paths
    parser.add_argument(
        "--dataset",
        default="rfmid",
        choices=["rfmid", "imagefolder"],
        help="Dataset format",
    )
    parser.add_argument("--data_path", default="./dataset", type=str)
    parser.add_argument(
        "--image_cache_dir",
        default="",
        type=str,
        help="Optional resized RFMiD cache directory",
    )
    parser.add_argument("--nb_classes", default=45, type=int)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default="./output_logs")
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Sigmoid threshold used for multi-label evaluation",
    )
    parser.add_argument(
        "--threshold_strategy",
        default="fixed",
        choices=["fixed", "global", "per_class"],
        help="Threshold calibration strategy used during evaluation",
    )
    parser.add_argument("--threshold_min", default=0.01, type=float)
    parser.add_argument("--threshold_max", default=0.99, type=float)
    parser.add_argument("--threshold_steps", default=99, type=int)
    parser.add_argument(
        "--threshold_min_positives",
        default=10,
        type=int,
        help="Minimum validation positives and negatives for per-class calibration",
    )
    parser.add_argument(
        "--threshold_min_prevalence_multiplier",
        default=0.25,
        type=float,
        help="Minimum predicted positives relative to validation positives",
    )
    parser.add_argument(
        "--threshold_max_prevalence_multiplier",
        default=3.0,
        type=float,
        help="Maximum predicted positives relative to validation positives",
    )
    parser.add_argument(
        "--threshold_max_below_global",
        default=0.15,
        type=float,
        help="Maximum amount a per-class threshold may fall below the global threshold",
    )
    parser.add_argument(
        "--use_pos_weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weight positive RFMiD labels by inverse prevalence",
    )
    parser.add_argument(
        "--pos_weight_max",
        default=20.0,
        type=float,
        help="Maximum BCE positive-class weight",
    )

    # >>> NEW: training data efficiency <<<
    parser.add_argument(
        "--dataratio", type=str, default="1.0",
        help="Training data ratio in (0, 1], for example 0.25",
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="If set, subsample training data in a class-stratified manner (requires support in build_dataset)."
    )

    # ---- Runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="Resume full state (optimizer, scaler, etc.)")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true", help="Evaluation only")
    parser.add_argument("--dist_eval", action="store_true", default=False,
                        help="Distributed evaluation (faster monitoring during training)")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true"); parser.set_defaults(pin_mem=True)
    parser.add_argument("--prefetch_factor", default=2, type=int)
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fused AdamW when training on CUDA",
    )

    # ---- Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    # ---- Misc
    parser.add_argument("--savemodel", action="store_true", default=True, help="Save best model")
    parser.add_argument("--norm", default="IMAGENET", type=str)
    parser.add_argument("--enhance", action="store_true", default=False)
    parser.add_argument("--datasets_seed", default=2026, type=int)

    return parser


# =========================
# Main
# =========================
def main(args):
    # ---- Optionally load args from resume (when training)
    if args.resume and not args.eval:
        resume_path = args.resume
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        print(f"Load checkpoint (args) from: {args.resume}")
        args = checkpoint["args"]
        args.resume = resume_path

    # ---- Distributed setup
    misc.init_distributed_mode(args)

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)

    # ---- Reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ---- Build model
    if args.model == "RETFound_mae":
        model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )

    # ---- Load pre-trained weights (if requested and not eval-only)
    if args.finetune and not args.eval:
        print(f"Preparing to load pre-trained weights: {args.finetune}")

        checkpoint_path, load_result = load_pretrained_weights(
            model, args.model, args.finetune
        )
        print(f"Loaded pre-trained checkpoint from: {checkpoint_path}")
        print(
            "Checkpoint load result: "
            f"{len(load_result.missing_keys)} missing keys, "
            f"{len(load_result.unexpected_keys)} unexpected keys"
        )

        # -- Re-init head
        if hasattr(model, "head") and hasattr(model.head, "weight"):
            trunc_normal_(model.head.weight, std=2e-5)

    # ---- Datasets & samplers
    dataset_train = build_dataset(is_train="train", args=args)
    dataset_val   = build_dataset(is_train="val",   args=args)
    dataset_test  = build_dataset(is_train="test",  args=args)
    class_names = get_class_names(dataset_train)

    if args.dataset != "rfmid":
        raise NotImplementedError(
            "The current engine is configured for multi-label RFMiD training"
        )

    if args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None:
        raise ValueError(
            "Mixup/CutMix are disabled for the RFMiD multi-label pipeline"
        )

    pos_weight = None
    if args.use_pos_weight:
        train_targets = get_target_matrix(dataset_train)
        positives = train_targets.sum(dim=0)
        negatives = len(train_targets) - positives
        pos_weight = torch.where(
            positives > 0,
            negatives / positives.clamp_min(1),
            torch.ones_like(positives),
        ).clamp(max=args.pos_weight_max)
        if misc.is_main_process():
            zero_positive = int((positives == 0).sum().item())
            print(
                f"Using BCE pos_weight (max={args.pos_weight_max:g}); "
                f"{zero_positive} labels have no positive training examples"
            )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )

    num_tasks   = misc.get_world_size()
    global_rank = misc.get_rank()

    if not args.eval:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f"Sampler_train = {sampler_train}")

    if args.dist_eval:
        sampler_val = DistributedEvalSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.dist_eval:
        sampler_test = DistributedEvalSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank
        )
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # ---- Logging
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
    else:
        log_writer = None

    # ---- DataLoaders
    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        print(f"len of train_set: {len(data_loader_train) * args.batch_size}")

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # ---- Mixup/CutMix
    mixup_fn = None
    mixup_active = (args.mixup > 0) or (args.cutmix > 0.) or (args.cutmix_minmax is not None)
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    # ---- Eval-only: resume weights
    if args.resume and args.eval:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        print(f"Load checkpoint for eval from: {args.resume}")
        model.load_state_dict(checkpoint["model"])

    model.to(device)
    model_without_ddp = model

    # ---- Adaptation toggle
    if args.adaptation == "lp":
        for name, param in model.named_parameters():
            param.requires_grad = ("head" in name)
        print("[Adaptation] Linear probe: training classifier head only.")
    else:
        print("[Adaptation] Full fine-tuning: training all parameters.")

    # ---- Count trainable params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params (M): {n_parameters / 1.e6:.2f}")

    # ---- LR scaling by effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    # ---- DDP (if available)
    if args.distributed and torch.cuda.device_count() > 1:
        ddp_kwargs = {}
        if args.adaptation == "lp":
            ddp_kwargs["find_unused_parameters"] = True
        else:
            ddp_kwargs["static_graph"] = True
        ddp_kwargs["gradient_as_bucket_view"] = True
        ddp_kwargs["broadcast_buffers"] = False
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], **ddp_kwargs
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model  # single-GPU

    # ---- Optimizer param groups (after freezing)
    no_weight_decay = (model_without_ddp.no_weight_decay()
                       if hasattr(model_without_ddp, "no_weight_decay") else [])


    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        weight_decay=args.weight_decay,
        no_weight_decay_list=no_weight_decay,
        layer_decay=args.layer_decay,
    )
    for g in param_groups:
        g["params"] = [p for p in g["params"] if p.requires_grad]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        fused=args.fused_optimizer and device.type == "cuda",
    )
    loss_scaler = NativeScaler()
    print(f"criterion = {criterion}")

    # ---- Load previous full state (optimizer, scaler, etc.)
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # =========================
    # Eval-only Short Circuit
    # =========================
    if args.eval:
        if "checkpoint" in locals() and isinstance(checkpoint, dict) and ("epoch" in checkpoint):
            print(f"Test with the best model at epoch = {checkpoint['epoch']}")
        calibrated_thresholds = None
        if args.threshold_strategy != "fixed":
            print(
                f"Calibrating {args.threshold_strategy} thresholds "
                "on the validation split"
            )
            _, _, calibrated_thresholds = engine_finetune.evaluate(
                data_loader_val,
                model,
                criterion,
                device,
                args,
                epoch=0,
                mode="val",
                log_writer=log_writer,
                class_names=class_names,
                calibrate=True,
                export_details=True,
            )

        test_stats, auc_roc, _ = engine_finetune.evaluate(
            data_loader_test,
            model,
            criterion,
            device,
            args,
            epoch=0,
            mode="test",
            log_writer=log_writer,
            class_names=class_names,
            thresholds=calibrated_thresholds,
            export_details=True,
        )
        misc.cleanup_distributed()
        return

    # =========================
    # Train Loop
    # =========================
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = engine_finetune.train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer, args=args
        )

        val_stats, val_score, _ = engine_finetune.evaluate(
            data_loader_val,
            model,
            criterion,
            device,
            args,
            epoch,
            mode="val",
            log_writer=log_writer,
            class_names=class_names,
        )

        if max_score < val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, mode="best"
                )
        if args.distributed:
            torch.distributed.barrier(device_ids=[args.gpu])
        print(f"Best epoch = {best_epoch}, Best score = {max_score:.4f}")

        if log_writer is not None:
            log_writer.add_scalar("loss/val", val_stats["loss"], epoch)
            log_writer.flush()

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     "epoch": epoch,
                     "n_parameters": n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, args.task, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # =========================
    # Final Test (Best Ckpt)
    # =========================
    ckpt_path = os.path.join(args.output_dir, args.task, "checkpoint-best.pth")
    if args.distributed:
        torch.distributed.barrier(device_ids=[args.gpu])
    checkpoint = torch.load(
        ckpt_path, map_location="cpu", weights_only=False
    )
    model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    print(f"Test with the best model, epoch = {checkpoint.get('epoch', -1)}:")
    _test_stats, _auc_roc, _ = engine_finetune.evaluate(
        data_loader_test,
        model,
        criterion,
        device,
        args,
        -1,
        mode="test",
        log_writer=None,
        class_names=class_names,
        export_details=True,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    misc.cleanup_distributed()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
