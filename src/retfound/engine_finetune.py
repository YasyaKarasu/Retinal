import csv
import json
import os
from contextlib import nullcontext
from typing import Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from timm.data import Mixup

from .util import lr_sched, misc


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()

    if log_writer:
        print(f"log_dir: {log_writer.log_dir}")

    header = f"Epoch: [{epoch}]"
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)

        update_grad = (
            (data_iter_step + 1) % accum_iter == 0
            or data_iter_step + 1 == len(data_loader)
        )
        sync_context = (
            model.no_sync()
            if hasattr(model, "no_sync") and not update_grad
            else nullcontext()
        )
        with sync_context:
            with torch.amp.autocast(
                device_type="cuda", enabled=device.type == "cuda"
            ):
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_value = loss.item()
            loss /= accum_iter
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=update_grad,
            )
        if update_grad:
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and update_grad:
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            log_writer.add_scalar("loss/train", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {
        key: meter.global_avg for key, meter in metric_logger.meters.items()
    }


def _macro_auc(targets, probabilities):
    valid = np.logical_and(
        targets.sum(axis=0) > 0, targets.sum(axis=0) < len(targets)
    )
    if not valid.any():
        return float("nan"), 0
    return (
        roc_auc_score(
            targets[:, valid], probabilities[:, valid], average="macro"
        ),
        int(valid.sum()),
    )


def _macro_average_precision(targets, probabilities):
    valid = targets.sum(axis=0) > 0
    if not valid.any():
        return float("nan"), 0
    return average_precision_score(
        targets[:, valid], probabilities[:, valid], average="macro"
    ), int(valid.sum())


def _macro_f1(targets, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    valid = targets.sum(axis=0) > 0
    return f1_score(
        targets[:, valid],
        predictions[:, valid],
        average="macro",
        zero_division=0,
    )


def _select_tied_threshold(grid, scores, reference):
    scores = np.asarray(scores)
    best_score = np.nanmax(scores)
    candidates = grid[np.isclose(scores, best_score)]
    return float(candidates[np.argmin(np.abs(candidates - reference))])


def calibrate_thresholds(targets, probabilities, class_names, args):
    """Calibrate thresholds while limiting implausible positive-rate inflation."""
    grid = np.linspace(
        args.threshold_min,
        args.threshold_max,
        args.threshold_steps,
        dtype=np.float64,
    )
    global_scores = [
        _macro_f1(targets, probabilities, threshold) for threshold in grid
    ]
    global_threshold = _select_tied_threshold(
        grid, global_scores, args.threshold
    )
    thresholds = np.full(len(class_names), global_threshold, dtype=np.float64)
    calibrated = np.zeros(len(class_names), dtype=bool)
    validation_predictions = np.zeros(len(class_names), dtype=np.int64)
    constrained_candidates = np.zeros(len(class_names), dtype=np.int64)

    if args.threshold_strategy == "per_class":
        for index in range(len(class_names)):
            positives = int(targets[:, index].sum())
            negatives = len(targets) - positives
            if (
                positives < args.threshold_min_positives
                or negatives < args.threshold_min_positives
            ):
                continue

            target = targets[:, index]
            minimum_predictions = max(
                1,
                int(
                    np.floor(
                        positives
                        * args.threshold_min_prevalence_multiplier
                    )
                ),
            )
            maximum_predictions = min(
                len(target),
                max(
                    positives,
                    int(
                        np.ceil(
                            positives
                            * args.threshold_max_prevalence_multiplier
                        )
                    ),
                ),
            )

            candidate_thresholds = []
            candidate_scores = []
            for threshold in grid:
                if (
                    threshold
                    < global_threshold
                    - args.threshold_max_below_global
                ):
                    continue
                prediction = probabilities[:, index] >= threshold
                predicted_positives = int(prediction.sum())
                if not (
                    minimum_predictions
                    <= predicted_positives
                    <= maximum_predictions
                ):
                    continue
                candidate_thresholds.append(threshold)
                candidate_scores.append(
                    f1_score(target, prediction, zero_division=0)
                )

            constrained_candidates[index] = len(candidate_thresholds)
            if not candidate_thresholds:
                continue

            candidate_thresholds = np.asarray(candidate_thresholds)
            thresholds[index] = _select_tied_threshold(
                candidate_thresholds,
                candidate_scores,
                global_threshold,
            )
            calibrated[index] = True

    for index, threshold in enumerate(thresholds):
        validation_predictions[index] = int(
            (probabilities[:, index] >= threshold).sum()
        )

    details = {
        "strategy": args.threshold_strategy,
        "global_threshold": global_threshold,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_steps": args.threshold_steps,
        "threshold_min_positives": args.threshold_min_positives,
        "threshold_min_prevalence_multiplier": (
            args.threshold_min_prevalence_multiplier
        ),
        "threshold_max_prevalence_multiplier": (
            args.threshold_max_prevalence_multiplier
        ),
        "threshold_max_below_global": args.threshold_max_below_global,
        "classes": {
            class_name: {
                "threshold": float(thresholds[index]),
                "calibrated_per_class": bool(calibrated[index]),
                "validation_positives": int(targets[:, index].sum()),
                "validation_predicted_positives": int(
                    validation_predictions[index]
                ),
                "constrained_candidates": int(
                    constrained_candidates[index]
                ),
            }
            for index, class_name in enumerate(class_names)
        },
    }
    return thresholds, details


def _per_class_rows(targets, probabilities, predictions, thresholds, class_names):
    rows = []
    for index, class_name in enumerate(class_names):
        target = targets[:, index]
        probability = probabilities[:, index]
        prediction = predictions[:, index]
        positives = int(target.sum())
        negatives = len(target) - positives

        auc = float("nan")
        if positives > 0 and negatives > 0:
            auc = roc_auc_score(target, probability)
        average_precision = float("nan")
        if positives > 0:
            average_precision = average_precision_score(target, probability)

        rows.append(
            {
                "class": class_name,
                "positive_count": positives,
                "negative_count": negatives,
                "prevalence": positives / len(target),
                "threshold": float(thresholds[index]),
                "predicted_positive_count": int(prediction.sum()),
                "precision": precision_score(
                    target, prediction, zero_division=0
                ),
                "recall": recall_score(
                    target, prediction, zero_division=0
                ),
                "f1": f1_score(target, prediction, zero_division=0),
                "roc_auc": auc,
                "average_precision": average_precision,
            }
        )
    return rows


def _write_detailed_outputs(
    output_dir,
    mode,
    strategy,
    image_ids,
    targets,
    probabilities,
    predictions,
    thresholds,
    class_names,
    threshold_details,
):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"{mode}_{strategy}"

    per_class_rows = _per_class_rows(
        targets, probabilities, predictions, thresholds, class_names
    )
    per_class_path = os.path.join(output_dir, f"per_class_{suffix}.csv")
    with open(per_class_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)

    prediction_path = os.path.join(output_dir, f"predictions_{suffix}.csv")
    prediction_fields = ["ID"]
    for class_name in class_names:
        prediction_fields.extend(
            [
                f"true_{class_name}",
                f"prob_{class_name}",
                f"pred_{class_name}",
            ]
        )
    with open(prediction_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for row_index, image_id in enumerate(image_ids):
            row = {"ID": image_id}
            for class_index, class_name in enumerate(class_names):
                row[f"true_{class_name}"] = int(
                    targets[row_index, class_index]
                )
                row[f"prob_{class_name}"] = float(
                    probabilities[row_index, class_index]
                )
                row[f"pred_{class_name}"] = int(
                    predictions[row_index, class_index]
                )
            writer.writerow(row)

    if threshold_details is not None:
        threshold_path = os.path.join(output_dir, "thresholds.json")
        with open(threshold_path, "w", encoding="utf-8") as handle:
            json.dump(threshold_details, handle, indent=2)


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    criterion,
    device,
    args,
    epoch,
    mode,
    log_writer,
    class_names,
    thresholds=None,
    calibrate=False,
    export_details=False,
):
    """Evaluate sigmoid multi-label predictions."""
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()
    image_ids_all, targets_all, probabilities_all = [], [], []

    for batch in metric_logger.log_every(
        data_loader, 10, f"{mode}:"
    ):
        if len(batch) == 3:
            images, targets, image_ids = batch
            image_ids_all.extend(list(image_ids))
        else:
            images, targets = batch
            image_ids_all.extend(
                str(len(image_ids_all) + index)
                for index in range(len(images))
            )
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda", enabled=device.type == "cuda"
        ):
            outputs = model(images)
            loss = criterion(outputs, targets)

        metric_logger.update(loss=loss.item())
        targets_all.append(targets.cpu())
        probabilities_all.append(torch.sigmoid(outputs).cpu())

    metric_logger.synchronize_between_processes()
    targets = torch.cat(targets_all).numpy().astype(np.int64)
    probabilities = torch.cat(probabilities_all).numpy()

    if misc.get_world_size() > 1 and args.dist_eval:
        local_result = (targets, probabilities, image_ids_all)
        gathered = [None] * misc.get_world_size()
        dist.all_gather_object(gathered, local_result)
        targets = np.concatenate([item[0] for item in gathered], axis=0)
        probabilities = np.concatenate(
            [item[1] for item in gathered], axis=0
        )
        image_ids_all = [
            image_id for item in gathered for image_id in item[2]
        ]

    threshold_details = None
    if calibrate:
        thresholds, threshold_details = calibrate_thresholds(
            targets, probabilities, class_names, args
        )
    elif thresholds is None:
        thresholds = np.full(
            len(class_names), args.threshold, dtype=np.float64
        )
    thresholds = np.asarray(thresholds, dtype=np.float64)

    predictions = (probabilities >= thresholds[None, :]).astype(np.int64)
    positive_labels = targets.sum(axis=0) > 0
    macro_targets = targets[:, positive_labels]
    macro_predictions = predictions[:, positive_labels]

    subset_accuracy = accuracy_score(targets, predictions)
    hamming = hamming_loss(targets, predictions)
    f1_macro = f1_score(
        macro_targets, macro_predictions, average="macro", zero_division=0
    )
    f1_micro = f1_score(
        targets, predictions, average="micro", zero_division=0
    )
    precision_macro = precision_score(
        macro_targets, macro_predictions, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        macro_targets, macro_predictions, average="macro", zero_division=0
    )
    jaccard_macro = jaccard_score(
        macro_targets, macro_predictions, average="macro", zero_division=0
    )
    roc_auc_macro, auc_labels = _macro_auc(targets, probabilities)
    average_precision_macro, ap_labels = _macro_average_precision(
        targets, probabilities
    )
    score = np.nanmean(
        [f1_macro, roc_auc_macro, average_precision_macro]
    )

    metrics = {
        "loss": metric_logger.meters["loss"].global_avg,
        "subset_accuracy": subset_accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "roc_auc_macro": roc_auc_macro,
        "average_precision_macro": average_precision_macro,
        "hamming_loss": hamming,
        "jaccard_macro": jaccard_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "score": score,
    }

    if misc.is_main_process():
        print(
            f"{mode} loss: {metrics['loss']:.4f}, "
            f"macro F1: {f1_macro:.4f}, micro F1: {f1_micro:.4f}, "
            f"macro AUROC: {roc_auc_macro:.4f} ({auc_labels} labels), "
            f"macro AP: {average_precision_macro:.4f} ({ap_labels} labels)"
        )
        print(
            f"subset accuracy: {subset_accuracy:.4f}, "
            f"hamming loss: {hamming:.4f}, "
            f"macro precision: {precision_macro:.4f}, "
            f"macro recall: {recall_macro:.4f}"
        )
        print(
            f"thresholds: min={thresholds.min():.3f}, "
            f"mean={thresholds.mean():.3f}, max={thresholds.max():.3f}"
        )

        output_dir = os.path.join(args.output_dir, args.task)
        os.makedirs(output_dir, exist_ok=True)
        strategy = (
            args.threshold_strategy if calibrate or thresholds is not None
            else "fixed"
        )
        results_path = os.path.join(
            output_dir, f"metrics_{mode}_{strategy}.csv"
        )
        file_exists = os.path.isfile(results_path)
        with open(results_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics))
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

        if log_writer:
            for metric_name, value in metrics.items():
                log_writer.add_scalar(f"perf/{metric_name}", value, epoch)

        if export_details:
            _write_detailed_outputs(
                output_dir,
                mode,
                strategy,
                image_ids_all,
                targets,
                probabilities,
                predictions,
                thresholds,
                class_names,
                threshold_details,
            )

    return metrics, score, thresholds
