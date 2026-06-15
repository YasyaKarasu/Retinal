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
from .util.datasets import (
    RFMID_CHALLENGE_CLASS_NAMES,
    get_rfmid_challenge_indices,
    project_rfmid_challenge_targets,
)


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
    challenge_criterion=None,
    class_names=None,
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
                logits, challenge_logits = _split_model_outputs(outputs)
                all_classes_loss = criterion(logits, targets)
                challenge_loss = None
                loss = all_classes_loss
                if challenge_logits is not None:
                    if challenge_criterion is None or class_names is None:
                        raise ValueError(
                            "Challenge logits require a challenge criterion "
                            "and RFMiD class names"
                        )
                    challenge_targets = project_rfmid_challenge_targets(
                        targets, class_names
                    )
                    challenge_loss = challenge_criterion(
                        challenge_logits, challenge_targets
                    )
                    loss = (
                        all_classes_loss
                        + args.challenge_loss_weight * challenge_loss
                    )

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
        metric_logger.update(
            loss=loss_value,
            loss_all_classes=all_classes_loss.item(),
            loss_challenge28=(
                challenge_loss.item()
                if challenge_loss is not None
                else None
            ),
        )
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


def _split_model_outputs(outputs):
    """Return all-class logits and optional native challenge logits."""
    if isinstance(outputs, dict):
        if "all_classes" not in outputs:
            raise KeyError("Model output is missing `all_classes` logits")
        return outputs["all_classes"], outputs.get("challenge28")
    return outputs, None


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
    label_space="",
):
    os.makedirs(output_dir, exist_ok=True)
    label_suffix = f"_{label_space}" if label_space else ""
    suffix = f"{mode}_{strategy}{label_suffix}"

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
        threshold_path = os.path.join(
            output_dir, f"thresholds{label_suffix}.json"
        )
        with open(threshold_path, "w", encoding="utf-8") as handle:
            json.dump(threshold_details, handle, indent=2)


def _project_challenge28(targets, probabilities, class_names):
    """Project all-class RFMiD outputs onto the 28-class challenge schema."""
    retained_indices, rare_indices = get_rfmid_challenge_indices(class_names)

    challenge_targets = np.concatenate(
        [
            targets[:, retained_indices],
            targets[:, rare_indices].max(axis=1, keepdims=True),
        ],
        axis=1,
    )
    challenge_probabilities = np.concatenate(
        [
            probabilities[:, retained_indices],
            probabilities[:, rare_indices].max(axis=1, keepdims=True),
        ],
        axis=1,
    )
    return (
        challenge_targets,
        challenge_probabilities,
        list(RFMID_CHALLENGE_CLASS_NAMES),
        [class_names[index] for index in rare_indices],
    )


def _compute_metrics(targets, probabilities, thresholds, loss=None):
    if probabilities.shape != targets.shape:
        raise ValueError(
            "Targets and probabilities must have the same shape, got "
            f"{targets.shape} and {probabilities.shape}"
        )
    if thresholds.shape != (targets.shape[1],):
        raise ValueError(
            f"Expected {targets.shape[1]} thresholds, got {thresholds.shape}"
        )

    predictions = (probabilities >= thresholds[None, :]).astype(np.int64)
    positive_labels = targets.sum(axis=0) > 0
    macro_targets = targets[:, positive_labels]
    macro_predictions = predictions[:, positive_labels]

    metrics = {}
    if loss is not None:
        metrics["loss"] = loss
    metrics.update(
        {
            "subset_accuracy": accuracy_score(targets, predictions),
            "f1_macro": f1_score(
                macro_targets,
                macro_predictions,
                average="macro",
                zero_division=0,
            ),
            "f1_micro": f1_score(
                targets, predictions, average="micro", zero_division=0
            ),
            "roc_auc_macro": _macro_auc(targets, probabilities)[0],
            "average_precision_macro": _macro_average_precision(
                targets, probabilities
            )[0],
            "hamming_loss": hamming_loss(targets, predictions),
            "jaccard_macro": jaccard_score(
                macro_targets,
                macro_predictions,
                average="macro",
                zero_division=0,
            ),
            "precision_macro": precision_score(
                macro_targets,
                macro_predictions,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                macro_targets,
                macro_predictions,
                average="macro",
                zero_division=0,
            ),
        }
    )
    metrics["score"] = np.nanmean(
        [
            metrics["f1_macro"],
            metrics["roc_auc_macro"],
            metrics["average_precision_macro"],
        ]
    )
    return metrics, predictions


def _print_metrics(
    mode,
    label_space,
    metrics,
    targets,
    probabilities,
    thresholds,
):
    auc_labels = _macro_auc(targets, probabilities)[1]
    ap_labels = _macro_average_precision(targets, probabilities)[1]
    prefix = f"{mode} {label_space}".strip()
    loss_text = (
        f" loss: {metrics['loss']:.4f},"
        if "loss" in metrics
        else ":"
    )
    print(
        f"{prefix}{loss_text} "
        f"macro F1: {metrics['f1_macro']:.4f}, "
        f"micro F1: {metrics['f1_micro']:.4f}, "
        f"macro AUROC: {metrics['roc_auc_macro']:.4f} "
        f"({auc_labels} labels), "
        f"macro AP: {metrics['average_precision_macro']:.4f} "
        f"({ap_labels} labels)"
    )
    print(
        f"subset accuracy: {metrics['subset_accuracy']:.4f}, "
        f"hamming loss: {metrics['hamming_loss']:.4f}, "
        f"macro precision: {metrics['precision_macro']:.4f}, "
        f"macro recall: {metrics['recall_macro']:.4f}"
    )
    print(
        f"thresholds: min={thresholds.min():.3f}, "
        f"mean={thresholds.mean():.3f}, max={thresholds.max():.3f}"
    )


def _write_metrics(output_dir, mode, strategy, metrics, label_space=""):
    label_suffix = f"_{label_space}" if label_space else ""
    results_path = os.path.join(
        output_dir, f"metrics_{mode}_{strategy}{label_suffix}.csv"
    )
    file_exists = os.path.isfile(results_path)
    with open(results_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


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
    challenge_thresholds=None,
    calibrate=False,
    export_details=False,
    write_metrics=True,
    quiet=False,
):
    """Evaluate sigmoid multi-label predictions."""
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()
    image_ids_all, targets_all, probabilities_all = [], [], []
    challenge_probabilities_all = []

    batches = (
        data_loader
        if quiet
        else metric_logger.log_every(data_loader, 10, f"{mode}:")
    )
    for batch in batches:
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
            logits, challenge_logits = _split_model_outputs(outputs)
            loss = criterion(logits, targets)

        metric_logger.update(loss=loss.item())
        targets_all.append(targets.cpu())
        probabilities_all.append(torch.sigmoid(logits).cpu())
        if challenge_logits is not None:
            challenge_probabilities_all.append(
                torch.sigmoid(challenge_logits).cpu()
            )

    metric_logger.synchronize_between_processes()
    targets = torch.cat(targets_all).numpy().astype(np.int64)
    probabilities = torch.cat(probabilities_all).numpy()
    native_challenge_probabilities = None
    if challenge_probabilities_all:
        native_challenge_probabilities = torch.cat(
            challenge_probabilities_all
        ).numpy()

    if misc.get_world_size() > 1 and args.dist_eval:
        local_result = (
            targets,
            probabilities,
            native_challenge_probabilities,
            image_ids_all,
        )
        gathered = [None] * misc.get_world_size()
        dist.all_gather_object(gathered, local_result)
        targets = np.concatenate([item[0] for item in gathered], axis=0)
        probabilities = np.concatenate(
            [item[1] for item in gathered], axis=0
        )
        if native_challenge_probabilities is not None:
            native_challenge_probabilities = np.concatenate(
                [item[2] for item in gathered], axis=0
            )
        image_ids_all = [
            image_id for item in gathered for image_id in item[3]
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

    loss = metric_logger.meters["loss"].global_avg
    metrics, predictions = _compute_metrics(
        targets, probabilities, thresholds, loss=loss
    )

    (
        challenge_targets,
        projected_challenge_probabilities,
        challenge_class_names,
        challenge_other_classes,
    ) = _project_challenge28(targets, probabilities, class_names)
    if native_challenge_probabilities is not None:
        challenge_probabilities = native_challenge_probabilities
        challenge_probability_source = "native_head"
    else:
        challenge_probabilities = projected_challenge_probabilities
        challenge_probability_source = "max_rare_probabilities"
    challenge_threshold_details = None
    if calibrate:
        challenge_thresholds, challenge_threshold_details = (
            calibrate_thresholds(
                challenge_targets,
                challenge_probabilities,
                challenge_class_names,
                args,
            )
        )
        challenge_threshold_details["label_space"] = "rfmid_challenge28"
        challenge_threshold_details["other_source_classes"] = (
            challenge_other_classes
        )
        challenge_threshold_details["probability_source"] = (
            challenge_probability_source
        )
    elif challenge_thresholds is None:
        challenge_thresholds = np.full(
            len(challenge_class_names), args.threshold, dtype=np.float64
        )
    challenge_thresholds = np.asarray(
        challenge_thresholds, dtype=np.float64
    )
    challenge_metrics, challenge_predictions = _compute_metrics(
        challenge_targets,
        challenge_probabilities,
        challenge_thresholds,
    )

    if misc.is_main_process():
        output_dir = os.path.join(args.output_dir, args.task)
        os.makedirs(output_dir, exist_ok=True)
        strategy = (
            args.threshold_strategy if calibrate or thresholds is not None
            else "fixed"
        )
        if not quiet:
            _print_metrics(
                mode, "", metrics, targets, probabilities, thresholds
            )
            _print_metrics(
                mode,
                "challenge28",
                challenge_metrics,
                challenge_targets,
                challenge_probabilities,
                challenge_thresholds,
            )

        if write_metrics:
            _write_metrics(output_dir, mode, strategy, metrics)
            _write_metrics(
                output_dir,
                mode,
                strategy,
                challenge_metrics,
                label_space="challenge28",
            )

        if log_writer:
            for metric_name, value in metrics.items():
                log_writer.add_scalar(f"perf/{metric_name}", value, epoch)
            for metric_name, value in challenge_metrics.items():
                log_writer.add_scalar(
                    f"perf_challenge28/{metric_name}", value, epoch
                )

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
            _write_detailed_outputs(
                output_dir,
                mode,
                strategy,
                image_ids_all,
                challenge_targets,
                challenge_probabilities,
                challenge_predictions,
                challenge_thresholds,
                challenge_class_names,
                challenge_threshold_details,
                label_space="challenge28",
            )

    return (
        metrics,
        metrics["score"],
        thresholds,
        challenge_thresholds,
        challenge_metrics,
    )
