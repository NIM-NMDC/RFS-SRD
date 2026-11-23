"""Class-aware detection matching and 101-point interpolated AP."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image

CLASS_NAMES = ["Normal", "Deformed", "Inverted", "Fractured", "Missing", "Displaced"]
IOU_THRESHOLDS = [round(value, 2) for value in np.arange(0.50, 0.951, 0.05)]

def load_ground_truth(images_dir: Path, labels_dir: Path) -> dict[str, list[dict]]:
    result = {}
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        records = []
        label_path = labels_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, cx, cy, box_width, box_height = map(float, line.split())
            records.append(
                {
                    "class_id": int(class_id),
                    "xyxy": np.asarray(
                        [
                            (cx - box_width / 2) * width,
                            (cy - box_height / 2) * height,
                            (cx + box_width / 2) * width,
                            (cy + box_height / 2) * height,
                        ],
                        dtype=float,
                    ),
                }
            )
        result[image_path.name] = records
    return result


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def image_match_stats(
    ground_truth: dict[str, list[dict]],
    predictions: dict[str, list[dict]],
    class_id: int,
    iou_threshold: float,
) -> dict[str, dict]:
    result = {}
    for image_name, image_gt in ground_truth.items():
        gt_boxes = np.asarray(
            [record["xyxy"] for record in image_gt if record["class_id"] == class_id], dtype=float
        ).reshape(-1, 4)
        image_predictions = sorted(
            [record for record in predictions.get(image_name, []) if int(record["class_id"]) == class_id],
            key=lambda record: float(record["score"]),
            reverse=True,
        )
        matched = np.zeros(len(gt_boxes), dtype=bool)
        scores = np.asarray([float(record["score"]) for record in image_predictions], dtype=float)
        true_positive = np.zeros(len(image_predictions), dtype=np.int8)
        for index, record in enumerate(image_predictions):
            ious = box_iou(np.asarray(record["xyxy"], dtype=float), gt_boxes)
            for gt_index in np.argsort(-ious):
                if ious[gt_index] < iou_threshold:
                    break
                if not matched[gt_index]:
                    matched[gt_index] = True
                    true_positive[index] = 1
                    break
        result[image_name] = {"scores": scores, "tp": true_positive, "n_gt": len(gt_boxes)}
    return result


def average_precision(scores: np.ndarray, true_positive: np.ndarray, n_gt: int) -> float:
    if n_gt == 0 or scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tp = true_positive[order].astype(float)
    fp = 1.0 - tp
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    interpolated = [
        precision[recall >= point].max() if np.any(recall >= point) else 0.0
        for point in np.linspace(0.0, 1.0, 101)
    ]
    return float(np.mean(interpolated))


def aggregate(
    stats: dict[str, dict],
    image_keys: list[str],
    threshold: float | dict[str, float] | None,
) -> dict[str, float]:
    score_parts = []
    tp_parts = []
    selected_tp = 0
    selected_count = 0
    n_gt = 0
    for image_name in image_keys:
        item = stats[image_name]
        scores = item["scores"]
        true_positive = item["tp"]
        score_parts.append(scores)
        tp_parts.append(true_positive)
        n_gt += int(item["n_gt"])
        if threshold is not None:
            image_threshold = threshold[image_name] if isinstance(threshold, dict) else threshold
            selected = scores >= image_threshold
            selected_tp += int(true_positive[selected].sum())
            selected_count += int(selected.sum())
    scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=float)
    true_positive = np.concatenate(tp_parts) if tp_parts else np.empty(0, dtype=np.int8)
    tp = selected_tp
    fp = selected_count - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": average_precision(scores, true_positive, n_gt),
        "miss_rate": 1.0 - recall,
        "false_discovery_rate": 1.0 - precision if tp + fp else float("nan"),
        "fppi": fp / len(image_keys) if image_keys else 0.0,
    }


def load_payload(path: Path) -> tuple[dict[str, list[dict]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = {row["image"]: row["detections"] for row in payload["images"]}
    return predictions, payload
