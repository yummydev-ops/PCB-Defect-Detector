#!/usr/bin/env python
"""
Run the frozen final PCB detector and attach heuristic severity scores.

Purpose:
- Load the frozen transfer-learning checkpoint selected as the final project model.
- Run prediction on a small set of target-dataset test images.
- Save standard overlay images plus structured outputs that include severity.
- Keep severity scoring reproducible for later reporting and UI work.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import yaml

from severity_scoring import score_detection


DEFAULT_MODEL_REF_CONFIG = Path("configs/final_project_model.yaml")
DEFAULT_OUTPUT_DIR = Path("data/inspection_outputs/final_model_severity_check")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen final PCB model and attach heuristic severity scores.")
    parser.add_argument(
        "--model-ref-config",
        type=Path,
        default=DEFAULT_MODEL_REF_CONFIG,
        help="Path to the final project model reference config.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional override for the model checkpoint path.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Optional override for the target dataset config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional override for the selected run name.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of test images to run prediction on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder to save overlay images and structured severity outputs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for prediction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device, e.g. cpu or 0.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def choose_images(images_dir: Path, num_samples: int) -> List[Path]:
    image_paths = sorted(images_dir.rglob("*.jpg"))
    image_paths.extend(sorted(images_dir.rglob("*.jpeg")))
    image_paths.extend(sorted(images_dir.rglob("*.png")))
    deduped: List[Path] = []
    seen = set()
    for path in image_paths:
        if path.name not in seen:
            deduped.append(path)
            seen.add(path.name)
    return deduped[:num_samples]


def prepare_runtime_dirs() -> None:
    repo_root = Path.cwd()
    yolo_config_dir = (repo_root / "UltralyticsConfig").resolve()
    mpl_config_dir = (repo_root / "MatplotlibConfig").resolve()
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def build_detection_records(results, class_names: List[str]) -> Dict:
    summary = {
        "images_checked": len(results),
        "images_with_predictions": 0,
        "total_predictions": 0,
        "average_severity_score": 0.0,
        "predicted_class_counts": {name: 0 for name in class_names},
        "per_image": [],
    }
    flat_rows: List[Dict] = []
    severity_sum = 0.0

    for result in results:
        image_height, image_width = result.orig_shape
        image_record = {
            "image": Path(result.path).name,
            "image_width": image_width,
            "image_height": image_height,
            "prediction_count": 0,
            "detections": [],
        }
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            summary["images_with_predictions"] += 1
            for class_id, confidence, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                class_index = int(class_id)
                class_name = class_names[class_index]
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                severity = score_detection(class_name, float(confidence), (x1, y1, x2, y2), image_width, image_height)
                detection = {
                    "class_id": class_index,
                    "class_name": class_name,
                    "confidence": round(float(confidence), 4),
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "bbox_xywh": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                    **severity,
                }
                image_record["detections"].append(detection)
                flat_rows.append(
                    {
                        "image": image_record["image"],
                        "class_id": class_index,
                        "class_name": class_name,
                        "confidence": detection["confidence"],
                        "x1": detection["bbox_xyxy"][0],
                        "y1": detection["bbox_xyxy"][1],
                        "x2": detection["bbox_xyxy"][2],
                        "y2": detection["bbox_xyxy"][3],
                        "width": detection["bbox_xywh"][2],
                        "height": detection["bbox_xywh"][3],
                        "relative_area": detection["relative_area"],
                        "severity_score": detection["severity_score"],
                        "severity_band": detection["severity_band"],
                    }
                )
                summary["predicted_class_counts"][class_name] += 1
                summary["total_predictions"] += 1
                severity_sum += float(detection["severity_score"])

        image_record["prediction_count"] = len(image_record["detections"])
        summary["per_image"].append(image_record)

    if summary["total_predictions"] > 0:
        summary["average_severity_score"] = round(severity_sum / summary["total_predictions"], 4)
    return {"summary": summary, "rows": flat_rows}


def write_outputs(
    output_dir: Path,
    run_name: str,
    weights_path: Path,
    sample_paths: List[Path],
    payload: Dict,
) -> None:
    summary = payload["summary"]
    rows = payload["rows"]

    json_path = output_dir / "severity_predictions.json"
    csv_path = output_dir / "severity_predictions.csv"
    note_path = output_dir / "severity_check_note.md"

    export_payload = {
        "run_name": run_name,
        "weights": weights_path.as_posix(),
        "severity_range": [0.0, 1.0],
        "summary": summary,
    }
    json_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")

    fieldnames = [
        "image",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "relative_area",
        "severity_score",
        "severity_band",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Final Model Severity Check",
        "",
        f"- Run name: `{run_name}`",
        f"- Checkpoint used: `{weights_path.as_posix()}`",
        f"- Images checked: {summary['images_checked']}",
        f"- Images with predictions: {summary['images_with_predictions']}",
        f"- Total detections: {summary['total_predictions']}",
        f"- Average severity score: {summary['average_severity_score']}",
        f"- Output folder: `{output_dir.as_posix()}`",
        f"- JSON output: `{json_path.as_posix()}`",
        f"- CSV output: `{csv_path.as_posix()}`",
        "",
        "## Sample Images",
    ]
    lines.extend(f"- `{path.relative_to(Path.cwd()).as_posix()}`" for path in sample_paths)
    lines.extend(
        [
            "",
            "## Practical Note",
            "",
            "- Each detection record now includes class name, confidence, bounding box, relative box area, severity score, and severity band.",
        ]
    )
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare_runtime_dirs()

    model_ref = load_yaml(args.model_ref_config)
    dataset_config_path = Path(args.dataset_config or model_ref["target_dataset_config"])
    weights_path = Path(args.weights or model_ref["selected_checkpoint"]).resolve()
    run_name = args.run_name or model_ref["selected_run_name"]

    dataset_config = load_yaml(dataset_config_path)
    test_images_dir = Path(dataset_config["splits"]["test"]["images_dir"]).resolve()
    class_names = list(dataset_config["class_names"])

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Missing test images directory: {test_images_dir}")

    sample_paths = choose_images(test_images_dir, args.num_samples)
    if not sample_paths:
        raise RuntimeError(f"No test images found in {test_images_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    results = model.predict(
        source=[str(path) for path in sample_paths],
        save=True,
        save_txt=False,
        save_conf=True,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=False,
    )

    payload = build_detection_records(results, class_names)
    write_outputs(output_dir, run_name, weights_path, sample_paths, payload)

    summary = payload["summary"]
    print(f"OUTPUT_DIR={output_dir}")
    print(f"IMAGES_CHECKED={summary['images_checked']}")
    print(f"IMAGES_WITH_PREDICTIONS={summary['images_with_predictions']}")
    print(f"TOTAL_PREDICTIONS={summary['total_predictions']}")
    print(f"AVERAGE_SEVERITY={summary['average_severity_score']}")


if __name__ == "__main__":
    main()
