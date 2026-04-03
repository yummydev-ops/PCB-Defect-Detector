#!/usr/bin/env python
"""
Quick visual inference check for the PKU YOLOv8 smoke-test model.

Purpose:
- Load the saved smoke-test weights.
- Run inference on a few PKU validation images.
- Save predicted overlays into a dedicated inspection folder.
- Write a short validation note with basic prediction counts.

This is only a visual/runtime sanity check. It does not retrain the model or
modify the source dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import yaml


DEFAULT_CONFIG = Path("configs/pku_coco_baseline.yaml")
DEFAULT_WEIGHTS = Path("runs/detect/runs/pku_smoke/yolov8n_pku_smoke/weights/best.pt")
DEFAULT_OUTPUT_DIR = Path("data/inspection_outputs/pku_smoke_predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick PKU smoke-model inference visualization check.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the PKU baseline dataset config.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the saved smoke-test YOLOv8 weights.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of validation images to run prediction on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder to save prediction overlays and the validation note.",
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
        default=0.05,
        help="Confidence threshold for the smoke-test inspection.",
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
    image_paths = sorted(images_dir.glob("*.jpg"))
    return image_paths[:num_samples]


def prepare_runtime_dirs() -> None:
    repo_root = Path.cwd()
    yolo_config_dir = (repo_root / "UltralyticsConfig").resolve()
    mpl_config_dir = (repo_root / "MatplotlibConfig").resolve()
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def summarize_results(results, class_names: List[str]) -> Dict:
    summary = {
        "images_checked": len(results),
        "images_with_predictions": 0,
        "total_predictions": 0,
        "predicted_class_counts": {name: 0 for name in class_names},
        "per_image": [],
    }

    for result in results:
        boxes = result.boxes
        image_summary = {
            "image": Path(result.path).name,
            "prediction_count": 0,
            "predicted_classes": [],
        }
        if boxes is not None and len(boxes) > 0:
            summary["images_with_predictions"] += 1
            class_ids = [int(v) for v in boxes.cls.tolist()]
            image_summary["prediction_count"] = len(class_ids)
            image_summary["predicted_classes"] = [class_names[idx] for idx in class_ids]
            summary["total_predictions"] += len(class_ids)
            for idx in class_ids:
                summary["predicted_class_counts"][class_names[idx]] += 1
        summary["per_image"].append(image_summary)

    return summary


def write_note(output_dir: Path, weights_path: Path, source_paths: List[Path], summary: Dict, conf: float) -> None:
    note_path = output_dir / "validation_note.md"
    lines = [
        "# PKU Smoke-Test Inference Check",
        "",
        f"- Weights: `{weights_path.as_posix()}`",
        f"- Images checked: {summary['images_checked']}",
        f"- Images with predictions: {summary['images_with_predictions']}",
        f"- Total predicted boxes: {summary['total_predictions']}",
        f"- Confidence threshold: {conf}",
        "",
        "## Sample Images",
    ]
    lines.extend(f"- `{path.name}`" for path in source_paths)
    lines.extend(
        [
            "",
            "## Predicted Class Counts",
        ]
    )
    for class_name, count in summary["predicted_class_counts"].items():
        lines.append(f"- `{class_name}`: {count}")

    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path = output_dir / "prediction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare_runtime_dirs()

    config = load_yaml(args.config)
    weights_path = args.weights.resolve()
    val_images_dir = Path(config["splits"]["val"]["images_dir"]).resolve()
    class_names = list(config["class_names"])

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Missing validation images directory: {val_images_dir}")

    sample_paths = choose_images(val_images_dir, args.num_samples)
    if not sample_paths:
        raise RuntimeError(f"No validation images found in {val_images_dir}")

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

    summary = summarize_results(results, class_names)
    write_note(output_dir, weights_path, sample_paths, summary, args.conf)

    print(f"OUTPUT_DIR={output_dir}")
    print(f"IMAGES_CHECKED={summary['images_checked']}")
    print(f"IMAGES_WITH_PREDICTIONS={summary['images_with_predictions']}")
    print(f"TOTAL_PREDICTIONS={summary['total_predictions']}")
    for class_name, count in summary["predicted_class_counts"].items():
        print(f"PREDICTED_{class_name.upper()}={count}")


if __name__ == "__main__":
    main()
