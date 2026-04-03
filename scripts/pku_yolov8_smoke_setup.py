#!/usr/bin/env python
"""
Minimal YOLOv8 smoke-test setup for the PKU COCO baseline dataset.

Purpose:
- Read the existing configs/pku_coco_baseline.yaml file.
- Validate the PKU COCO train/val/test split paths.
- Prepare a separate YOLO-format workspace for Ultralytics when requested.
- Provide an optional 1-epoch smoke-test training entry point for Colab.

This does not modify the source dataset. Any YOLO-ready files are written into a
separate workspace directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml


DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_WORKSPACE = Path("data/yolo_ready/pku_yolov8_smoke")
DEFAULT_DATA_YAML = Path("configs/pku_yolov8_smoke_data.yaml")
DEFAULT_PROJECT = Path("runs/pku_smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a minimal YOLOv8 smoke-test setup for PKU COCO.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pku_coco_baseline.yaml"),
        help="Path to the existing PKU baseline config.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Separate YOLO-format workspace directory.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Ultralytics data YAML to generate for smoke tests.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Pretrained YOLOv8 detection weights to use for the smoke test.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Smoke-test epoch count. Keep this small.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for the smoke test.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader workers for the smoke test.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.02,
        help="Fraction of the training dataset to use for the smoke test.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device for Colab smoke tests, e.g. 0 or cpu.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Ultralytics runs/project directory for the smoke test.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n_pku_smoke",
        help="Ultralytics run name for the smoke test.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create the separate YOLO-format workspace now.",
    )
    parser.add_argument(
        "--run-smoke-test",
        action="store_true",
        help="Run the 1-epoch training smoke test after preparation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate config and split paths without writing files or training.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_coco(annotation_path: Path) -> Dict:
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def coco_bbox_to_yolo(bbox: List[float], width: int, height: int) -> List[float]:
    x, y, w, h = bbox
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    return [cx, cy, w / width, h / height]


def validate_source_config(config: Dict) -> Dict[str, Dict[str, object]]:
    split_info = {}
    for split_name in ("train", "val", "test"):
        split_cfg = config["splits"][split_name]
        images_dir = Path(split_cfg["images_dir"])
        annotation_path = Path(split_cfg["annotations"])
        coco = load_coco(annotation_path)
        images = {img["id"]: img for img in coco["images"]}
        anns_by_image = defaultdict(list)
        issues = {
            "missing_image_refs": 0,
            "missing_image_files": 0,
            "invalid_category_ids": 0,
            "malformed_bbox": 0,
            "nonpositive_bbox": 0,
            "out_of_bounds_bbox": 0,
        }

        valid_category_ids = set(config["source_category_ids"].values())

        for ann in coco["annotations"]:
            image_id = ann.get("image_id")
            if image_id not in images:
                issues["missing_image_refs"] += 1
                continue
            anns_by_image[image_id].append(ann)
            category_id = ann.get("category_id")
            if category_id not in valid_category_ids:
                issues["invalid_category_ids"] += 1
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                issues["malformed_bbox"] += 1
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                issues["nonpositive_bbox"] += 1
            img = images[image_id]
            if x < 0 or y < 0 or (x + w) > img["width"] or (y + h) > img["height"]:
                issues["out_of_bounds_bbox"] += 1

        for img in coco["images"]:
            image_path = images_dir / img["file_name"]
            if not image_path.exists():
                issues["missing_image_files"] += 1

        split_info[split_name] = {
            "images_dir": images_dir,
            "annotation_path": annotation_path,
            "coco": coco,
            "issues": issues,
            "image_count": len(coco["images"]),
            "annotation_count": len(coco["annotations"]),
        }
    return split_info


def prepare_yolo_workspace(config: Dict, split_info: Dict[str, Dict[str, object]], workspace: Path, data_yaml_path: Path) -> None:
    class_names = config["class_names"]
    source_category_ids = config["source_category_ids"]
    category_id_to_index = {source_category_ids[name]: idx for idx, name in enumerate(class_names)}

    for split_name in ("train", "val", "test"):
        info = split_info[split_name]
        images_dir = info["images_dir"]
        coco = info["coco"]
        anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        target_image_dir = workspace / "images" / split_name
        target_label_dir = workspace / "labels" / split_name
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        for img in coco["images"]:
            source_image_path = images_dir / img["file_name"]
            target_image_path = target_image_dir / Path(img["file_name"]).name
            link_or_copy(source_image_path, target_image_path)

            label_lines = []
            for ann in anns_by_image.get(img["id"], []):
                category_id = ann["category_id"]
                if category_id not in category_id_to_index:
                    continue
                cx, cy, w, h = coco_bbox_to_yolo(ann["bbox"], width=img["width"], height=img["height"])
                label_lines.append(f"{category_id_to_index[category_id]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            label_path = target_label_dir / f"{target_image_path.stem}.txt"
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    ensure_parent(data_yaml_path)
    yolo_yaml = {
        "path": workspace.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    data_yaml_path.write_text(yaml.safe_dump(yolo_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_smoke_test(args: argparse.Namespace, data_yaml_path: Path) -> None:
    repo_yolo_config = Path.cwd() / "UltralyticsConfig"
    repo_yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(repo_yolo_config.resolve()))

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=data_yaml_path.as_posix(),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        fraction=args.fraction,
        project=args.project.as_posix(),
        name=args.name,
        device=args.device,
        pretrained=True,
        task="detect",
        cache=False,
        exist_ok=True,
        verbose=True,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    split_info = validate_source_config(config)

    should_prepare = args.prepare or args.run_smoke_test

    if should_prepare and not args.dry_run:
        prepare_yolo_workspace(config, split_info, args.workspace, args.data_yaml)

    print(f"CONFIG={args.config.resolve()}")
    print(f"MODEL={args.model}")
    print(f"WORKSPACE={args.workspace.resolve()}")
    print(f"DATA_YAML={args.data_yaml.resolve()}")
    for split_name in ("train", "val", "test"):
        info = split_info[split_name]
        print(f"{split_name.upper()}_IMAGES={info['image_count']}")
        print(f"{split_name.upper()}_ANNOTATIONS={info['annotation_count']}")
        for issue_name, issue_value in info["issues"].items():
            print(f"{split_name.upper()}_{issue_name.upper()}={issue_value}")

    if args.dry_run:
        print("STATUS=Dry run completed. Config and split paths resolved successfully.")
        return

    if should_prepare:
        print("STATUS=YOLO workspace prepared and smoke-test dataset YAML is ready.")
    else:
        print("STATUS=Setup file is ready. Run with --prepare to build the YOLO workspace.")

    if args.run_smoke_test:
        run_smoke_test(args, args.data_yaml)


if __name__ == "__main__":
    main()
