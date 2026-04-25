#!/usr/bin/env python
"""
Colab-ready YOLOv8 detection training entry point for the DeepPCB baseline dataset.

Purpose:
- Keep the DeepPCB detection baseline separate from PKU.
- Reuse the converted DeepPCB COCO config already in this repo.
- Prepare a clean YOLO-format workspace for Ultralytics when needed.
- Launch pretrained YOLOv8 detection fine-tuning in Google Colab.

DeepPCB currently provides `trainval` and `test` COCO splits. For this baseline
workflow, `trainval` is used for training and `test` is reused as the validation
and inference-check split. This keeps the workflow simple without changing the
source dataset contents.

Typical Colab usage from the repo root:
    python scripts/deeppcb_yolov8_colab_train.py --prepare-only
    python scripts/deeppcb_yolov8_colab_train.py --epochs 10 --device 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pku_yolov8_smoke_setup import load_yaml, prepare_yolo_workspace, validate_source_config


DEFAULT_CONFIG = Path("configs/deeppcb_coco_baseline.yaml")
DEFAULT_WORKSPACE = Path("data/yolo_ready/deeppcb_yolov8_baseline")
DEFAULT_DATA_YAML = Path("configs/deeppcb_yolov8_baseline_data.yaml")
DEFAULT_PROJECT = Path("runs/deeppcb_baseline")
DEFAULT_NAME = "yolov8n_deeppcb_baseline_colab"
DEFAULT_MODEL = "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab-ready DeepPCB YOLOv8 baseline trainer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the DeepPCB baseline COCO config.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Separate YOLO-format workspace directory for DeepPCB.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Ultralytics data YAML to generate for DeepPCB training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Pretrained YOLOv8 detection weights to fine-tune.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Baseline epoch count for the first DeepPCB Colab run.",
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
        default=16,
        help="Batch size for Colab training.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device in Colab, e.g. 0 for GPU.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Ultralytics project directory for the DeepPCB baseline run.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help="Run name for the DeepPCB baseline training job.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training split to use. Keep 1.0 for the real baseline.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only validate the config and build the YOLO workspace without training.",
    )
    return parser.parse_args()


def prepare_runtime_dirs() -> None:
    repo_root = Path.cwd()
    yolo_config_dir = (repo_root / "UltralyticsConfig").resolve()
    mpl_config_dir = (repo_root / "MatplotlibConfig").resolve()
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def adapt_deeppcb_config(config: dict) -> dict:
    train_split = config["splits"]["trainval"]
    test_split = config["splits"]["test"]
    return {
        "dataset_name": config["dataset_name"],
        "format": config["format"],
        "dataset_root": config["dataset_root"],
        "class_names": config["class_names"],
        "source_category_ids": config["source_category_ids"],
        "splits": {
            "train": train_split,
            "val": test_split,
            "test": test_split,
        },
    }


def print_split_summary(split_info: dict) -> None:
    for split_name in ("train", "val", "test"):
        info = split_info[split_name]
        print(f"{split_name.upper()}_IMAGES={info['image_count']}")
        print(f"{split_name.upper()}_ANNOTATIONS={info['annotation_count']}")
        for issue_name, issue_value in info["issues"].items():
            print(f"{split_name.upper()}_{issue_name.upper()}={issue_value}")


def assert_clean_source(split_info: dict) -> None:
    issue_total = 0
    for split_name in ("train", "val", "test"):
        issue_total += sum(int(v) for v in split_info[split_name]["issues"].values())
    if issue_total:
        raise RuntimeError("DeepPCB config validation found dataset issues. Resolve them before Colab training.")


def run_training(args: argparse.Namespace, data_yaml_path: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=data_yaml_path.as_posix(),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project.as_posix(),
        name=args.name,
        fraction=args.fraction,
        patience=args.patience,
        pretrained=True,
        task="detect",
        cache=False,
        exist_ok=True,
        verbose=True,
    )


def main() -> None:
    args = parse_args()
    prepare_runtime_dirs()

    config = load_yaml(args.config)
    adapted_config = adapt_deeppcb_config(config)
    split_info = validate_source_config(adapted_config)
    assert_clean_source(split_info)
    prepare_yolo_workspace(adapted_config, split_info, args.workspace, args.data_yaml)

    print(f"CONFIG={args.config.resolve()}")
    print(f"WORKSPACE={args.workspace.resolve()}")
    print(f"DATA_YAML={args.data_yaml.resolve()}")
    print(f"MODEL={args.model}")
    print("TRAIN_SOURCE_SPLIT=trainval")
    print("VAL_SOURCE_SPLIT=test")
    print("TEST_SOURCE_SPLIT=test")
    print_split_summary(split_info)

    if args.prepare_only:
        print("STATUS=DeepPCB YOLO workspace and training data YAML are ready for Colab.")
        return

    run_training(args, args.data_yaml)


if __name__ == "__main__":
    main()
