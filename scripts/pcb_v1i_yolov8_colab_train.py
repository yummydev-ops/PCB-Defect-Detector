#!/usr/bin/env python
"""
Colab-ready YOLOv8 detection baseline trainer for the cleaned PCB v1i dataset.

Purpose:
- Keep this cleaned dataset separate from PKU and DeepPCB.
- Reuse the validated COCO export already in this repo.
- Prepare a clean YOLO-format workspace for Ultralytics when needed.
- Launch pretrained YOLOv8 detection fine-tuning in Google Colab.
- Run a short post-training prediction check on test images.

Typical Colab usage from the repo root:
    python scripts/pcb_v1i_yolov8_colab_train.py --prepare-only
    python scripts/pcb_v1i_yolov8_colab_train.py --epochs 10 --device 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pku_yolov8_smoke_setup import load_yaml, prepare_yolo_workspace, validate_source_config


DEFAULT_CONFIG = Path("configs/pcb_v1i_coco_baseline.yaml")
DEFAULT_WORKSPACE = Path("data/yolo_ready/pcb_v1i_yolov8_baseline")
DEFAULT_DATA_YAML = Path("configs/pcb_v1i_yolov8_baseline_data.yaml")
DEFAULT_PROJECT = Path("runs/pcb_v1i_baseline")
DEFAULT_NAME = "yolov8n_pcb_v1i_baseline_colab"
DEFAULT_MODEL = "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab-ready YOLOv8 baseline trainer for the cleaned PCB v1i dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the cleaned PCB v1i COCO baseline config.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Separate YOLO-format workspace directory for the cleaned PCB v1i dataset.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Ultralytics data YAML to generate for training.",
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
        help="Baseline epoch count for the first cleaned-dataset Colab run.",
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
        help="Ultralytics project directory for this baseline run.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help="Run name for the cleaned-dataset baseline training job.",
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
        "--predict-samples",
        type=int,
        default=8,
        help="Number of test images to use for the post-training prediction check.",
    )
    parser.add_argument(
        "--predict-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for the post-training prediction check.",
    )
    parser.add_argument(
        "--predict-output-dir",
        type=Path,
        default=None,
        help="Optional folder name to use for saved prediction overlays. Defaults to data/inspection_outputs/<run_name>_predictions.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only validate the config and build the YOLO workspace without training.",
    )
    return parser.parse_args()


def resolve_predict_output_dir(args: argparse.Namespace) -> Path:
    if args.predict_output_dir is not None:
        return args.predict_output_dir
    return Path(f"data/inspection_outputs/{args.name}_predictions")


def prepare_runtime_dirs() -> None:
    repo_root = Path.cwd()
    yolo_config_dir = (repo_root / "UltralyticsConfig").resolve()
    mpl_config_dir = (repo_root / "MatplotlibConfig").resolve()
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


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
        raise RuntimeError("Cleaned PCB v1i config validation found dataset issues. Resolve them before Colab training.")


def choose_prediction_images(images_dir: Path, num_samples: int) -> list[Path]:
    image_paths = sorted(images_dir.glob("*.jpg"))
    image_paths.extend(sorted(images_dir.glob("*.jpeg")))
    image_paths.extend(sorted(images_dir.glob("*.png")))
    deduped = []
    seen = set()
    for path in image_paths:
        if path.name not in seen:
            deduped.append(path)
            seen.add(path.name)
    return deduped[:num_samples]


def run_training(args: argparse.Namespace, data_yaml_path: Path) -> Path:
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
    return Path(model.trainer.save_dir)


def run_post_training_check(args: argparse.Namespace, config: dict, weights_path: Path) -> Path:
    from ultralytics import YOLO

    test_images_dir = Path(config["splits"]["test"]["images_dir"]).resolve()
    sample_paths = choose_prediction_images(test_images_dir, args.predict_samples)
    if not sample_paths:
        raise RuntimeError(f"No test images found for prediction check in {test_images_dir}")

    requested_output_dir = resolve_predict_output_dir(args).resolve()
    requested_output_dir.parent.mkdir(parents=True, exist_ok=True)

    predictor = YOLO(str(weights_path))
    results = predictor.predict(
        source=[str(path) for path in sample_paths],
        save=True,
        save_txt=False,
        save_conf=True,
        imgsz=args.imgsz,
        conf=args.predict_conf,
        device=args.device,
        project=str(requested_output_dir.parent),
        name=requested_output_dir.name,
        exist_ok=True,
        verbose=False,
    )

    actual_output_dir = Path(results[0].save_dir) if results else requested_output_dir
    total_predictions = 0
    for result in results:
        if result.boxes is not None:
            total_predictions += len(result.boxes)

    print(f"PREDICT_OUTPUT_DIR={actual_output_dir}")
    print(f"PREDICT_IMAGES_CHECKED={len(sample_paths)}")
    print(f"PREDICT_TOTAL_BOXES={total_predictions}")
    return actual_output_dir


def main() -> None:
    args = parse_args()
    prepare_runtime_dirs()

    config = load_yaml(args.config)
    split_info = validate_source_config(config)
    assert_clean_source(split_info)
    prepare_yolo_workspace(config, split_info, args.workspace, args.data_yaml)

    print(f"CONFIG={args.config.resolve()}")
    print(f"WORKSPACE={args.workspace.resolve()}")
    print(f"DATA_YAML={args.data_yaml.resolve()}")
    print(f"MODEL={args.model}")
    print(f"RUN_NAME={args.name}")
    print_split_summary(split_info)

    if args.prepare_only:
        print("STATUS=Cleaned PCB v1i YOLO workspace and training data YAML are ready for Colab.")
        return

    run_dir = run_training(args, args.data_yaml)
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise RuntimeError(f"Expected best.pt was not created at {best_weights}")

    predict_output_dir = run_post_training_check(args, config, best_weights)

    print(f"RUN_DIR={run_dir}")
    print(f"BEST_WEIGHTS={best_weights}")
    print(f"LAST_WEIGHTS={last_weights}")
    print(f"PREDICTION_DIR={predict_output_dir}")


if __name__ == "__main__":
    main()
