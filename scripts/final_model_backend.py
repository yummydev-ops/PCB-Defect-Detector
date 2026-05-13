#!/usr/bin/env python
"""
Reusable backend inference and report-generation pipeline for the final PCB model.

This module is designed to be imported later by a Streamlit UI, while still
providing a simple CLI for local verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import yaml

from severity_scoring import score_detection


DEFAULT_MODEL_REF_CONFIG = Path("configs/final_project_model.yaml")
DEFAULT_OUTPUT_ROOT = Path("data/inspection_outputs/final_model_backend")
DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 128
DEFAULT_MERGE_IOU = 0.30


@dataclass
class TileInfo:
    tile_id: str
    x_offset: int
    y_offset: int
    width: int
    height: int
    image: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reusable backend inference pipeline for the final PCB model.")
    parser.add_argument("--image", type=Path, required=True, help="Path to one PCB image for backend inference.")
    parser.add_argument(
        "--model-ref-config",
        type=Path,
        default=DEFAULT_MODEL_REF_CONFIG,
        help="Path to the frozen final-model reference config.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional override for the final model checkpoint path.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Optional override for the target dataset config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated backend outputs.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device, e.g. cpu or 0.")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Tile size for large-image inference. Images at or below this size run as a single tile.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=DEFAULT_TILE_OVERLAP,
        help="Tile overlap in pixels for large-image inference.",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=DEFAULT_MERGE_IOU,
        help="IoU threshold for merging duplicate detections after tiling.",
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


def load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)


def compute_tile_positions(full_size: int, tile_size: int, overlap: int) -> List[int]:
    if full_size <= tile_size:
        return [0]
    stride = max(tile_size - overlap, 1)
    positions = list(range(0, full_size - tile_size + 1, stride))
    last_start = full_size - tile_size
    if positions[-1] != last_start:
        positions.append(last_start)
    return sorted(set(positions))


def split_into_tiles(image_bgr, tile_size: int, overlap: int) -> List[TileInfo]:
    image_height, image_width = image_bgr.shape[:2]
    x_positions = compute_tile_positions(image_width, tile_size, overlap)
    y_positions = compute_tile_positions(image_height, tile_size, overlap)

    tiles: List[TileInfo] = []
    for row_index, y_offset in enumerate(y_positions):
        for col_index, x_offset in enumerate(x_positions):
            tile = image_bgr[y_offset : y_offset + tile_size, x_offset : x_offset + tile_size].copy()
            tile_height, tile_width = tile.shape[:2]
            tile_id = f"tile_r{row_index:02d}_c{col_index:02d}"
            tiles.append(
                TileInfo(
                    tile_id=tile_id,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    width=tile_width,
                    height=tile_height,
                    image=tile,
                )
            )
    return tiles


def xyxy_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def merge_detections_with_nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    merged: List[Dict] = []
    class_groups: Dict[int, List[Dict]] = {}
    for detection in detections:
        class_groups.setdefault(int(detection["class_id"]), []).append(detection)

    for _, group in class_groups.items():
        remaining = sorted(group, key=lambda item: float(item["confidence"]), reverse=True)
        while remaining:
            current = remaining.pop(0)
            merged.append(current)
            survivors: List[Dict] = []
            for candidate in remaining:
                if xyxy_iou(current["bbox_xyxy"], candidate["bbox_xyxy"]) <= iou_threshold:
                    survivors.append(candidate)
            remaining = survivors
    return sorted(merged, key=lambda item: float(item["confidence"]), reverse=True)


def collect_tile_detections(results, tiles: Sequence[TileInfo], class_names: List[str], image_width: int, image_height: int) -> List[Dict]:
    detections: List[Dict] = []
    for result, tile in zip(results, tiles):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for class_id, confidence, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            full_x1 = x1 + tile.x_offset
            full_y1 = y1 + tile.y_offset
            full_x2 = x2 + tile.x_offset
            full_y2 = y2 + tile.y_offset
            class_index = int(class_id)
            class_name = class_names[class_index]
            severity = score_detection(
                class_name,
                float(confidence),
                (full_x1, full_y1, full_x2, full_y2),
                image_width,
                image_height,
            )
            detections.append(
                {
                    "tile_id": tile.tile_id,
                    "class_id": class_index,
                    "class_name": class_name,
                    "confidence": round(float(confidence), 4),
                    "bbox_xyxy": [round(full_x1, 2), round(full_y1, 2), round(full_x2, 2), round(full_y2, 2)],
                    "bbox_xywh": [
                        round(full_x1, 2),
                        round(full_y1, 2),
                        round(max(0.0, full_x2 - full_x1), 2),
                        round(max(0.0, full_y2 - full_y1), 2),
                    ],
                    **severity,
                }
            )
    return detections


def draw_overlay(image_bgr, detections: Sequence[Dict], output_path: Path) -> None:
    canvas = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox_xyxy"]]
        label = (
            f"{detection['class_name']} "
            f"conf={detection['confidence']:.2f} "
            f"sev={detection['severity_score']:.2f} "
            f"({detection['severity_band']})"
        )
        color = (0, 0, 255) if detection["severity_band"] == "high" else (0, 165, 255) if detection["severity_band"] == "medium" else (0, 200, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text_origin = (x1, max(20, y1 - 10))
        cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def export_structured_outputs(
    output_dir: Path,
    image_path: Path,
    run_name: str,
    weights_path: Path,
    tiling_info: Dict,
    detections: Sequence[Dict],
) -> Dict[str, Path]:
    json_path = output_dir / "report.json"
    csv_path = output_dir / "report.csv"

    summary = {
        "image_reference": image_path.as_posix(),
        "run_name": run_name,
        "weights": weights_path.as_posix(),
        "tile_count": tiling_info["tile_count"],
        "used_tiling": tiling_info["used_tiling"],
        "detections": len(detections),
    }
    json_payload = {
        "summary": summary,
        "tiling": tiling_info,
        "detections": list(detections),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    fieldnames = [
        "image_reference",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "severity_score",
        "severity_band",
        "relative_area",
        "tile_id",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for detection in detections:
            writer.writerow(
                {
                    "image_reference": image_path.as_posix(),
                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "x1": detection["bbox_xyxy"][0],
                    "y1": detection["bbox_xyxy"][1],
                    "x2": detection["bbox_xyxy"][2],
                    "y2": detection["bbox_xyxy"][3],
                    "width": detection["bbox_xywh"][2],
                    "height": detection["bbox_xywh"][3],
                    "severity_score": detection["severity_score"],
                    "severity_band": detection["severity_band"],
                    "relative_area": detection["relative_area"],
                    "tile_id": detection["tile_id"],
                }
            )
    return {"json": json_path, "csv": csv_path}


def run_backend_pipeline(
    image_path: Path,
    model_ref_config: Path = DEFAULT_MODEL_REF_CONFIG,
    weights: Path | None = None,
    dataset_config: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cpu",
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    merge_iou: float = DEFAULT_MERGE_IOU,
) -> Dict[str, object]:
    prepare_runtime_dirs()

    model_ref = load_yaml(model_ref_config)
    dataset_config_path = Path(dataset_config or model_ref["target_dataset_config"])
    dataset_info = load_yaml(dataset_config_path)
    weights_path = Path(weights or model_ref["selected_checkpoint"]).resolve()
    run_name = model_ref["selected_run_name"]
    class_names = list(dataset_info["class_names"])

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    image_height, image_width = image_bgr.shape[:2]

    tiles = split_into_tiles(image_bgr, tile_size=tile_size, overlap=tile_overlap)
    used_tiling = len(tiles) > 1

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    results = model.predict(
        source=[tile.image for tile in tiles],
        save=False,
        save_txt=False,
        save_conf=False,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )

    raw_detections = collect_tile_detections(results, tiles, class_names, image_width, image_height)
    merged_detections = merge_detections_with_nms(raw_detections, iou_threshold=merge_iou)

    output_dir = ensure_dir(output_root / sanitize_stem(image_path))
    overlay_path = output_dir / "annotated_overlay.jpg"
    draw_overlay(image_bgr, merged_detections, overlay_path)

    tiling_info = {
        "used_tiling": used_tiling,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_count": len(tiles),
        "merge_iou_threshold": merge_iou,
        "tiles": [
            {
                "tile_id": tile.tile_id,
                "x_offset": tile.x_offset,
                "y_offset": tile.y_offset,
                "width": tile.width,
                "height": tile.height,
            }
            for tile in tiles
        ],
    }
    exported = export_structured_outputs(output_dir, image_path.resolve(), run_name, weights_path, tiling_info, merged_detections)

    return {
        "output_dir": output_dir,
        "overlay_path": overlay_path,
        "json_path": exported["json"],
        "csv_path": exported["csv"],
        "run_name": run_name,
        "weights_path": weights_path,
        "used_tiling": used_tiling,
        "tile_count": len(tiles),
        "detection_count": len(merged_detections),
    }


def main() -> None:
    args = parse_args()
    result = run_backend_pipeline(
        image_path=args.image.resolve(),
        model_ref_config=args.model_ref_config,
        weights=args.weights,
        dataset_config=args.dataset_config,
        output_root=args.output_root.resolve(),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        merge_iou=args.merge_iou,
    )
    print(f"OUTPUT_DIR={result['output_dir']}")
    print(f"ANNOTATED_IMAGE={result['overlay_path']}")
    print(f"JSON_REPORT={result['json_path']}")
    print(f"CSV_REPORT={result['csv_path']}")
    print(f"USED_TILING={result['used_tiling']}")
    print(f"TILE_COUNT={result['tile_count']}")
    print(f"DETECTION_COUNT={result['detection_count']}")


if __name__ == "__main__":
    main()
