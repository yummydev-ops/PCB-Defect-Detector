#!/usr/bin/env python
"""
Convert DeepPCB annotations into a COCO-style dataset.

DeepPCB stores paired images and labels in a custom layout:
  - tested images:    PCBData/<group>/<id>/<sample>_test.jpg
  - template images:  PCBData/<group>/<id>/<sample>_temp.jpg
  - labels:           PCBData/<group>/<id>_not/<sample>.txt

Each label line is:
  x1 y1 x2 y2 class_id

For COCO export we use only the tested image because the bounding boxes belong
to the defective/tested board image. The template image remains in the source
dataset and is not copied into the COCO output.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2


CLASS_DEFINITIONS = [
    {"id": 1, "name": "open", "supercategory": "pcb_defect"},
    {"id": 2, "name": "short", "supercategory": "pcb_defect"},
    {"id": 3, "name": "mousebite", "supercategory": "pcb_defect"},
    {"id": 4, "name": "spur", "supercategory": "pcb_defect"},
    # DeepPCB README maps class id 5 to "copper" while the dataset description
    # refers to spurious copper. We keep the original README label here.
    {"id": 5, "name": "copper", "supercategory": "pcb_defect"},
    {"id": 6, "name": "pin-hole", "supercategory": "pcb_defect"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DeepPCB to COCO format.")
    parser.add_argument(
        "--deep-pcb-root",
        type=Path,
        default=Path("data/resources/DeepPCB-master"),
        help="Path to the DeepPCB-master folder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/coco_master/deeppcb"),
        help="Output folder for the converted COCO dataset.",
    )
    parser.add_argument(
        "--limit-trainval",
        type=int,
        default=None,
        help="Optional limit for the number of trainval samples to export.",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Optional limit for the number of test samples to export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output folder.",
    )
    return parser.parse_args()


def build_info(split_name: str, source_root: Path, sample_limit: Optional[int]) -> Dict[str, object]:
    limit_note = "full split export" if sample_limit is None else f"pilot export limited to {sample_limit} samples"
    return {
        "description": f"DeepPCB converted to COCO ({split_name}, {limit_note})",
        "version": "1.0",
        "year": datetime.now(timezone.utc).year,
        "date_created": datetime.now(timezone.utc).isoformat(),
        "contributor": "Codex conversion script",
        "source_root": str(source_root.as_posix()),
    }


def image_size(image_path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def parse_split_line(line: str) -> Tuple[str, str]:
    image_rel, label_rel = line.strip().split()
    return image_rel, label_rel


def resolve_test_image_path(pcb_data_root: Path, image_rel: str) -> Tuple[Path, str]:
    base_rel = Path(image_rel)
    base_stem = base_rel.stem
    group_dir = base_rel.parts[0]
    source_image_path = pcb_data_root / base_rel.parent / f"{base_stem}_test.jpg"
    if not source_image_path.exists():
        raise FileNotFoundError(f"Expected tested image not found: {source_image_path}")
    file_name = f"{group_dir}/{base_stem}_test.jpg"
    return source_image_path, file_name


def resolve_label_path(pcb_data_root: Path, label_rel: str) -> Path:
    label_path = pcb_data_root / label_rel
    if not label_path.exists():
        raise FileNotFoundError(f"Expected label file not found: {label_path}")
    return label_path


def coco_annotations_for_label_file(
    label_path: Path,
    image_id: int,
    annotation_start_id: int,
    width: int,
    height: int,
) -> Tuple[List[Dict[str, object]], int]:
    annotations: List[Dict[str, object]] = []
    next_annotation_id = annotation_start_id

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label line in {label_path}: {raw_line!r}")

        x1, y1, x2, y2, class_id = map(int, parts)
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width <= 0 or bbox_height <= 0:
            raise ValueError(f"Non-positive bbox in {label_path}: {raw_line!r}")
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(f"Out-of-bounds bbox in {label_path}: {raw_line!r}")

        annotations.append(
            {
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x1, y1, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0,
                "segmentation": [],
            }
        )
        next_annotation_id += 1

    return annotations, next_annotation_id


def convert_split(
    split_name: str,
    split_file: Path,
    pcb_data_root: Path,
    output_root: Path,
    limit: Optional[int],
) -> Dict[str, int]:
    image_output_dir = output_root / "images" / split_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    coco_images: List[Dict[str, object]] = []
    coco_annotations: List[Dict[str, object]] = []

    image_id = 1
    annotation_id = 1
    copied_images = 0

    lines = split_file.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[:limit]

    for line in lines:
        image_rel, label_rel = parse_split_line(line)
        source_image_path, coco_file_name = resolve_test_image_path(pcb_data_root, image_rel)
        label_path = resolve_label_path(pcb_data_root, label_rel)

        target_image_path = image_output_dir / coco_file_name
        target_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image_path, target_image_path)
        copied_images += 1

        width, height = image_size(source_image_path)
        coco_images.append(
            {
                "id": image_id,
                "file_name": coco_file_name.replace("\\", "/"),
                "width": width,
                "height": height,
            }
        )

        image_annotations, annotation_id = coco_annotations_for_label_file(
            label_path=label_path,
            image_id=image_id,
            annotation_start_id=annotation_id,
            width=width,
            height=height,
        )
        coco_annotations.extend(image_annotations)
        image_id += 1

    annotation_output_dir = output_root / "annotations"
    annotation_output_dir.mkdir(parents=True, exist_ok=True)
    annotation_output_path = annotation_output_dir / f"instances_{split_name}.json"

    coco_payload = {
        "info": build_info(split_name=split_name, source_root=pcb_data_root.parent, sample_limit=limit),
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": CLASS_DEFINITIONS,
    }
    annotation_output_path.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")

    return {
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "copied_images": copied_images,
    }


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output folder already exists and is not empty: {output_root}. "
            "Use --overwrite to allow reuse."
        )
    output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    pcb_data_root = args.deep_pcb_root / "PCBData"
    trainval_file = pcb_data_root / "trainval.txt"
    test_file = pcb_data_root / "test.txt"

    ensure_output_root(args.output_root, overwrite=args.overwrite)

    split_summaries = {
        "trainval": convert_split(
            split_name="trainval",
            split_file=trainval_file,
            pcb_data_root=pcb_data_root,
            output_root=args.output_root,
            limit=args.limit_trainval,
        ),
        "test": convert_split(
            split_name="test",
            split_file=test_file,
            pcb_data_root=pcb_data_root,
            output_root=args.output_root,
            limit=args.limit_test,
        ),
    }

    summary = {
        "output_root": str(args.output_root.resolve()),
        "splits": split_summaries,
        "categories": CLASS_DEFINITIONS,
        "mode": "pilot" if args.limit_trainval is not None or args.limit_test is not None else "full",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
