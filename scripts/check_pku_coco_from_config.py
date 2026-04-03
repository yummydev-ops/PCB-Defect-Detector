#!/usr/bin/env python
"""
Minimal PKU COCO loading and visualization check driven by the existing config.

This is intended to be simple enough to run locally or in Google Colab after
mounting the project folder and changing into the repo directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


COLORS = {
    1: (0, 215, 255),    # missing_hole
    2: (255, 128, 0),    # mouse_bite
    3: (0, 255, 0),      # open_circuit
    4: (0, 0, 255),      # short
    5: (255, 0, 255),    # spur
    6: (255, 255, 0),    # spurious_copper
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PKU COCO loading and visualization check.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pku_coco_baseline.yaml"),
        help="Path to the PKU dataset config YAML.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to inspect. Default: train",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of train images to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/inspection_outputs/pku_baseline_check"),
        help="Folder to save rendered images and the validation note.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def load_config(config_path: Path) -> Dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def validate_split(coco_data: Dict, images_dir: Path, valid_category_ids: set[int]) -> Dict[str, int]:
    images = {img["id"]: img for img in coco_data["images"]}
    issues = {
        "missing_image_refs": 0,
        "missing_image_files": 0,
        "invalid_category_ids": 0,
        "malformed_bbox": 0,
        "nonpositive_bbox": 0,
        "out_of_bounds_bbox": 0,
        "empty_images": 0,
    }
    anns_by_image: Dict[int, List[Dict]] = {}

    for ann in coco_data["annotations"]:
        image_id = ann.get("image_id")
        if image_id not in images:
            issues["missing_image_refs"] += 1
            continue
        anns_by_image.setdefault(image_id, []).append(ann)

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

    for image_id, img in images.items():
        if image_id not in anns_by_image or not anns_by_image[image_id]:
            issues["empty_images"] += 1
        image_path = images_dir / img["file_name"]
        if not image_path.exists():
            issues["missing_image_files"] += 1

    return issues


def choose_samples(
    coco_data: Dict,
    category_name_to_id: Dict[str, int],
    num_samples: int,
) -> List[int]:
    images = {img["id"]: img for img in coco_data["images"]}
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in coco_data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    selected: List[int] = []
    selected_ids = set()
    used_original_names = set()

    for class_name, category_id in category_name_to_id.items():
        chosen = None
        for image_id in sorted(images):
            if image_id in selected_ids:
                continue
            anns = anns_by_image.get(image_id, [])
            if not any(ann["category_id"] == category_id for ann in anns):
                continue
            original_name = images[image_id].get("extra", {}).get("name", images[image_id]["file_name"])
            if original_name in used_original_names:
                continue
            chosen = image_id
            break
        if chosen is not None:
            selected.append(chosen)
            selected_ids.add(chosen)
            used_original_names.add(images[chosen].get("extra", {}).get("name", images[chosen]["file_name"]))
        if len(selected) >= num_samples:
            return selected

    remaining = sorted(
        [
            (image_id, len(anns_by_image.get(image_id, [])))
            for image_id in images
            if image_id not in selected_ids
        ],
        key=lambda item: (-item[1], images[item[0]].get("extra", {}).get("name", images[item[0]]["file_name"])),
    )

    for image_id, _ in remaining:
        original_name = images[image_id].get("extra", {}).get("name", images[image_id]["file_name"])
        if original_name in used_original_names:
            continue
        selected.append(image_id)
        selected_ids.add(image_id)
        used_original_names.add(original_name)
        if len(selected) >= num_samples:
            break

    return selected


def render_samples(
    coco_data: Dict,
    images_dir: Path,
    output_dir: Path,
    sample_ids: List[int],
    category_id_to_name: Dict[int, str],
) -> Tuple[List[Tuple[Path, str, int]], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = {img["id"]: img for img in coco_data["images"]}
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in coco_data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    rendered = []
    checked_annotations = 0

    for index, image_id in enumerate(sample_ids, start=1):
        img_meta = images[image_id]
        image_path = images_dir / img_meta["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        anns = anns_by_image.get(image_id, [])
        checked_annotations += len(anns)

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))
            category_id = ann["category_id"]
            label = category_id_to_name.get(category_id, f"class_{category_id}")
            color = COLORS.get(category_id, (220, 220, 220))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(0, y1 - th - baseline - 4)
            cv2.rectangle(image, (x1, top), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                image,
                label,
                (x1 + 3, max(th + 1, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        short_name = Path(img_meta["file_name"]).name
        out_path = output_dir / f"{index:02d}_{safe_name(short_name)}"
        cv2.imwrite(str(out_path), image)
        original_name = img_meta.get("extra", {}).get("name", img_meta["file_name"])
        rendered.append((out_path, original_name, len(anns)))

    return rendered, checked_annotations


def create_contact_sheet(rendered: List[Tuple[Path, str, int]], output_dir: Path) -> Path:
    thumb_w = 420
    thumb_h = 420
    caption_h = 34
    cols = 2
    rows = math.ceil(len(rendered) / cols)
    sheet = np.full((rows * (thumb_h + caption_h), cols * thumb_w, 3), 245, dtype=np.uint8)

    for i, (path, original_name, ann_count) in enumerate(rendered):
        row = i // cols
        col = i % cols
        image = cv2.imread(str(path))
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        y0 = row * (thumb_h + caption_h)
        x0 = col * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
        caption = f"{original_name} | anns={ann_count}"
        cv2.putText(sheet, caption[:52], (x0 + 8, y0 + thumb_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)

    contact_sheet_path = output_dir / "pku_baseline_contact_sheet.jpg"
    cv2.imwrite(str(contact_sheet_path), sheet)
    return contact_sheet_path


def write_validation_note(
    output_dir: Path,
    config_path: Path,
    split_name: str,
    images_checked: int,
    annotations_checked: int,
    issues: Dict[str, int],
    contact_sheet_path: Path,
) -> Path:
    note_path = output_dir / "validation_note.md"
    is_clean = all(value == 0 for value in issues.values())
    note = f"""# PKU Baseline Check

- Config used: `{config_path.as_posix()}`
- Split checked: `{split_name}`
- Images checked visually: {images_checked}
- Annotations checked visually: {annotations_checked}
- Contact sheet: `{contact_sheet_path.as_posix()}`
- Sample result: {'Clean sample with no obvious issues found.' if is_clean else 'Issues were detected in the sample/integrity check.'}

## Integrity Check

- missing_image_refs: {issues['missing_image_refs']}
- missing_image_files: {issues['missing_image_files']}
- invalid_category_ids: {issues['invalid_category_ids']}
- malformed_bbox: {issues['malformed_bbox']}
- nonpositive_bbox: {issues['nonpositive_bbox']}
- out_of_bounds_bbox: {issues['out_of_bounds_bbox']}
- empty_images: {issues['empty_images']}
"""
    note_path.write_text(note, encoding="utf-8")
    return note_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    split_cfg = config["splits"][args.split]
    images_dir = Path(split_cfg["images_dir"])
    annotations_path = Path(split_cfg["annotations"])
    output_dir = args.output_dir

    coco_data = json.loads(annotations_path.read_text(encoding="utf-8"))
    class_names = config["class_names"]
    source_category_ids = config["source_category_ids"]
    category_name_to_id = {name: source_category_ids[name] for name in class_names}
    category_id_to_name = {source_category_ids[name]: name for name in class_names}

    issues = validate_split(coco_data, images_dir, valid_category_ids=set(category_id_to_name))
    sample_ids = choose_samples(coco_data, category_name_to_id, args.num_samples)
    rendered, checked_annotations = render_samples(
        coco_data=coco_data,
        images_dir=images_dir,
        output_dir=output_dir,
        sample_ids=sample_ids,
        category_id_to_name=category_id_to_name,
    )
    contact_sheet_path = create_contact_sheet(rendered, output_dir)
    note_path = write_validation_note(
        output_dir=output_dir,
        config_path=args.config,
        split_name=args.split,
        images_checked=len(rendered),
        annotations_checked=checked_annotations,
        issues=issues,
        contact_sheet_path=contact_sheet_path,
    )

    print(f"OUTPUT_DIR={output_dir.resolve()}")
    print(f"CONTACT_SHEET={contact_sheet_path.resolve()}")
    print(f"VALIDATION_NOTE={note_path.resolve()}")
    print(f"IMAGES_CHECKED={len(rendered)}")
    print(f"ANNOTATIONS_CHECKED={checked_annotations}")
    for issue_name, issue_value in issues.items():
        print(f"{issue_name.upper()}={issue_value}")


if __name__ == "__main__":
    main()
