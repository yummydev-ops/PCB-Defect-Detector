# PCB-Defect-Detection.v1i.coco Inspection

## Dataset Path

- `data/resources/PCB-Defect-Detection.v1i.coco`

## Structure

- `train/`
  - images
  - `_annotations.coco.json`
- `valid/`
  - images
  - `_annotations.coco.json`
- `test/`
  - images
  - `_annotations.coco.json`
- `README.dataset.txt`
- `README.roboflow.txt`

## Format

- Export style: `Roboflow COCO`
- Annotation format: `COCO JSON`
- Label type: `bounding boxes only`
- Segmentation masks: `not present`

Each split contains images plus a single `_annotations.coco.json`. No YOLO `.txt` label files were found.

## Split Counts

- `train`: `6537` images, `7398` annotations
- `valid`: `600` images, `674` annotations
- `test`: `300` images, `351` annotations

## Classes

- `1`: `missing_hole`
- `2`: `mouse_bite`
- `3`: `open_circuit`
- `4`: `short`
- `5`: `spur`
- `6`: `spurious_copper`

Note:
- Category `0` (`Missing_hole-Mouse_bite`) exists in the category list, but it is not used by the actual annotations in any split.

## Integrity Check

- All JSON image references resolve correctly.
- No extra image files were found outside the JSON listings.
- No invalid bbox shapes were found.
- No non-positive bbox sizes were found.
- `segmentation` fields are present in COCO records but are empty (`[]`), so this should be treated as a detection-only dataset.

## Practical Assessment

- The dataset looks cleaned and baseline-ready for object detection.
- The split structure is already suitable for training: `train`, `valid`, and `test` are present and usable as-is.
- The filenames and Roboflow export note show the dataset is already treated/augmented, not raw source data.

## Baseline Readiness

This dataset is ready for a baseline **detection** workflow without conversion.
