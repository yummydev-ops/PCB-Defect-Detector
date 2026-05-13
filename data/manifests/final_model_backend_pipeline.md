# Final Model Backend Pipeline

This note describes the reusable backend inference and report-generation flow for the frozen final PCB defect model.

## Pipeline Flow

1. Load the frozen final-model reference from `configs/final_project_model.yaml`.
2. Accept one PCB image as input.
3. If the image is larger than the tile size, split it into overlapping tiles.
4. Run the final detection model on each tile.
5. Convert tile detections back into full-image coordinates.
6. Merge duplicate detections with simple class-wise IoU-based NMS.
7. Compute normalized severity for each merged detection.
8. Save a technician-friendly report as JSON, CSV, and a full-image annotated overlay.

## How Tiling Is Handled

- Default tile size: `640`
- Default overlap: `128`
- Images at or below the tile size run as a single tile.
- Larger images are covered with overlapping tiles so defects near tile borders are less likely to be missed.

## How Detections Are Merged

- Each tile prediction is shifted back into the original image coordinate space.
- Duplicates created by overlapping tiles are merged with a simple class-wise NMS step.
- Default merge IoU threshold: `0.30`

## How Severity Is Included

- Every merged detection keeps:
  - defect class
  - bounding box
  - confidence
  - severity score
  - severity band
  - image reference
- Severity uses the existing normalized heuristic from `scripts/severity_scoring.py`.

## Output Style

- `report.json`: full structured report for later UI/report logic
- `report.csv`: flat technician-friendly detection table
- `annotated_overlay.jpg`: full-image visual output with labels and severity

## Reuse Later

- The backend entry point is `scripts/final_model_backend.py`.
- It is designed to be called later by the Streamlit UI without changing the core inference logic.
