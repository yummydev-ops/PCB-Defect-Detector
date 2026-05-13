# Streamlit UI Note

This Streamlit app provides a technician-friendly front end for the frozen final PCB defect detector.

## What The UI Does

- Accepts one uploaded PCB image
- Runs the existing backend inference pipeline
- Displays the annotated detection overlay
- Shows a defect table with:
  - defect class
  - confidence
  - severity score
  - severity band
- Provides downloadable JSON and CSV reports

## What The User Uploads

- One PCB image in a common image format such as JPG or PNG

## What Outputs Are Produced

- Annotated full-image overlay
- Structured JSON report
- Flat CSV report

## How The Backend Is Called

- The UI imports and calls `run_backend_pipeline()` from `scripts/final_model_backend.py`
- The backend handles:
  - optional tiling
  - model inference
  - tile-merge logic
  - severity scoring
  - report generation

This keeps the UI thin and reusable while the backend remains the single source of inference logic.
