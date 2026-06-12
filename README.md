# PCB-Defect-Detector

A practical PCB defect detector built around a frozen YOLOv8 detection backbone, reusable backend inference pipeline, normalized severity scoring, and a technician-friendly Streamlit interface.

## Current Working System

The stable project backbone is the transfer-learning checkpoint:

- `yolov8n_deeppcb_to_pcb_v1i_transfer_colab`
- checkpoint: `runs/detect/runs/pcb_v1i_transfer/yolov8n_deeppcb_to_pcb_v1i_transfer_colab/weights/best.pt`

The current system can:

- accept one PCB image
- tile large images before inference when needed
- run the frozen final detector
- merge tile detections back into full-image coordinates
- compute normalized severity scores
- generate technician-friendly JSON and CSV reports
- display annotated results in Streamlit

## Main Project Files

- Final model reference: `configs/final_project_model.yaml`
- Backend pipeline: `scripts/final_model_backend.py`
- Severity scoring: `scripts/severity_scoring.py`
- Streamlit app: `streamlit_app.py`

## Outputs

Each run can produce:

- annotated image overlay
- JSON report
- CSV report
- severity score and severity band for each detection

Typical output locations:

- Streamlit runs: `data/inspection_outputs/streamlit_runs`
- Direct backend runs: `data/inspection_outputs/final_model_backend`

## Quick Start

From the repo root:

```powershell
python -m streamlit run streamlit_app.py
```

Then:

1. Upload one PCB image
2. Click `Run Detection`
3. Review the annotated image, defect table, and downloadable reports

## Documentation Notes

Short project notes for the completed system are stored under `data/manifests/`, including:

- final model selection
- backend pipeline flow
- severity scoring method
- Streamlit UI note
- end-to-end smoke test evidence
