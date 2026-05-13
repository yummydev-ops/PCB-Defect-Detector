# End-to-End System Smoke Test

- Input image: `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole04_y0_x1080_jpg.rf.79b3881c050909c662b3252c43b3536d.jpg`
- Final model used: `yolov8n_deeppcb_to_pcb_v1i_transfer_colab`
- Final checkpoint: `runs/detect/runs/pcb_v1i_transfer/yolov8n_deeppcb_to_pcb_v1i_transfer_colab/weights/best.pt`
- Backend file: `scripts/final_model_backend.py`
- Severity module: `scripts/severity_scoring.py`
- Streamlit UI file: `streamlit_app.py`
- Streamlit output root used in this test: `data/inspection_outputs/streamlit_runs`

## Output Artifacts

- Annotated overlay: `annotated_overlay.jpg`
- Structured JSON report: `report.json`
- Flat CSV report: `report.csv`

## Result

- The end-to-end flow completed successfully.
- Backend processing ran on the sample image, produced one merged detection, attached a normalized severity score, and wrote the expected overlay plus JSON/CSV report files into the Streamlit output path.
- This confirms the current project flow is working from image input through backend inference, severity scoring, report generation, and UI-compatible artifact output.
