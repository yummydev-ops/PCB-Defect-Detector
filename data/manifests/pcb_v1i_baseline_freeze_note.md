# PCB v1i Baseline Freeze Note

## Baseline Locked

- Main detection baseline: `yolov8n_pcb_v1i_baseline_colab`
- Dataset: `PCB-Defect-Detection.v1i.coco`
- Model: `yolov8n.pt`
- Task: `detection`

## Final Baseline Metrics

- Precision: `0.97831`
- Recall: `0.97537`
- mAP50: `0.98915`
- mAP50-95: `0.57440`

## Status

- This cleaned-dataset YOLOv8n run remains the **frozen direct PCB v1i baseline** for comparison and reference.
- It is **not** the selected final project backbone anymore.
- The active final project model is `yolov8n_deeppcb_to_pcb_v1i_transfer_colab`, recorded in `data/manifests/final_project_model_selection.md`.
- No further retraining is planned on this baseline for now.
- The project should now move to the next **non-training** step using this run as the reference detector.
