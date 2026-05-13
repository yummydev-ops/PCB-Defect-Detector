# Final Project Model Selection

## Selected Backbone

- Final project model: `yolov8n_deeppcb_to_pcb_v1i_transfer_colab`
- Backbone family: `yolov8n.pt`
- Task: `detection`
- Status: `frozen for downstream project work`

## Why This Model Was Chosen

- It follows the final project direction of **sequential fine-tuning**:
  - start from the trained DeepPCB detector
  - continue training on the cleaned `PCB-Defect-Detection.v1i.coco` dataset
- It gives the project one practical end-to-end backbone that carries both:
  - earlier defect knowledge from DeepPCB
  - final adaptation to the cleaned target dataset used for the project system

## Transfer Path

- Source checkpoint: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/weights/best.pt`
- Target dataset: `PCB-Defect-Detection.v1i.coco`
- Target dataset config: `configs/pcb_v1i_coco_baseline.yaml`
- Final selected checkpoint: `runs/detect/runs/pcb_v1i_transfer/yolov8n_deeppcb_to_pcb_v1i_transfer_colab/weights/best.pt`

## Final Metrics

- Precision: `0.97514`
- Recall: `0.98006`
- mAP50: `0.98884`
- mAP50-95: `0.56531`

## Practical Note

- No further retraining is planned on this model for now.
- Future severity scoring, reporting, and web UI steps should treat this checkpoint as the default project backbone.
