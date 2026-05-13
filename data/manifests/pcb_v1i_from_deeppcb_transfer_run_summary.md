# PCB v1i From DeepPCB Transfer Run Summary

This note records the completed **sequential fine-tuning workflow** that starts from the completed DeepPCB detector and adapts it to the cleaned `PCB-Defect-Detection.v1i.coco` dataset.

## Run Identity

- Purpose: `transfer learning from DeepPCB to PCB v1i`
- Source checkpoint: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/weights/best.pt`
- Target dataset: `PCB-Defect-Detection.v1i.coco`
- Task: `detection`
- Training style: `sequential fine-tuning`
- Runner: `Google Colab`
- Script: `scripts/pcb_v1i_yolov8_colab_train.py`
- Notebook: `notebook/pcb_v1i_from_deeppcb_transfer_colab.ipynb`
- COCO config: `configs/pcb_v1i_coco_baseline.yaml`
- Generated data YAML: `configs/pcb_v1i_yolov8_baseline_data.yaml`
- Run name: `yolov8n_deeppcb_to_pcb_v1i_transfer_colab`

## Split Usage

- Training split: `train`
- Validation split: `valid`
- Post-training inference-check split: `test`

## Training Settings

- Default source model path: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/weights/best.pt`
- Default epochs: `10`
- Default image size: `640`
- Default batch size: `16`
- Default workers: `2`
- Default device: `0`
- Default fraction: `1.0`
- Default patience: `20`

## Main Artifacts

- YOLO workspace: `data/yolo_ready/pcb_v1i_yolov8_baseline`
- Run folder: `runs/detect/runs/pcb_v1i_transfer/yolov8n_deeppcb_to_pcb_v1i_transfer_colab`
- Best weights: `runs/detect/runs/pcb_v1i_transfer/yolov8n_deeppcb_to_pcb_v1i_transfer_colab/weights/best.pt`
- Last weights: `runs/detect/runs/pcb_v1i_transfer/yolov8n_deeppcb_to_pcb_v1i_transfer_colab/weights/last.pt`
- Metrics CSV: `runs/detect/runs/pcb_v1i_transfer/yolov8n_deeppcb_to_pcb_v1i_transfer_colab/results.csv`
- Run args: `runs/detect/runs/pcb_v1i_transfer/yolov8n_deeppcb_to_pcb_v1i_transfer_colab/args.yaml`
- Prediction images: `data/inspection_outputs/yolov8n_deeppcb_to_pcb_v1i_transfer_colab_predictions`

## Final Metrics

- Final epoch: `10`
- Train box loss: `1.43224`
- Train cls loss: `0.76498`
- Train dfl loss: `1.26339`
- Validation box loss: `1.47127`
- Validation cls loss: `0.60985`
- Validation dfl loss: `1.30848`
- Precision: `0.97514`
- Recall: `0.98006`
- mAP50: `0.98884`
- mAP50-95: `0.56531`

## Best Recorded Metrics

- Best mAP50: `0.99008`
- Best mAP50 epoch: `9`
- Best mAP50-95: `0.56770`
- Best mAP50-95 epoch: `9`

## Practical Reading

- This run does **not** merge DeepPCB and PCB v1i into one dataset.
- The DeepPCB checkpoint is used only as the starting detector before adapting to the cleaned PCB v1i dataset.
- The final class space remains the PCB v1i classes.
- The transfer run completed successfully and saved the expected training artifacts plus eight post-training prediction overlays.
- Detection performance remained strong on the target dataset after transfer.
