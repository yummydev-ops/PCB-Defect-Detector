# PCB v1i Baseline Run Summary

This note records the first YOLOv8 detection baseline workflow for the cleaned `PCB-Defect-Detection.v1i.coco` dataset.

## Run Identity

- Dataset: `PCB-Defect-Detection.v1i.coco`
- Model: `yolov8n.pt`
- Task: `detection`
- Training style: `fine-tuning from pretrained weights`
- Runner: `Google Colab`
- Script: `scripts/pcb_v1i_yolov8_colab_train.py`
- COCO config: `configs/pcb_v1i_coco_baseline.yaml`
- Generated data YAML: `configs/pcb_v1i_yolov8_baseline_data.yaml`
- Run name: `yolov8n_pcb_v1i_baseline_colab`

## Split Usage

- Training split: `train`
- Validation split: `valid`
- Post-training inference-check split: `test`

## Baseline Settings

- Default epochs: `10`
- Default image size: `640`
- Default batch size: `16`
- Default workers: `2`
- Default device: `0`
- Default fraction: `1.0`
- Default patience: `20`

## Main Artifacts

- YOLO workspace: `data/yolo_ready/pcb_v1i_yolov8_baseline`
- Run folder: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab`
- Best weights: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab/weights/best.pt`
- Last weights: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab/weights/last.pt`
- Metrics CSV: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab/results.csv`
- Run args: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab/args.yaml`
- Prediction images: `data/inspection_outputs/yolov8n_pcb_v1i_baseline_colab_predictions`

## Final Metrics

- Final epoch: `10`
- Train box loss: `1.43074`
- Train cls loss: `0.76408`
- Train dfl loss: `1.35441`
- Validation box loss: `1.44712`
- Validation cls loss: `0.61272`
- Validation dfl loss: `1.40394`
- Precision: `0.97831`
- Recall: `0.97537`
- mAP50: `0.98915`
- mAP50-95: `0.57440`

## Best Recorded Metrics

- Best mAP50: `0.98915`
- Best mAP50 epoch: `10`
- Best mAP50-95: `0.57440`
- Best mAP50-95 epoch: `10`

## Practical Reading

- This workflow keeps the cleaned PCB v1i dataset separate from PKU and DeepPCB.
- It uses the existing train/valid/test split directly, without conversion.
- The first goal was a clean detection-only reference baseline before any later training changes.
- The 10-epoch Colab baseline completed successfully and saved the expected training artifacts.
- Detection performance is already strong on this cleaned dataset, especially compared with the older PKU baseline.
