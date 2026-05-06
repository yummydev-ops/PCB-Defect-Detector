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

## Expected Artifacts

- YOLO workspace: `data/yolo_ready/pcb_v1i_yolov8_baseline`
- Run folder: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab`
- Best weights: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab/weights/best.pt`
- Last weights: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab/weights/last.pt`
- Metrics CSV: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab/results.csv`
- Run args: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab/args.yaml`
- Prediction images: `runs/detect/data/inspection_outputs/yolov8n_pcb_v1i_baseline_colab_predictions`

## Final Metrics

- Final epoch: `TBD`
- Precision: `TBD`
- Recall: `TBD`
- mAP50: `TBD`
- mAP50-95: `TBD`

## Practical Reading

- This workflow keeps the cleaned PCB v1i dataset separate from PKU and DeepPCB.
- It uses the existing train/valid/test split directly, without conversion.
- The first goal is a clean detection-only reference baseline before any later training changes.
