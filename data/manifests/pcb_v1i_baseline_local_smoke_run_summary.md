# PCB v1i Baseline Local Smoke Run Summary

This note records the first short practical baseline run executed with the cleaned `PCB-Defect-Detection.v1i.coco` workflow.

## Run Identity

- Dataset: `PCB-Defect-Detection.v1i.coco`
- Run name: `yolov8n_pcb_v1i_baseline_local_smoke`
- Model: `yolov8n.pt`
- Task: `detection`
- Training style: `fine-tuning from pretrained weights`
- Runner: `local practical baseline confirmation`
- Script: `scripts/pcb_v1i_yolov8_colab_train.py`

## Split Usage

- Training split: `train`
- Validation split: `valid`
- Post-training inference-check split: `test`

## Run Settings

- Epochs: `1`
- Fraction: `0.01`
- Image size: `640`
- Batch size: `8`
- Device: `cpu`

## Main Artifacts

- Run folder: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_local_smoke`
- Best weights: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_local_smoke/weights/best.pt`
- Last weights: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_local_smoke/weights/last.pt`
- Metrics CSV: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_local_smoke/results.csv`
- Run args: `runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_local_smoke/args.yaml`
- Prediction images: `data/inspection_outputs/yolov8n_pcb_v1i_baseline_local_smoke_predictions`

## Final Metrics

- Final epoch: `1`
- Train box loss: `1.36767`
- Train cls loss: `6.47271`
- Train dfl loss: `1.26155`
- Validation box loss: `2.92015`
- Validation cls loss: `7.00670`
- Validation dfl loss: `2.71272`
- Precision: `0.00000`
- Recall: `0.00000`
- mAP50: `0.00000`
- mAP50-95: `0.00000`

## Practical Reading

- The cleaned-dataset YOLOv8 pipeline ran end to end and saved the expected training artifacts.
- The post-training inference check also completed and saved prediction images.
- This short `1`-epoch run is only a workflow confirmation, not the full Colab baseline.
- The model produced `0` predicted boxes on the saved `8`-image test check, so the detector is not meaningful yet at this stage.
