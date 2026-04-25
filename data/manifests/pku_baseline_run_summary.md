# PKU Baseline Run Summary

This note records the first real Google Colab baseline training run for the
PKU YOLOv8 detection workflow.

## Run Identity

- Dataset: `PKU COCO baseline`
- Model: `yolov8n.pt`
- Task: `detection`
- Training style: `fine-tuning from pretrained weights`
- Runner: `Google Colab`
- Script: `scripts/pku_yolov8_colab_train.py`
- Data YAML: `configs/pku_yolov8_baseline_data.yaml`
- Run name: `yolov8n_pku_baseline_colab`

## Key Settings

- Epochs: `10`
- Image size: `640`
- Batch size: `16`
- Workers: `2`
- Device: `0`
- Fraction: `1.0`
- Patience: `20`

## Main Artifacts

- Run folder: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab`
- Best weights: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab/weights/best.pt`
- Last weights: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab/weights/last.pt`
- Metrics CSV: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab/results.csv`
- Run args: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab/args.yaml`
- Sample prediction outputs: `runs/detect/data/inspection_outputs/pku_colab_baseline_predictions`

## Final Metrics

- Final epoch: `10`
- Train box loss: `3.11098`
- Train cls loss: `3.82078`
- Train dfl loss: `1.38010`
- Validation box loss: `3.12067`
- Validation cls loss: `3.46577`
- Validation dfl loss: `1.30289`
- Precision: `0.12047`
- Recall: `0.04990`
- mAP50: `0.01893`
- mAP50-95: `0.00717`

## Best Recorded Metrics

- Best mAP50: `0.01921` at epoch `9`
- Best mAP50-95: `0.00717` at epoch `10`

## Practical Reading

- The PKU Colab baseline workflow is working end to end.
- The model is learning somewhat: training and validation losses both decrease over the 10 epochs.
- Detection quality is still weak after this short run.
- Quick spot checks of the saved prediction images suggest the model is not yet producing strong visible detections.
- This run should be treated as the first real baseline, but it is still undertrained for useful performance.

## Recommended Next Single Step

- Run a longer Colab PKU baseline training job with the same setup before changing the dataset or model design.
