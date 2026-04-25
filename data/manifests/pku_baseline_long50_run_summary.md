# PKU Baseline Long50 Run Summary

This note records the completed longer PKU-only Colab baseline run.

## Run Identity

- Run name: `yolov8n_pku_baseline_colab_long50`
- Dataset: `PKU COCO baseline`
- Model: `yolov8n.pt`
- Task: `detection`
- Training style: `fine-tuning from pretrained weights`
- Runner: `Google Colab`
- Script: `scripts/pku_yolov8_colab_train.py`
- Notebook: `notebook/pku_yolov8_baseline_colab.ipynb`
- Data YAML: `configs/pku_yolov8_baseline_data.yaml`

## Training Settings

- Epochs: `50`
- Image size: `640`
- Batch size: `16`
- Workers: `2`
- Device: `0`
- Fraction: `1.0`
- Patience: `20`

## Main Artifacts

- Run folder: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab_long50`
- Best weights: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab_long50/weights/best.pt`
- Last weights: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab_long50/weights/last.pt`
- Metrics CSV: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab_long50/results.csv`
- Run args: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab_long50/args.yaml`
- Prediction images: `runs/detect/data/inspection_outputs/yolov8n_pku_baseline_colab_long50_predictions`

## Final Metrics

- Final epoch: `50`
- Train box loss: `2.65213`
- Train cls loss: `2.93342`
- Train dfl loss: `1.22766`
- Validation box loss: `2.81502`
- Validation cls loss: `2.96187`
- Validation dfl loss: `1.19432`
- Precision: `0.12634`
- Recall: `0.06582`
- mAP50: `0.04391`
- mAP50-95: `0.01627`

## Best Recorded Metrics

- Best mAP50: `0.04516`
- Best mAP50 epoch: `46`
- Best mAP50-95: `0.01681`
- Best mAP50-95 epoch: `44`

## Practical Reading

- The longer PKU-only Colab baseline completed successfully and saved all expected artifacts.
- The model improved compared with the shorter 10-epoch baseline.
- Losses decreased further, and mAP50 rose to about `0.044`, but the detector is still weak overall.
- Quick spot checks of the saved prediction images suggest visible detections are still limited at the current confidence setting.

## Recommended Next Single Step

- Review and compare the `10`-epoch and `50`-epoch PKU baseline runs side by side before changing any dataset or model settings.
