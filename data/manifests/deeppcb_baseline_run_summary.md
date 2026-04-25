# DeepPCB Baseline Run Summary

This note is the placeholder summary for the first DeepPCB-only Colab baseline run.
Fill in the metric fields after the run completes.

## Run Identity

- Run name: `yolov8n_deeppcb_baseline_colab`
- Dataset: `DeepPCB COCO baseline`
- Model: `yolov8n.pt`
- Task: `detection`
- Training style: `fine-tuning from pretrained weights`
- Runner: `Google Colab`
- Script: `scripts/deeppcb_yolov8_colab_train.py`
- Notebook: `notebook/deeppcb_yolov8_baseline_colab.ipynb`
- Data YAML: `configs/deeppcb_yolov8_baseline_data.yaml`

## Planned Training Settings

- Epochs: `10`
- Image size: `640`
- Batch size: `16`
- Workers: `2`
- Device: `0`
- Fraction: `1.0`
- Patience: `20`

## Dataset Handling

- Training split used in this workflow: `trainval`
- Validation split used in this workflow: `test`
- Inference-check split used in this workflow: `test`

## Expected Artifacts

- Run folder: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab`
- Best weights: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/weights/best.pt`
- Last weights: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/weights/last.pt`
- Metrics CSV: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/results.csv`
- Run args: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/args.yaml`
- Prediction images: `data/inspection_outputs/yolov8n_deeppcb_baseline_colab_predictions`

## Final Metrics

- Final epoch: `TBD`
- Train box loss: `TBD`
- Train cls loss: `TBD`
- Train dfl loss: `TBD`
- Validation box loss: `TBD`
- Validation cls loss: `TBD`
- Validation dfl loss: `TBD`
- Precision: `TBD`
- Recall: `TBD`
- mAP50: `TBD`
- mAP50-95: `TBD`

## Best Recorded Metrics

- Best mAP50: `TBD`
- Best mAP50 epoch: `TBD`
- Best mAP50-95: `TBD`
- Best mAP50-95 epoch: `TBD`

## After The Colab Run Finishes

- Confirm that `best.pt`, `last.pt`, and `results.csv` exist in the run folder.
- Open `results.csv` and record the final and best metrics above.
- Check the prediction image folder to confirm the model is producing visible detections.
