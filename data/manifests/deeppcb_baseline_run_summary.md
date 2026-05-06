# DeepPCB Baseline Run Summary

This note records the completed first DeepPCB-only Colab baseline run.

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

## Training Settings

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

## Main Artifacts

- Run folder: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab`
- Best weights: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/weights/best.pt`
- Last weights: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/weights/last.pt`
- Metrics CSV: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/results.csv`
- Run args: `runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/args.yaml`
- Prediction images: `not currently present in the synced repo outputs`

## Final Metrics

- Final epoch: `10`
- Train box loss: `0.99683`
- Train cls loss: `0.78994`
- Train dfl loss: `0.95062`
- Validation box loss: `0.89814`
- Validation cls loss: `0.62905`
- Validation dfl loss: `0.91828`
- Precision: `0.91156`
- Recall: `0.91519`
- mAP50: `0.95582`
- mAP50-95: `0.71398`

## Best Recorded Metrics

- Best mAP50: `0.95582`
- Best mAP50 epoch: `10`
- Best mAP50-95: `0.71398`
- Best mAP50-95 epoch: `10`

## Practical Reading

- The DeepPCB-only Colab baseline completed successfully and saved the core training artifacts.
- Detection quality is strong on this baseline setup.
- The final metrics are far stronger than the PKU-only baseline runs.
- Prediction images are not currently present in the synced repo outputs, so the qualitative inference check still needs to be saved separately if we want visual documentation.

## Recommended Next Single Step

- Run and save a short DeepPCB visual inference check using `best.pt` on a few test images so the baseline has both metric and qualitative evidence.
