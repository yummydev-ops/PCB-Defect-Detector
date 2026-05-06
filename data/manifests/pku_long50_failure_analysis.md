# PKU Baseline Failure Analysis

## Run

- Run name: `yolov8n_pku_baseline_colab_long50`
- Dataset: `PKU COCO baseline`
- Model: `yolov8n.pt`
- Training duration: `50` epochs
- Metrics source: `runs/detect/runs/pku_baseline/yolov8n_pku_baseline_colab_long50/results.csv`
- Prediction folder: `runs/detect/data/inspection_outputs/yolov8n_pku_baseline_colab_long50_predictions`

## Key Metrics

- Final precision: `0.12634`
- Final recall: `0.06582`
- Final mAP50: `0.04391`
- Final mAP50-95: `0.01627`

## Main Failure Patterns

1. **Missed small defects / dominant false negatives**
   - The saved validation prediction images are mostly empty, with no visible predicted boxes on sampled defect images.
   - This matches the very low recall and suggests the model often fails to trigger on true defects at all.

2. **Low confidence on true defect regions**
   - The prediction outputs suggest the model is conservative rather than noisy.
   - The main problem is under-detection, not a large number of obvious false positives.

3. **Weak localization quality**
   - The gap between low `mAP50` and very low `mAP50-95` indicates localization remains weak even when the model improves slightly.
   - Small PCB defects leave very little room for bbox error, so minor misalignment likely hurts scoring heavily.

4. **Class separation is still unclear because detections are sparse**
   - There is not much visual evidence of consistent class-specific predictions in the saved samples.
   - The primary failure appears earlier in the pipeline: detecting the defect at all.

## Most Likely Failure Reasons

- Defects occupy a very small portion of each `640x640` image.
- The PKU images contain subtle visual differences, so defect cues are easy to lose after resizing.
- `yolov8n.pt` is a lightweight baseline model and may be too limited for this small-object setting.
- Longer training alone helped slightly, but it did not solve the core small-defect sensitivity problem.

## Representative Saved Outputs

- `runs/detect/data/inspection_outputs/yolov8n_pku_baseline_colab_long50_predictions/01_missing_hole_02_jpg.rf.2052cfb5a78ca346757e187f8310848f.jpg`
- `runs/detect/data/inspection_outputs/yolov8n_pku_baseline_colab_long50_predictions/01_missing_hole_13_jpg.rf.a689ce325d2f55233db89581316d8c39.jpg`
- `runs/detect/data/inspection_outputs/yolov8n_pku_baseline_colab_long50_predictions/01_mouse_bite_02_jpg.rf.da9da140d18fc011c38bda197baae37c.jpg`

## Practical Next Improvement Direction

Keep the PKU workflow separate and focus the next baseline change on **small-object sensitivity**, starting with a higher input image size in the same PKU-only YOLO detection setup before changing datasets, labels, or tasks.
