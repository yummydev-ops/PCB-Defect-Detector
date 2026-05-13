# Cleaned PCB v1i vs DeepPCB Baseline Comparison

This note compares the two completed YOLOv8 detection baselines that are currently being kept separate.

## Side-by-Side Metrics

| Run name | Dataset | Epochs | Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `yolov8n_pcb_v1i_baseline_colab` | `PCB-Defect-Detection.v1i.coco` | 10 | `yolov8n.pt` | 0.97831 | 0.97537 | 0.98915 | 0.57440 |
| `yolov8n_deeppcb_baseline_colab` | `DeepPCB COCO baseline` | 10 | `yolov8n.pt` | 0.91156 | 0.91519 | 0.95582 | 0.71398 |

## Practical Reading

- The cleaned PCB v1i baseline is stronger on `precision`, `recall`, and `mAP50`.
- The DeepPCB baseline is stronger on `mAP50-95`, which suggests better performance under stricter localization scoring.
- Both runs are strong baselines, but they are strong in slightly different ways.

## Important Caution

- These results are **not directly interchangeable** as if they came from the same benchmark.
- The datasets differ in image characteristics, split construction, preprocessing history, and task difficulty.
- The comparison is useful for project tracking, but it should not be treated as a simple winner-takes-all ranking across datasets.

## Short Conclusion

- If the question is general detection success at IoU `0.50`, the cleaned PCB v1i baseline is stronger.
- If the question is stricter box quality across higher IoU thresholds, the DeepPCB baseline is stronger.
