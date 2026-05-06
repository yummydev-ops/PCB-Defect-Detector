# Cross-Dataset Baseline Comparison

This note compares the best completed PKU-only baseline run with the completed
DeepPCB-only baseline run.

## Side-By-Side Metrics

| Item | PKU Baseline | DeepPCB Baseline |
| --- | --- | --- |
| Run name | `yolov8n_pku_baseline_colab_long50` | `yolov8n_deeppcb_baseline_colab` |
| Dataset | `PKU COCO baseline` | `DeepPCB COCO baseline` |
| Epochs | `50` | `10` |
| Model | `yolov8n.pt` | `yolov8n.pt` |
| Final mAP50 | `0.04391` | `0.95582` |
| Final mAP50-95 | `0.01627` | `0.71398` |

## Performance Gap

- The DeepPCB baseline performed much better than the PKU baseline.
- DeepPCB reached very strong detection quality after `10` epochs.
- PKU improved with a longer run, but overall detection quality remained weak.

## Important Caution

- These results are not directly comparable as if they came from the same task difficulty.
- PKU and DeepPCB still use separate datasets, separate native labels, and different underlying data characteristics.
- Treat this comparison as a baseline reference, not as proof that one model setup should be transferred directly across datasets without further work.

## Conclusion

- Best-performing baseline so far: `yolov8n_deeppcb_baseline_colab`
- Stronger run by final mAP50: `DeepPCB`
- Stronger run by final mAP50-95: `DeepPCB`
