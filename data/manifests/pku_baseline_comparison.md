# PKU Baseline Run Comparison

This note compares the two completed PKU-only YOLOv8 detection baseline runs.

## Side-By-Side Metrics

| Item | Run 1 | Run 2 |
| --- | --- | --- |
| Run name | `yolov8n_pku_baseline_colab` | `yolov8n_pku_baseline_colab_long50` |
| Epochs | `10` | `50` |
| Model | `yolov8n.pt` | `yolov8n.pt` |
| Dataset | `PKU COCO baseline` | `PKU COCO baseline` |
| Final mAP50 | `0.01893` | `0.04391` |
| Final mAP50-95 | `0.00717` | `0.01627` |
| Best mAP50 | `0.01921` at epoch `9` | `0.04516` at epoch `46` |
| Best mAP50-95 | `0.00717` at epoch `10` | `0.01681` at epoch `44` |

## Practical Reading

- The `50`-epoch PKU baseline performed better than the `10`-epoch run.
- Final mAP50 improved from `0.01893` to `0.04391`.
- Final mAP50-95 improved from `0.00717` to `0.01627`.
- Best recorded mAP50 and best recorded mAP50-95 also improved in the longer run.
- Even with that improvement, the detector is still weak overall, so the longer run should be treated as a better baseline rather than a strong final model.

## Conclusion

- Best-performing PKU run so far: `yolov8n_pku_baseline_colab_long50`
