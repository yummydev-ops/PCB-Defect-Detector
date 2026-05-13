# Final Model Selection Comparison

This note compares the three completed detection runs that matter for the final project model decision.

## Side-by-Side Metrics

| Run name | Dataset | Epochs | Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `yolov8n_pcb_v1i_baseline_colab` | `PCB-Defect-Detection.v1i.coco` | 10 | `yolov8n.pt` | 0.97831 | 0.97537 | 0.98915 | 0.57440 |
| `yolov8n_deeppcb_baseline_colab` | `DeepPCB COCO baseline` | 10 | `yolov8n.pt` | 0.91156 | 0.91519 | 0.95582 | 0.71398 |
| `yolov8n_deeppcb_to_pcb_v1i_transfer_colab` | `PCB-Defect-Detection.v1i.coco` | 10 | `DeepPCB best.pt -> PCB v1i fine-tuning` | 0.97514 | 0.98006 | 0.98884 | 0.56531 |

## Practical Reading

- The two runs that matter most for the **final project model** are the two evaluated on `PCB-Defect-Detection.v1i.coco`:
  - `yolov8n_pcb_v1i_baseline_colab`
  - `yolov8n_deeppcb_to_pcb_v1i_transfer_colab`
- The transfer run kept recall slightly higher, but it did **not** improve the main target-dataset scores enough to replace the frozen baseline:
  - precision dropped from `0.97831` to `0.97514`
  - mAP50 dropped from `0.98915` to `0.98884`
  - mAP50-95 dropped from `0.57440` to `0.56531`
- The DeepPCB baseline remains a strong supporting result, but it should not be chosen as the final project model because it was trained and evaluated on a different dataset with different data characteristics.
- DeepPCB and PCB v1i results are therefore useful for context, but not directly interchangeable as a final-model decision basis.

## Final Recommendation

- **Final project model:** `yolov8n_deeppcb_to_pcb_v1i_transfer_colab`
- **Reason:** the project decision is to use the sequential fine-tuning path as the final backbone, so the selected checkpoint carries both DeepPCB pre-adaptation and final PCB v1i adaptation in one deployable model.
- **Role of the direct PCB v1i baseline:** keep it as a strong reference run and comparison point, but not as the chosen final checkpoint.
