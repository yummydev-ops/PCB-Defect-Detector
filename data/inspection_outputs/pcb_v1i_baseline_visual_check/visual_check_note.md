# PCB v1i Baseline Visual Inference Check

- Run name: `yolov8n_pcb_v1i_baseline_colab`
- Checkpoint used: `C:/Users/yummy/Downloads/PCB-Defect-Detector/runs/detect/runs/pcb_v1i_baseline/yolov8n_pcb_v1i_baseline_colab/weights/best.pt`
- Images checked: 8
- Images with predictions: 6
- Total predicted boxes: 6
- Output folder: `C:/Users/yummy/Downloads/PCB-Defect-Detector/data/inspection_outputs/pcb_v1i_baseline_visual_check`
- Confidence threshold: 0.25

## Sample Images
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole04_y0_x1080_jpg.rf.79b3881c050909c662b3252c43b3536d.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole101_y1080_x1080_jpg.rf.824d8fd0828fed91263b7ee42d18e56d.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole104_y0_x1080_jpg.rf.39e80e2e23acb3accbf83229f234ab00.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole105_y540_x0_jpg.rf.36d5414c91bd866e3bf6ec29b4f2e509.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole106_y540_x1620_jpg.rf.627b1eee4ca2a90cec1311773944aee3.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole108_y1620_x1620_jpg.rf.455966fc44ac544ce2e7ae8cd798f453.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole109_y0_x1080_jpg.rf.f2e27a3c7dc83f5ff6d76446dabbcf45.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole109_y1620_x1620_jpg.rf.2d2b765f205f62995cf49f855570f49f.jpg`

## Predicted Class Counts
- `missing_hole`: 6
- `mouse_bite`: 0
- `open_circuit`: 0
- `short`: 0
- `spur`: 0
- `spurious_copper`: 0

## Observation

- Detections appear on `6` of the `8` checked images. In quick spot checks, the predicted boxes align reasonably with visible missing-hole regions, but coverage is still uneven in this small qualitative sample.
