# Final Model Severity Check

- Run name: `yolov8n_deeppcb_to_pcb_v1i_transfer_colab`
- Checkpoint used: `C:/Users/yummy/Downloads/PCB-Defect-Detector/runs/detect/runs/pcb_v1i_transfer/yolov8n_deeppcb_to_pcb_v1i_transfer_colab/weights/best.pt`
- Images checked: 4
- Images with predictions: 3
- Total detections: 3
- Average severity score: 0.7986
- Output folder: `C:/Users/yummy/Downloads/PCB-Defect-Detector/data/inspection_outputs/final_model_severity_check`
- JSON output: `C:/Users/yummy/Downloads/PCB-Defect-Detector/data/inspection_outputs/final_model_severity_check/severity_predictions.json`
- CSV output: `C:/Users/yummy/Downloads/PCB-Defect-Detector/data/inspection_outputs/final_model_severity_check/severity_predictions.csv`

## Sample Images
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole04_y0_x1080_jpg.rf.79b3881c050909c662b3252c43b3536d.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole101_y1080_x1080_jpg.rf.824d8fd0828fed91263b7ee42d18e56d.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole104_y0_x1080_jpg.rf.39e80e2e23acb3accbf83229f234ab00.jpg`
- `data/resources/PCB-Defect-Detection.v1i.coco/test/missing_hole105_y540_x0_jpg.rf.36d5414c91bd866e3bf6ec29b4f2e509.jpg`

## Practical Note

- Each detection record now includes class name, confidence, bounding box, relative box area, severity score, and severity band.
