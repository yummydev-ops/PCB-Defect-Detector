# DeepPCB Baseline Visual Inference Check

- Run name: `yolov8n_deeppcb_baseline_colab`
- Checkpoint used: `C:/Users/yummy/Downloads/github files/PCB-Defect-Detector/runs/detect/runs/deeppcb_baseline/yolov8n_deeppcb_baseline_colab/weights/best.pt`
- Images checked: 8
- Images with predictions: 8
- Total predicted boxes: 40
- Output folder: `C:/Users/yummy/Downloads/github files/PCB-Defect-Detector/data/inspection_outputs/deeppcb_baseline_predictions`
- Confidence threshold: 0.25

## Sample Images
- `data/coco_master/deeppcb_full/images/test/group00041/00041200_test.jpg`
- `data/coco_master/deeppcb_full/images/test/group00041/00041201_test.jpg`
- `data/coco_master/deeppcb_full/images/test/group00041/00041202_test.jpg`
- `data/coco_master/deeppcb_full/images/test/group00041/00041203_test.jpg`
- `data/coco_master/deeppcb_full/images/test/group00041/00041204_test.jpg`
- `data/coco_master/deeppcb_full/images/test/group00041/00041205_test.jpg`
- `data/coco_master/deeppcb_full/images/test/group00041/00041206_test.jpg`
- `data/coco_master/deeppcb_full/images/test/group00041/00041207_test.jpg`

## Predicted Class Counts
- `open`: 8
- `short`: 4
- `mousebite`: 7
- `spur`: 9
- `copper`: 6
- `pin-hole`: 6

## Observation

- Detections appear consistently across the checked images, and quick spot checks suggest the boxes align reasonably with visible defect regions.
