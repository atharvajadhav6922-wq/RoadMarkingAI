# Models Directory

This directory is reserved for optional AI/ML model weights.

## Baseline Behaviour

By default, **RoadMarkingAI uses a classical computer-vision pipeline** (OpenCV HSV + Canny + Hough) that requires **no model weights**.

## Plugging In a Trained Segmentation Model

To replace or augment the classical detector:

1. Place your model file here, e.g.:
   - `models/lane_seg.pt` (Ultralytics YOLO/SAM)
   - `models/lane_seg.onnx` (ONNX Runtime)
   - `models/unet_lane.pth` (PyTorch U-Net)

2. Implement `SegmentationLaneDetector` in `src/lane_detector.py` by subclassing `BaseLaneDetector`.

3. Update `config/settings.py` → `MODEL_PATH` to point to your weight file.

4. The `inspection_pipeline.py` will automatically use the new detector when a model is found.

## Supported Model Architectures (planned)

| Architecture | Library | Notes |
|---|---|---|
| YOLOv8-seg | `ultralytics` | Uncomment in requirements.txt |
| SegFormer | `transformers` | Requires HuggingFace |
| U-Net | `torch` | Custom implementation needed |
| DeepLabV3+ | `torch` | Custom implementation needed |
| ONNX (any) | `onnxruntime` | Portable, no GPU needed |

## Training Data

No model is included. To train your own detector:

- Annotate road-marking images with segmentation masks
- Use tools like CVAT, Label Studio, or Roboflow
- Train a segmentation model on your annotated dataset
- Export weights and place here
