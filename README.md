# RedhawkRCDetection
A small Python-based project for RC control with OpenCV and feature detection.


## ✨ Features & Functionality

### 1) Data Ingestion & Management
- **Dataset layout**: expects `data/images/` and `data/labels/` (YOLO-style) with a `data/splits.yaml` for train/val/test.
- **Flexible sources**: local folders or cloud links (Kaggle/Drive) you can mount or sync.
- **Label formats**: YOLO `.txt` by default; adapters for COCO JSON or Pascal VOC XML.
- **Split helpers**: scripts/notebooks to create and verify stratified splits and class balance.

### 2) Preprocessing & Augmentation
- **Image transforms**: resize/letterbox, normalization, color jitter, blur/sharpen.
- **Geometric aug**: flip, rotate, mosaic/mixup (if using Ultralytics).
- **Configurable pipelines**: controlled via CLI flags or YAML.
- **Reproducibility**: seed control for deterministic runs.

### 3) Training Pipeline
- **One-command training** (Ultralytics-style `yolo detect train …`) or `train.py`.
- **Model zoo**: start from pretrained (`yolov8n/s/m/l` etc.), swap via YAML/CLI.
- **Hyperparameters**: batch, imgsz, epochs, LR schedule, warmup, EMA.
- **Checkpointing**: best/last; resume training from any epoch.
- **Device**: CPU/GPU auto-detect; optional multi-GPU (`--device 0,1`).

### 4) Inference & Serving
- **Batch inference**: on folders or file lists; saves annotated outputs.
- **Video/webcam**: `--source 0` or path/URL; optional live display.
- **Thresholds**: confidence & IoU/NMS configurable globally or per class.
- **Outputs**: images/video with boxes + raw JSON/CSV predictions.
- **Export** *(optional)*: ONNX / TorchScript for deployment/TensorRT.

### 5) Evaluation
- **Metrics**: mAP@0.5 and mAP@0.5:.95, per-class precision/recall, confusion matrix.
- **Reports**: text, CSV/JSON; notebooks for PR curves & per-class AP.
- **Repro**: experiment-named run folders for comparisons.

### 6) Visualization & Debugging
- **Dataset viewer**: sample labeled images to check annotations.
- **Prediction viewer**: GT vs predictions; filter by class/confidence.
- **Error analysis**: false pos/neg, small/occluded object breakdowns.

### 7) Configuration & CLI
- **YAML configs** (e.g., `data/splits.yaml`, `configs/train.yaml`) centralize paths/hparams.
- **CLI overrides** for rapid iteration (`--epochs 50 --imgsz 640 --batch 16`).

### 8) Modularity & Extensibility
- **Separation** of data, training, inference, and evaluation utilities.
- **Utils**: logging, I/O, drawing, metrics shared across scripts.
- **Plug-ins**: add custom datasets, augmentations, or heads with minimal changes.

### 9) Notebooks (if present)
- **End-to-end**: EDA → train → infer → evaluate.
- **Interactive knobs**: thresholds, augmentations, NMS settings.
- **Sanitization**: keep tokens in `.env`; clear outputs before commits.

### 10) Quality & Reproducibility
- **Deterministic seeds** (`--seed`).
- **Versioned runs** with fixed configs.
- **Tests**: `pytest` stubs for core utils.

---

## 🔧 Common Commands (recap)

**Train**
```bash
yolo detect train data=data/splits.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=16
```

**Infer on images/folders**
```bash
python inference.py --source path/to/images --weights runs/detect/train/weights/best.pt --conf 0.25 --iou 0.45
```

**Webcam / stream**
```bash
python inference.py --source 0 --weights runs/detect/train/weights/best.pt
```

**Evaluate**
```bash
python eval.py --weights runs/detect/train/weights/best.pt --data data/splits.yaml
```

---

## 🧩 How the Pieces Fit Together

1. **Prepare data** → confirm folder layout & YAML config.  
2. **(Optional) Explore** in notebooks; adjust augmentations/thresholds.  
3. **Train** a chosen model; monitor runs in `runs/`.  
4. **Infer** on new media; save annotated outputs + raw predictions.  
5. **Evaluate** to get mAP/precision/recall; visualize strengths/weaknesses.  
6. **Export/Deploy** via ONNX/TorchScript or wrap in Streamlit/FastAPI.

