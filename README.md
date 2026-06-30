# AegisSafeRoad — Traffic Sentinel

<p align="center">
  <img src="./aegis_safe_traffic.webp" width="224" height="286" alt="Aegis Safe Work Shield" />
</p>


## Car Crash Detection Pipeline: Dataset Engineering, Feature Extraction, and Model Training

**Project:** AegisSafeRoad / Traffic Sentinel  
**Task:** Binary video classification — Normal Traffic vs. Car Crash  
**Author:** Ryan (Software Engineering, AI II — Universidad, Colombia)  
**Hardware:** Google Colab Pro — T4 / L4 GPU  
**Framework:** PyTorch 2.x, OpenCV, FFmpeg, Ultralytics  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Decision](#2-architecture-decision)
3. [Dataset Sources](#3-dataset-sources)
4. [ETL Stage 1 — Kaggle Download](#4-etl-stage-1--kaggle-download)
5. [ETL Stage 2 — Dataset Organization](#5-etl-stage-2--dataset-organization)
6. [ETL Stage 3 — Video Preprocessing and Tensor Extraction](#6-etl-stage-3--video-preprocessing-and-tensor-extraction)
7. [Model Architecture](#7-model-architecture)
8. [Training Pipeline](#8-training-pipeline)
9. [Results](#9-results)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Overlay Model — YOLOv11n-seg](#11-overlay-model--yolov11n-seg)
12. [Project Structure](#12-project-structure)
13. [Deployment Notes](#13-deployment-notes)

---

## 1. Project Overview

Traffic Sentinel is one of two sub-projects within AegisSafeRoad, a full-stack mobile-first web application for smart city road safety. Traffic Sentinel focuses exclusively on real-time car crash detection and semantic scene overlay from smartphone camera streams.

The system processes live video from the rear-facing camera of a user smartphone, running crash detection inference on the backend via a trained MobileNetV2 + Temporal Attention MLP model exported to ONNX, and rendering an instance segmentation overlay on the client side via onnxruntime-web.

The companion sub-project, ANPR, handles vehicle detection, license plate detection, OCR recognition of Colombian yellow plates, and road sign detection using three independent YOLOv8 models already trained and exported to ONNX.

---

## 2. Architecture Decision

### Model Selection

After evaluating multiple temporal architectures, the following design was selected for the crash detection core model:

**MobileNetV2 (ImageNet pretrained) + Temporal Attention MLP**

The primary alternatives considered were:

- MobileNetV3-Large + LSTM: rejected due to stateful ONNX export complexity. The hidden state tensors `(h0, c0)` in LSTM require explicit management in ONNX Runtime, which was identified as the root cause of a prior failed pipeline attempt.
- MobileNet-3D (Conv3D): rejected due to higher memory requirements during training and inflexibility in varying input frame counts.
- MobileNetV3-Large + Temporal Attention: rejected in favor of MobileNetV2 as a lighter backbone sufficient for binary classification without the Squeeze-and-Excitation overhead of V3.

The selected architecture produces a fully static ONNX computation graph with no recurrent state and no dynamic axes on internal tensors, enabling clean export and reliable inference in both ONNX Runtime on the backend and onnxruntime-web in the browser.

### Two-Stage Training Strategy

The training follows a two-stage fine-tuning protocol:

- **Stage 1:** MobileNetV2 fully frozen. Only the Temporal Attention MLP is trained for 4 epochs. This stabilizes the MLP weights before fine-tuning begins, preventing large MLP gradients from corrupting ImageNet-pretrained convolutional weights.
- **Stage 2:** Last 4 convolutional blocks of MobileNetV2 unfrozen. Both the CNN layers and the MLP are trained jointly with differential learning rates. This allows the backbone to learn crash-specific visual textures including metal deformation, debris, smoke, dust, and post-impact scene structure, while preserving low-level feature representations from ImageNet pretraining.

### Inference Format

A single ONNX file is exported at the end of training:

```
Input  : (1, 16, 3, 224, 224)  float32
Output : (1, 1)                float32  sigmoid probability
```

---

## 3. Dataset Sources

Three publicly available datasets were combined to construct a balanced binary classification dataset.

### 3.1 Car Crash Dataset (CCD)

- **Source:** Cogito2012/CarCrashDataset, hosted on Google Drive
- **Content:** 1,500 crash videos and 3,000 normal driving videos captured by dashcams
- **Format:** MP4, 50 frames per video at 10 FPS (5 seconds per clip)
- **Annotation:** Frame-level binary labels provided in `Crash-1500.txt`. Each line specifies the video name, a 50-element binary array indicating crash frames, timing condition (Day/Night), weather condition (Normal/Snowy/Rainy), and ego-vehicle involvement flag.
- **Label mapping:** `Normal/` folder maps to label 0, `Crash-1500/` folder maps to label 1.
- **Special handling:** Frame-level annotations from `Crash-1500.txt` were parsed to identify `crash_start`, the index of the first frame where the crash occurs. This enabled crash-centered temporal sampling during feature extraction.

### 3.2 TU-DAT (Traffic Anomalies Dataset)

- **Source:** Shared Google Drive folder linked from MDPI Sensors 2025 paper
- **Content:** 50 positive (crash) videos, 51 negative (normal) videos, 18 challenging-environment videos
- **Format:** MOV — handled explicitly in the ETL pipeline via dual-extension glob
- **Note on naming convention:** The dataset authors use "Positive" to denote crash presence and "Negative" to denote normal traffic, following anomaly detection convention. This is counterintuitive but was correctly mapped: `Negative_Videos` to label 0, `Positive_Videos` to label 1, `Challenging_Videos` (all confirmed car crashes) to label 1.
- **Challenging subset:** Videos 29 through 41 include night conditions, heavy cloud cover, snow, and very low light. All 18 were manually reviewed and confirmed as crash events.
- **Additional data:** 21 smartphone-recorded videos of normal Colombian urban traffic were added to the negative class to improve domain generalization for the target deployment environment.

### 3.3 Real-world Vehicle Crash Dataset — Balanced (Kaggle)

- **Source:** `umitka/real-world-vehicle-crash-dataset-balanced` on Kaggle
- **Content:** 1,500 videos organized into three severity classes: minor, moderate, major (500 per class)
- **Format:** MP4, approximately 20 seconds average duration
- **Label collapse:** All three severity classes were collapsed into a single label 1 (crash). The original train/val/test structure was discarded in favor of the unified split defined in this pipeline.
- **License:** CC BY-NC 4.0

### Dataset Summary

| Source | Label 0 (Normal) | Label 1 (Crash) | Format |
|---|---|---|---|
| CCD | 3,000 | 1,500 | MP4 |
| TU-DAT | 72 | 68 | MOV |
| Kaggle | 0 | 1,500 | MP4 |
| **Total** | **3,083** | **3,077** | Mixed |

Final balance ratio: **1.002x** (near-perfect 50/50 distribution).

---

## 4. ETL Stage 1 — Kaggle Download

Script: `01_kaggle_download_ETL.py`

The Kaggle dataset was downloaded programmatically from Google Colab using the new Kaggle API token format (`KGAT_...`), which stores credentials in `~/.kaggle/access_token` with permissions `600` rather than the legacy `kaggle.json` format.

The download process authenticates the token, downloads and decompresses the dataset to `/content/tmp_kaggle/` on local Colab disk for faster I/O, copies all extracted files to the destination directory on Drive, removes the temporary local directory, and reports file counts per severity category with label assignment confirmation.

---

## 5. ETL Stage 2 — Dataset Organization

Script: `02_etl_organize_processed.py`

All raw videos from the three sources were consolidated into a unified directory structure with standardized filenames and a continuous global index per label class starting from 1.

### Output structure

```
datasets/processed/
├── normal_0/
│   ├── normal_0001.mp4
│   └── ...  (3,075 files)
└── crash_1/
    ├── crash_0001.mp4
    └── ...  (3,068 files)
```

### Processing rules

Original filenames were discarded and replaced with the standardized `{label}_{index:04d}.mp4` format. The Kaggle dataset structure was traversed with `rglob("*.mp4")` to flatten all severity subfolders into a single crash class. The script is idempotent: files already present at the destination are skipped, allowing safe re-execution after interruption. Final verification: 6,143 total files, balance ratio 1.002x.

---

## 6. ETL Stage 3 — Video Preprocessing and Tensor Extraction

Script: `03_etl_video_tensor_extraction.py`

This stage converts all videos into fixed-shape float16 tensors suitable for direct DataLoader consumption during training, eliminating all video I/O overhead from the training loop.

### Preprocessing pipeline per video

```
source .mp4 / .mov
    -> FFmpeg: scale to 224x224, force 30 FPS, remove audio (-an), libx264 codec
    -> OpenCV: read 16 frames according to sampling strategy
    -> NumPy: HWC uint8 -> CHW float16, normalize to [0, 1] by dividing by 255
    -> np.save: (16, 3, 224, 224) float16 written to Drive
```

Audio removal was applied universally. Several source videos contained audio tracks irrelevant to visual crash detection.

### Temporal sampling strategies

**Uniform sampling** — label 0, all normal videos:

Frames are sampled at equal intervals across the full video duration regardless of length.

```python
indices = np.linspace(0, total_frames - 1, 16).astype(int)
```

**Crash-centered sampling** — label 1, CCD Crash-1500 with frame-level annotation:

The `crash_start` index is extracted from `Crash-1500.txt` for each of the 1,500 CCD crash videos. Eight frames are sampled before the crash onset and eight frames from the crash onset forward. This guarantees that the impact event is always represented in the second half of the temporal sequence.

```python
pre_frames  = np.linspace(max(0, crash_start - 8), crash_start, 8, endpoint=False)
post_frames = np.linspace(crash_start, min(total_frames-1, crash_start + 16), 8)
indices     = np.concatenate([pre_frames, post_frames])
```

**Biased sampling** — label 1, TU-DAT and Kaggle crash videos without frame-level annotation:

Frames are sampled uniformly over the range covering 40% to 100% of the video duration. Crash events in dashcam clips typically occur in the second half of trimmed recordings.

```python
start   = int(0.40 * total_frames)
indices = np.linspace(start, total_frames - 1, 16).astype(int)
```

### Manifest

A global `manifest.csv` was generated with schema:

```
file_path, label, split, sampling, source_origin_path
```

### Dataset split

Stratified random split maintaining 50/50 class balance within each partition:

| Split | Total | Normal | Crash |
|---|---|---|---|
| train | 5,039 | 2,519 | 2,520 |
| val | 1,000 | 500 | 500 |
| test | 117 | 60 | 57 |

One video (`normal_3051.mp4`) was rejected during extraction due to fewer than 16 decodable frames after FFmpeg processing. This was the only error across 6,156 processed videos.

### Tensor storage

```
Tensor shape  : (16, 3, 224, 224)  float16
Size per file : 4.59 MB
Total dataset : ~28 GB
```

---

## 7. Model Architecture

### CrashDetector

```
Input: (B, 16, 3, 224, 224)
    -> reshape to (B*16, 3, 224, 224)
    -> MobileNetV2 features + AdaptiveAvgPool2d + Flatten
    -> (B*16, 1280)
    -> reshape to (B, 16, 1280)
    -> TemporalAttention: Linear(1280->128)->Tanh->Linear(128->1)->Softmax->weighted sum
    -> (B, 1280)
    -> Linear(1280->512)->ReLU->Dropout(0.3)
    -> Linear(512->128)->ReLU->Dropout(0.3)
    -> Linear(128->1)->Sigmoid
Output: (B, 1)
```

**Parameter count:**

| Component | Parameters |
|---|---|
| MobileNetV2 | 2,223,872 |
| Temporal Attention | 164,097 |
| MLP Classifier | 721,665 |
| **Total** | **3,109,634** |

---

## 8. Training Pipeline

Script: `training.py`

### DataLoader

`CrashDataset` reads `.npy` tensors and applies per-frame ImageNet normalization at load time: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`. Tensors are stored as float16 and converted to float32 in `__getitem__`.

### Training configuration

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Epochs | 4 | 20 (early stopped at 6) |
| LR MLP | 1e-3 | 1e-4 |
| LR CNN | frozen | 1e-5 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Batch size | 16 | 16 |
| Weight decay | 1e-4 | 1e-4 |
| Loss | BCELoss | BCELoss |
| Early stopping patience | 6 | 6 |

---

## 9. Results

### Training progression — Stage 1

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC |
|---|---|---|---|---|---|
| 1 | 0.2759 | 0.8836 | 0.0976 | 0.9680 | 0.9937 |
| 2 | 0.1261 | 0.9572 | 0.0960 | 0.9690 | 0.9952 |
| 3 | 0.1100 | 0.9624 | 0.0733 | 0.9780 | 0.9958 |
| 4 | 0.0866 | 0.9729 | 0.0765 | 0.9740 | 0.9959 |

### Training progression — Stage 2

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC |
|---|---|---|---|---|---|
| 1 | 0.0826 | 0.9721 | 0.0584 | 0.9830 | 0.9974 |
| 2 | 0.0405 | 0.9877 | 0.0591 | 0.9820 | 0.9974 |
| 3 | 0.0331 | 0.9873 | 0.0643 | 0.9790 | 0.9976 |
| 4 | 0.0228 | 0.9914 | 0.0726 | 0.9820 | 0.9982 |
| 5 | 0.0180 | 0.9928 | 0.0823 | 0.9800 | 0.9975 |
| 6 | 0.0139 | 0.9944 | 0.0782 | 0.9790 | 0.9977 |

Best epoch: global epoch 5 (Stage 2 epoch 1), `val_loss=0.0584`.

### Test set results

| Metric | Value |
|---|---|
| Accuracy | 0.9829 |
| F1-Score | 0.9828 |
| Precision | 0.9661 |
| Recall | **1.0000** |
| AUC-ROC | 0.9950 |

**Confusion matrix:**

```
                  Predicted Normal    Predicted Crash
True Normal               58                  2
True Crash                 0                 57
```

The Recall of 1.000 (zero false negatives) is the most critical result for a road safety system. No crash event was missed on the held-out test set. The two false positives represent the conservative bias of the model, which is the preferred error direction for safety-critical applications.

### Model size

| Format | Size |
|---|---|
| best.pt | 16.0 MB |
| aegis_crash_detector.onnx | 16.2 MB |

---

## 10. Inference Pipeline

The correct preprocessing pipeline for inference must match the training pipeline exactly.

Two critical issues identified during initial inference testing:

1. The `CrashDetector.classifier` ends with `nn.Sigmoid()`. Calling `torch.sigmoid()` on the output applies sigmoid twice, collapsing all probabilities toward 0.5.
2. ImageNet normalization must be applied after dividing by 255, before passing frames to the model.

Correct inference:

```python
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_video(video_path, n_frames=16, img_size=224):
    cap     = cv2.VideoCapture(str(video_path))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=np.int32)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()
    tensor = torch.from_numpy(
        np.stack(frames).astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2)
    tensor = torch.stack([NORMALIZE(tensor[i]) for i in range(n_frames)])
    return tensor

@torch.no_grad()
def predict_video(model, video_path, device):
    x    = preprocess_video(video_path).unsqueeze(0).to(device)
    prob = model(x).item()    # sigmoid already applied inside model
    return {"label": "CRASH" if prob >= 0.5 else "NORMAL", "probability": prob}
```

---

## 11. Overlay Model — YOLOv11n-seg

Script: `train_yolo11n_seg.py`

A separate instance segmentation model was trained for aesthetic scene overlay in the Traffic Sentinel frontend. This model has no functional relationship to the crash detection pipeline.

| Parameter | Value |
|---|---|
| Base model | yolo11n-seg.pt (Ultralytics, COCO-Seg pretrained) |
| Dataset | COCO-Seg (80 classes, instance segmentation) |
| Image size | 256x256 |
| Epochs | 30 (fine-tuning) |
| Export format | ONNX, opset 17, float32, static input shape |
| ONNX size | 11.5 MB |
| Inference target | Client-side, onnxruntime-web |
| Expected FPS on device | 2–4 FPS on mid-range Android |

The model runs entirely on the user device. No inference roundtrip to the backend server is required for the overlay.

---

## 12. Project Structure

```
AegisSafeRoad/
├── datasets/
│   ├── raws/
│   │   ├── CCD/
│   │   │   ├── Normal/
│   │   │   ├── Crash-1500/
│   │   │   └── Crash-1500.txt
│   │   ├── TU-DAT/real/
│   │   │   ├── Negative_Videos/
│   │   │   ├── Positive_Videos/
│   │   │   └── Challenging_Videos/
│   │   └── CRASH-DATASET-KAGGLE/
│   └── processed/
│       ├── normal_0/
│       └── crash_1/
├── video_tensors/
│   ├── manifest.csv
│   ├── train/
│   ├── val/
│   └── test/
├── training/
│   ├── best.pt
│   ├── aegis_crash_detector.onnx
│   ├── training_log.csv
│   ├── test_results.txt
│   └── plots/
└── TrafficSentinel/
    └── yolo11n_seg_overlay/
        ├── weights/best.pt
        └── export/aegis_overlay_seg.onnx
```

---

## 13. Deployment Notes

### Backend — FastAPI + ONNX Runtime

The crash detection model runs server-side via ONNX Runtime on AWS EC2. The backend receives 16 frames from the smartphone WebSocket stream, preprocesses to `(1, 16, 3, 224, 224)` float32 with ImageNet normalization, and returns a crash probability. Confirmed crash events are logged to PostgreSQL via RDS.

### Frontend — Next.js + TypeScript + onnxruntime-web

The overlay model (`aegis_overlay_seg.onnx`, 11.5 MB) runs client-side via onnxruntime-web using the WebGL backend where available, falling back to WASM. Inference is triggered at 2–4 FPS independently of the camera stream frame rate.

### Infrastructure

- Backend: Docker container on AWS EC2 via docker-compose
- Database: AWS RDS PostgreSQL
- Frontend: Static Next.js build served from AWS S3

---

## Citations

**CCD:**

```
@InProceedings{BaoMM2020,
    author    = {Bao, Wentao and Yu, Qi and Kong, Yu},
    title     = {Uncertainty-based Traffic Accident Anticipation
                 with Spatio-Temporal Relational Learning},
    booktitle = {ACM Multimedia Conference},
    year      = {2020}
}
```

**Kaggle dataset:** Umit Ka, Real-world Vehicle Crash Dataset (Balanced), Kaggle, CC BY-NC 4.0.

**TU-DAT:** Published in MDPI Sensors, Vol. 25, No. 11, 2025.