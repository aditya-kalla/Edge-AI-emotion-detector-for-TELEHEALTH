

# 👁️ EmotionSensor-SOTA: Partitioned Affective Computing

A clinical-grade, real-time emotion recognition engine. This project transitions from standard macro-expression detection into a **Bi-Level Partitioned Architecture** designed to bridge the gap between low-power edge deployment and high-fidelity diagnostic accuracy[cite: 1, 3].

## 🏗️ Architectural Overview

To solve the "Resolution vs. Semantics" bottleneck, the project is divided into two distinct engines:

### **Partition A: The "Sentry" (Current Phase)**
*   **Architecture:** YOLOv10-Nano (NMS-Free).
*   **Target:** ESP32-S3 / Cortex M4 (Edge Hardware).
*   **Optimization:** INT8 Quantization, Rotational Invariance, and Temporal Mode Filtering.
*   **Role:** High-speed macro-emotion detection (Happy, Sad, Anger, Fear, Surprise, Contempt, Disgust, Neutral).

### **Partition B: The "Diagnostic" (Planned Phase)**
*   **Architecture:** Dual-Stream Vision Transformer (ViT) + MediaPipe Landmark Graph.
*   **Target:** Server-side / WebApp.
*   **Optimization:** 3D Pose Normalization, AU-Aware Multi-Label Heads[cite: 2, 4].
*   **Role:** Clinical-grade micro-expression mapping using Ekman's 23 Facial Action Units (AUs)[cite: 3, 4].

---

## 📈 Technical Evolution (Phase Log)

### **Phase 1: Baseline Foundation**
*   Established a stable starting point using YOLOv10n.
*   Validated the viability of 512KB RAM deployment on edge silicon.

### **Phase 2: Bias Identification**
*   Discovered "Mouth-Bias" where the model ignored upper-face morphology.
*   Transitioned to SGD with Momentum to stabilize gradient descent.

### **Phase 3: Pixel-Density Scaling**
*   Jumped resolution from $96 \times 96$ to $160 \times 160$.
*   Captured micro-expressions for Disgust and Contempt that were previously lost in downscaling.

### **Phase 4: The Surgical Protocol (SOTA Reached)**
*   **Adversarial Masking:** Implemented `erasing=0.4` on the lower face to force the model to learn "periocular" (eye-area) features[cite: 1].
*   **Nesterov Momentum:** Escaped the "Neutral Black Hole" local minima.
*   **Results:** Achieved a benchmark **mAP@50 of 0.7857**.

### **Phase 5: Production Hardening (In Progress)**
*   **Rotational Invariance:** Fine-tuning with $30^\circ$ augmentation to fix head-tilt failures.
*   **Temporal Smoothing:** Implementation of a custom `SentryEmotionFilter` to eliminate logic jitter and prioritize micro-expressions like Contempt[cite: 4].
*   **Quantization:** INT8 Post-Training Quantization (PTQ) for edge latency reduction.

---

## 🔬 Core Insights & Observations

Based on rigorous field testing, this project implements logic to combat four systemic neural failures:
1.  **Neutral Drift:** Using an "Anti-Neutral" bias in the temporal filter to capture subtle sadness[cite: 3].
2.  **Rotational Sensitivity:** Corrected via affine-transformation-simulated training[cite: 2].
3.  **The 3-Feature Limit:** Moving Partition B toward FACS (Facial Action Coding System) to recognize all 23 key action units rather than just eyes/mouth[cite: 4].
4.  **Semantic Tradeoffs:** Solving the Anger vs. Happy conflict by indexing corrugator/brow muscle tension over nasal structure distortion[cite: 1, 4].

---

## 🚀 Quick Start

### **1. Requirements**
```bash
pip install ultralytics streamlit mediapipe opencv-python
```

### **2. Running the Sentry (Partition A)**
```python
from ultralytics import YOLO
model = YOLO('models/sota_eye_specialist.pt')
results = model.predict(source=0, imgsz=160, stream=True)
```

---

## 📜 References
*   **nsae064.pdf**: Structural neuroanatomy of human facial behaviors[cite: 1].
*   **s41597-023-02693-z.pdf**: Multimodal datasets for emotion perception[cite: 2].
*   **Snoek et al. (2023)**: Testing and exploring models of facial expressions[cite: 3].
*   **iMotions Guidebook**: Facial Action Coding System (FACS) Visual Guide[cite: 4].

---
