PHASE 1: ARCHITECTURAL BASELINE
-------------------------------
OBJECTIVE: 
Establish a stable starting point for the Emotion Recognition engine using the YOLOv10-Nano architecture.

CORE ACTIONS:
1. Dataset Preparation: Integrated AffectNet-YOLO format data.
2. Optimizer: Experimented with Muon-inspired optimization to achieve rapid initial convergence.
3. Architecture: Selected YOLOv10n (Nano) to ensure the model stays under 5MB for edge deployment.

KEY RESULTS:
- Successfully established a baseline mAP.
- Validated NMS-free inference for reduced latency on low-end chipsets.