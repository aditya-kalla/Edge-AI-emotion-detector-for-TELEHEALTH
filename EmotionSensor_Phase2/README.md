PHASE 2: BIAS IDENTIFICATION & REFINEMENT
-----------------------------------------
OBJECTIVE: 
Identify semantic failures and transition to a more stable optimization path.

CORE ACTIONS:
1. Optimizer Shift: Moved from Muon to SGD with Momentum (0.937) to stabilize weight updates.
2. Learning Rate Tuning: Implemented a refined LR schedule (lr0=0.01) to prevent gradient explosion.
3. Error Analysis: Identified "Happy Bias"—the model was over-relying on the mouth shape to predict emotions.

KEY RESULTS:
- mAP@50 climbed to ~0.56.
- Established the need for a higher resolution to distinguish subtle eye-area movements.