PHASE 3: PIXEL-DENSITY OPTIMIZATION
-----------------------------------
OBJECTIVE: 
Overcome the "Resolution Wall" by moving from 96x96 to 160x160 input.

CORE ACTIONS:
1. Resolution Jump: Scaled the input grid to 160x160. 
2. Transfer Learning: Leveraged weights from Phase 2 to accelerate convergence at the higher resolution.
3. Hardware Validation: Confirmed 160x160 fits within the 512KB RAM envelope of target edge devices.

KEY RESULTS:
- Massive improvement in 'Surprise' detection.
- mAP stabilized at 0.5815.
- Identified the "60cm Wall"—detection failure at distance.