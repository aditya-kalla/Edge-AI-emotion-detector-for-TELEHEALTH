PHASE 4: THE SURGICAL PROTOCOL (SOTA)
-------------------------------------
OBJECTIVE: 
Force feature disentanglement to solve Surprise-Anger conflicts and Neutral-bias.

CORE ACTIONS:
1. Adversarial Masking: Implemented 'erasing=0.4' targeting the lower face. By hiding the mouth, the AI was forced to master periocular (eye/brow) features.
2. High-Velocity SGD: Used Nesterov Momentum to "shake" the model out of the Neutral local minima.
3. Scale Invariance: Expanded training scale range to 0.5 to fix the "60cm Wall" distance failure.
4. Asymmetric Loss: Increased cls_loss gain to 2.0 to punish misclassifications of complex emotions (Sadness/Fear).

KEY RESULTS:
- Final mAP@50: 0.7857 (Significant SOTA breakthrough).
- Sadness mAP climbed to 0.719 (Escaped the Neutral Black Hole).
- Eliminated "No Face Detected" errors at distances over 60cm.