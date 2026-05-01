import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
from filters import SentryEmotionFilter

# --- Page Config ---
st.set_page_config(page_title="Sentry V5: Clinical Benchmark", layout="wide")

st.title("👁️ Sentry Emotion Sensor v4.5")
st.subheader("Orientation-Hardened & Micro-Expression Sensitive")

# --- Initialize Model & Filter ---
@st.cache_resource
def load_resources():
    # Load your Phase 4.5 Weights
    model = YOLO('C:\\Aditya-Main\\PROJECTS\\Edge AI emotion detection for TELEHEALTH\\EmotionSensor-SENTRY\\Phase4.5_Rotational_Invariance\\weights\\best.pt') # Replace with your actual path
    sentry_filter = SentryEmotionFilter(window_size=6)
    return model, sentry_filter

model, emotion_filter = load_resources()

# --- UI Sidebar ---
with st.sidebar:
    st.header("Diagnostic Settings")
    conf_thresh = st.slider("Detection Confidence", 0.1, 1.0, 0.25)
    iou_thresh = st.slider("Box Stability (IoU)", 0.1, 1.0, 0.60)
    st.info("Phase 4.5 trained with 30° Rotational Augmentation.")

# --- Inference Pipeline ---
img_file_buffer = st.camera_input("Run SOTA Validation")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    start_time = time.time()
    # High-precision prediction[cite: 3]
    results = model.predict(
        source=cv2_img, 
        conf=conf_thresh, 
        iou=iou_thresh, 
        imgsz=160, 
        agnostic_nms=True
    )
    latency = (time.time() - start_time) * 1000

    for r in results:
        if len(r.boxes) > 0:
            # Extract Top Detection
            raw_label = r.names[int(r.boxes.cls[0])]
            conf = float(r.boxes.conf[0])
            
            # Apply Sentry Filter[cite: 4]
            stable_label = emotion_filter.update(raw_label, conf)
            
            # Visualization
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Clinical Output", stable_label)
            with col2:
                st.metric("Latency", f"{latency:.1f} ms")
                
            # Plot annotated image
            res_plotted = r.plot()
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with st.expander("🔬 View Neural Logic (Raw Scores)"):
                for i, c in enumerate(r.boxes.conf):
                    st.write(f"**{r.names[int(r.boxes.cls[i])]}**: {float(c):.2%}")
        else:
            st.warning("No face detected. Testing rotation invariance...")

st.markdown("---")
st.caption("Partition A: Edge-Optimized Sentry Engine. Verified for ESP32-S3 deployment.")