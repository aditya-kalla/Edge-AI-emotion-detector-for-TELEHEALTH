import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(page_title="Emotion Sensor - Phase 2 Test", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSuccess { background-color: #1e2a1e; border: 1px solid #4caf50; }
    .stWarning { background-color: #2a2a1e; border: 1px solid #ffcc00; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Emotion Sensor: Phase 2 Evaluation")
st.caption("Comparing Baseline (Phase 1) vs. Refined (Phase 2)")

# --- Sidebar: Model Selection ---
with st.sidebar:
    st.header("⚙️ Test Controls")
    
    # Selection between your two models
    model_choice = st.radio(
        "Select Model to Test:",
        ("Phase 1: Baseline", "Phase 2: Refined"),
        index=1 # Default to Phase 2
    )
    
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.35)
    
    st.divider()
    st.info("**Phase 2 Improvements:**\n- Freezing Layers applied\n- Reduced Learning Rate\n- Fixed Happy Bias")

# --- Load the selected model ---
@st.cache_resource
def load_model(choice):
    # Ensure these filenames match what you have in your folder!
    if choice == "Phase 1: Baseline":
        path = "C:\\Aditya-Main\\PROJECTS\\Edge AI emotion detection for TELEHEALTH\\EmotionSensor_Phase1\\weights\\best.pt" 
    else:
        path = "C:\\Aditya-Main\\PROJECTS\\Edge AI emotion detection for TELEHEALTH\\EmotionSensor_Phase2\\weights\\refined_best.pt" # Your new refined model
        
    if os.path.exists(path):
        return YOLO(path)
    else:
        return None

model = load_model(model_choice)

if model is None:
    st.error(f"⚠️ Model file for '{model_choice}' not found in directory!")
    st.stop()

# --- Camera Input ---
img_file_buffer = st.camera_input("Snapshot for Comparison")

if img_file_buffer is not None:
    # 1. Process Image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 2. Run Inference
    # We keep imgsz=96 to match your training exactly
    results = model.predict(source=cv2_img, conf=conf_threshold, imgsz=96)

    # 3. UI Display
    for r in results:
        if len(r.boxes) > 0:
            top_class = int(r.boxes.cls[0])
            label = r.names[top_class]
            prob = float(r.boxes.conf[0])
            
            # Show the specific result for this model
            color_box = "stSuccess" if label != "Happy" else "stWarning"
            st.markdown(f"### Result ({model_choice}): **{label}**")
            st.write(f"Confidence Score: **{prob:.2%}**")
            
            # Plot Result
            res_plotted = r.plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, use_container_width=True)
        else:
            st.warning("No face detected. Adjust lighting or move closer.")

st.divider()
st.caption("SOTA Emotion Engine Project - Aditya")