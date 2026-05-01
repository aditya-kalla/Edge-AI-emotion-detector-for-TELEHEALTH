import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import os
import time

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object

# --- Page Config ---
st.set_page_config(page_title="Emotion Sensor Pro: 160px Master", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #080a0e; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("👁️ Emotion Sensor: Phase 3 Master")
st.caption("Target Resolution: 160x160 | High-Precision Inference")

st.info(
    "Use this app from http://localhost:8501 on the same PC. "
    "Camera access is usually blocked on non-HTTPS network URLs like 192.168.x.x."
)

if not WEBRTC_AVAILABLE:
    st.warning(
        "Live webcam mode needs streamlit-webrtc. Install it with: pip install streamlit-webrtc"
    )

# --- Load Model ---
@st.cache_resource
def load_master_model():
    # Make sure you renamed your Phase 3 download to 'best_master_160.pt'
    model_path = "C:\\Aditya-Main\\PROJECTS\\Edge AI emotion detection for TELEHEALTH\\EmotionSensor_Phase3\\weights\\best.pt" 
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_master_model()

if model is None:
    st.error("❌ 'best_master_160.pt' not found! Please place the file in this folder.")
    st.stop()

# --- Controls ---
with st.sidebar:
    st.header("⚙️ Precision Controls")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 1.0, 0.25)
    st.info("💡 **Tip:** Keep at 160px for the most accurate eyebrow detection.")
    with st.expander("Camera Not Working? Fix Checklist"):
        st.markdown(
            """
1. Open **http://localhost:8501** (not the 192.168.x.x URL).
2. In browser address bar, allow camera permission for this site.
3. Close other apps using camera (Zoom, Teams, Meet, WhatsApp).
4. In Windows Settings -> Privacy & security -> Camera, enable browser camera access.
5. Refresh the page after changing permissions.
            """
        )

def run_inference(cv2_img):
    # 1. Start Timer for Performance Check
    start_time = time.time()

    # 2. RUN INFERENCE (Crucial: imgsz=160)
    results = model.predict(
        source=cv2_img,
        conf=conf_thresh,
        imgsz=160,
        save=False,
        stream=True
    )

    end_time = time.time()
    latency = (end_time - start_time) * 1000

    # 3. Process Results
    for r in results:
        if len(r.boxes) > 0:
            # Extract top prediction
            conf_scores = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()

            top_idx = np.argmax(conf_scores)
            label = r.names[int(class_ids[top_idx])]

            # UI Feedback
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Emotion", label)
            with col2:
                st.metric("Inference Speed", f"{latency:.1f} ms")

            # Plot and Show
            res_plotted = r.plot(line_width=2, font_size=1)
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="Master Brain Prediction", use_container_width=True)

            # Show raw confidence logic for debugging close classes
            with st.expander("🔬 View Deep Logic (Confidence Scores)"):
                for i, score in enumerate(conf_scores):
                    st.write(f"**{r.names[int(class_ids[i])]}**: {score:.2%}")
        else:
            st.warning("No face detected. Try better lighting.")


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_latency_ms = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        start_time = time.time()

        results = model.predict(
            source=img,
            conf=st.session_state.get("live_conf_thresh", 0.25),
            imgsz=160,
            save=False,
            verbose=False
        )

        self.last_latency_ms = (time.time() - start_time) * 1000

        if len(results) > 0 and len(results[0].boxes) > 0:
            r = results[0]
            conf_scores = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()
            top_idx = int(np.argmax(conf_scores))
            label = r.names[int(class_ids[top_idx])]
            top_conf = float(conf_scores[top_idx])

            plotted = r.plot(line_width=2, font_size=1)
            cv2.putText(
                plotted,
                f"{label} | {top_conf:.1%} | {self.last_latency_ms:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            return av.VideoFrame.from_ndarray(plotted, format="bgr24")

        cv2.putText(
            img,
            f"No face detected | {self.last_latency_ms:.1f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Inference Engine ---
input_mode = st.radio(
    "Input Source",
    ["Live Webcam", "Upload Image"],
    horizontal=True
)

if input_mode == "Live Webcam":
    if WEBRTC_AVAILABLE:
        st.session_state["live_conf_thresh"] = conf_thresh
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key="emotion-live-webcam",
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.caption("Install streamlit-webrtc, then rerun app for live webcam mode.")
else:
    upload = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "webp"])
    if upload is not None:
        file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
        cv2_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if cv2_img is None:
            st.error("Could not read the uploaded image. Please try a valid JPG, PNG, or WEBP file.")
        else:
            run_inference(cv2_img)

st.divider()
st.caption("Aditya | Telehealth Emotion Engine | v3.0 (160px)")