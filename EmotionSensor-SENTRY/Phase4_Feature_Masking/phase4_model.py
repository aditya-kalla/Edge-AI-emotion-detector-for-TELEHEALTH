import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import os
import time
from pathlib import Path

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# --- Page Config ---
st.set_page_config(page_title="Victory Test: SOTA Emotion Engine", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border: 1px solid #238636; border-radius: 10px; }
    .status-box { padding: 20px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏆 Phase 4 Victory Test: The Surgical Master")
st.caption("Model: YOLOv10-Nano (Eye-Specialist) | Target mAP: 0.78")

# --- Load the Masterwork ---
@st.cache_resource
def load_sota_model():
    model_path = Path(__file__).resolve().parent / "weights" / "best.pt"
    if model_path.exists():
        return YOLO(str(model_path))
    return None

model = load_sota_model()

if model is None:
    st.error("❌ 'sota_eye_specialist.pt' not found! Please rename and move the Phase 4 best.pt here.")
    st.stop()


def run_inference(frame_bgr, conf_thresh):
    start = time.time()
    results = model.predict(source=frame_bgr, conf=conf_thresh, imgsz=160, verbose=False)
    latency = (time.time() - start) * 1000

    annotated_frame = frame_bgr
    top_label = None
    top_conf = None
    probabilities = []
    

    for result in results:
        if len(result.boxes) > 0:
            probs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()

            top_label = result.names[int(cls_ids[0])]
            top_conf = float(probs[0])
            probabilities = [(result.names[int(cls_ids[i])], float(probs[i])) for i in range(len(probs))]
            annotated_frame = result.plot(line_width=3)
            break

    return annotated_frame, top_label, top_conf, latency, probabilities


if WEBRTC_AVAILABLE:
    class EmotionVideoProcessor(VideoProcessorBase):
        def __init__(self, conf_thresh):
            self.conf_thresh = conf_thresh
            self.latest_result = None

        def recv(self, frame):
            frame_bgr = frame.to_ndarray(format="bgr24")
            annotated_frame, top_label, top_conf, latency, probabilities = run_inference(frame_bgr, self.conf_thresh)
            self.latest_result = {
                "top_label": top_label,
                "top_conf": top_conf,
                "latency": latency,
                "probabilities": probabilities,
            }
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- Layout ---
col_ui, col_cam = st.columns([1, 2])

with col_ui:
    st.markdown("### 📊 Metrics")
    conf_thresh = st.slider("Sensitivity (Conf)", 0.10, 1.0, 0.40)
    
    st.info("""
    **Testing Protocol:**
    1. **Sadness:** Try a subtle lip-tilt.
    2. **Anger vs Surprise:** Try both with mouth open.
    3. **Distance:** Stand 60cm+ away.
    """)

# --- Camera Inference ---
with col_cam:
    st.markdown("### 📷 Live Camera Detection")

    tab_live, tab_upload = st.tabs(["Live Webcam", "Upload Images"])

    with tab_live:
        if WEBRTC_AVAILABLE:
            rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            ctx = webrtc_streamer(
                key="phase4-live-emotion",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=lambda: EmotionVideoProcessor(conf_thresh),
            )

            if ctx.video_processor and ctx.video_processor.latest_result:
                latest = ctx.video_processor.latest_result
                metrics_col1, metrics_col2 = st.columns(2)
                metrics_col1.metric(
                    "Top Emotion",
                    latest["top_label"] or "No face detected",
                    f"{latest['top_conf']:.1%}" if latest["top_conf"] is not None else "--",
                )
                metrics_col2.metric("Edge Latency", f"{latest['latency']:.1f}ms")

                if latest["probabilities"]:
                    with st.expander("🔬 View Surgical Probability Distribution"):
                        for label, probability in latest["probabilities"]:
                            st.write(f"**{label}**: {probability:.2%}")
        else:
            st.warning("streamlit-webrtc is not installed, so the app is falling back to still-frame capture.")
            img_file_buffer = st.camera_input("Capture SOTA Benchmark")

            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                annotated_frame, top_label, top_conf, latency, probabilities = run_inference(cv2_img, conf_thresh)

                if top_label is not None:
                    m1, m2 = st.columns(2)
                    m1.metric("Top Emotion", top_label, f"{top_conf:.1%}")
                    m2.metric("Edge Latency", f"{latency:.1f}ms")

                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                    with st.expander("🔬 View Surgical Probability Distribution"):
                        for label, probability in probabilities:
                            st.write(f"**{label}**: {probability:.2%}")
                else:
                    st.warning("Distance check: No face detected. Adjust lighting or position.")

    with tab_upload:
        uploaded_images = st.file_uploader(
            "Upload one or more face images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )

        if uploaded_images:
            for idx, uploaded_file in enumerate(uploaded_images, start=1):
                st.markdown(f"#### Image {idx}: {uploaded_file.name}")
                file_bytes = uploaded_file.read()
                cv2_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

                if cv2_img is None:
                    st.error(f"Could not decode image: {uploaded_file.name}")
                    continue

                annotated_frame, top_label, top_conf, latency, probabilities = run_inference(cv2_img, conf_thresh)

                if top_label is not None:
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Top Emotion", top_label, f"{top_conf:.1%}")
                    col_m2.metric("Inference Latency", f"{latency:.1f}ms")
                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                    with st.expander(f"Probabilities: {uploaded_file.name}"):
                        for label, probability in probabilities:
                            st.write(f"**{label}**: {probability:.2%}")
                else:
                    st.warning(f"No face detected in {uploaded_file.name}. Try a clearer image.")

st.divider()
st.markdown("🚀 *Aditya's SOTA Emotion Sensor v4.0*")