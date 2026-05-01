{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d20a48a9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Creating new Ultralytics Settings v0.0.6 file  \n",
      "View Ultralytics Settings with 'yolo settings' or at 'C:\\Users\\lenovo\\AppData\\Roaming\\Ultralytics\\settings.json'\n",
      "Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st \n",
    "import cv2\n",
    "from ultralytics import YOLO\n",
    "import numpy as np\n",
    "\n",
    "st.set_page_config(page_title = \"EmotionDetector AI\", layout = \"wide\")\n",
    "st.title(\"SOTA Emotion Sensor - Phase 1\")\n",
    "\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    model = YOLO(\"C:\\\\Aditya-Main\\\\PROJECTS\\\\Edge AI emotion detection for TELEHEALTH\\\\weights\\\\best.pt\")\n",
    "\n",
    "model = load_model()\n",
    "\n",
    "st.sidebar.header(\"Settings\")\n",
    "conf_threshold = st.sidebar.slider(\"Confidence Threshold\", 0.0, 1.0, 0.5, 0.01)\n",
    "\n",
    "img_file_buffer = st.camera_input(\"Take a photo to analyse or use live feed\")\n",
    "\n",
    "if img_file_buffer is not None: \n",
    "\n",
    "    bytes_data = img_file_buffer.getvalue()\n",
    "    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np_uint8), cv2.IMREAD_COLOR)\n",
    "\n",
    "    results = model(cv2_img, conf=conf_threshold)\n",
    "\n",
    "    for r in results:\n",
    "        if len(r.boxes) > 0:\n",
    "            top_class = int(r.boxes.cls[0])\n",
    "            label = r.names[top_class]\n",
    "            prob = r.boxes.conf[0]\n",
    "\n",
    "            st.success(f\"Detected Emotion: **{label}** ({prob:.2%})\")\n",
    "            \n",
    "            # Draw on image\n",
    "            res_plotted = r.plot()\n",
    "            st.image(res_plotted, caption=\"AI Analysis\", channels=\"BGR\")\n",
    "        else:\n",
    "            st.warning(\"No face detected. Try moving closer to the camera.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4363e6a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-18 14:27:54.903 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:27:54.905 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:27:54.973 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\lenovo\\AppData\\Roaming\\Python\\Python314\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-04-18 14:27:54.974 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:27:54.975 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:27:54.978 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.set_page_config(page_title = \"EmotionDetector AI\", layout = \"wide\")\n",
    "st.title(\"SOTA Emotion Sensor - Phase 1\")\n",
    "\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    model = YOLO(\"C:\\\\Aditya-Main\\\\PROJECTS\\\\Edge AI emotion detection for TELEHEALTH\\\\weights\\\\best.pt\")\n",
    "\n",
    "model = load_model()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "58f347a9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-04-18 14:33:07.195 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.196 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.197 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.198 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.199 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.200 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.202 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.203 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.204 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.205 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.207 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.208 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.210 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.210 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-04-18 14:33:07.211 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.sidebar.header(\"Settings\")\n",
    "conf_threshold = st.sidebar.slider(\"Confidence Threshold\", 0.0, 1.0, 0.5, 0.01)\n",
    "\n",
    "img_file_buffer = st.camera_input(\"Take a photo to analyse or use live feed\")\n",
    "\n",
    "if img_file_buffer is not None: \n",
    "\n",
    "    bytes_data = img_file_buffer.getvalue()\n",
    "    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np_uint8), cv2.IMREAD_COLOR)\n",
    "\n",
    "    results = model(cv2_img, conf=conf_threshold)\n",
    "\n",
    "    for r in results:\n",
    "        if len(r.boxes) > 0:\n",
    "            top_class = int(r.boxes.cls[0])\n",
    "            label = r.names[top_class]\n",
    "            prob = r.boxes.conf[0]\n",
    "\n",
    "            st.success(f\"Detected Emotion: **{label}** ({prob:.2%})\")\n",
    "            \n",
    "            # Draw on image\n",
    "            res_plotted = r.plot()\n",
    "            st.image(res_plotted, caption=\"AI Analysis\", channels=\"BGR\")\n",
    "        else:\n",
    "            st.warning(\"No face detected. Try moving closer to the camera.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
