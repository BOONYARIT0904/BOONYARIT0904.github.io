import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import os
import base64
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision


# --- 1. ฟังก์ชันเล่นเสียง ---
def play_alarm_sound(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
        st.markdown(md, unsafe_allow_html=True)
    else:
        st.error(f"ไฟล์เสียง {file_path} หายไป!")


# --- 2. ตั้งค่าโมเดล ---
model_path = "face_landmarker.task"

if not os.path.exists(model_path):
    st.error(f"ไม่พบไฟล์โมเดลที่: {model_path}")
    st.stop()


@st.cache_resource
def get_mediapipe_detector(model_path_arg):
    base_options = tasks.BaseOptions(model_asset_path=model_path_arg)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=False,
                                           output_facial_transformation_matrixes=False,
                                           num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)


detector = get_mediapipe_detector(model_path)

# --- 3. UI Settings ---
st.title("Driver eye")
st.sidebar.header("Settings")

if 'COUNTER' not in st.session_state: st.session_state.COUNTER = 0
if 'ALARM_ON' not in st.session_state: st.session_state.ALARM_ON = False
if 'running' not in st.session_state: st.session_state.running = False

EYE_CLOSED_LIMIT = st.sidebar.slider("Sensitivity", 5, 30, 10)
alarm_sound_enabled = st.sidebar.checkbox("Enable Alarm", value=True)

start_btn = st.sidebar.button("Start Detection")
stop_btn = st.sidebar.button("Stop Detection")

st.subheader("Real-time Video Feed")
video_placeholder = st.empty()
status_placeholder = st.empty()


# --- 4. ฟังก์ชันหลัก ---
def process_video_stream():
    if 'cap' not in st.session_state or st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

    while st.session_state.get('running', False):
        ret, frame = st.session_state.cap.read()
        if not ret: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = detector.detect(mp_image)
        current_status = "EYES OPEN"

        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]

            # --- ส่วนที่เพิ่มใหม่: คำนวณหาขอบสี่เหลี่ยมใบหน้า ---
            h, w, _ = frame.shape  # ดึงขนาดของวิดีโอมา
            x_min = int(min([lm.x for lm in face_landmarks]) * w)
            y_min = int(min([lm.y for lm in face_landmarks]) * h)
            x_max = int(max([lm.x for lm in face_landmarks]) * w)
            y_max = int(max([lm.y for lm in face_landmarks]) * h)

            # วาดสี่เหลี่ยมสีแดง (BGR: 0, 0, 255) ความหนา 2
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            # --------------------------------------------------

            # (โค้ดเดิมของคุณที่คำนวณ eye_distance ต่อจากนี้...)
            p_upper = face_landmarks[159]
            p_lower = face_landmarks[145]
            # ...
            # คำนวณระยะห่างเปลือกตา (EAR แบบง่าย)
            eye_distance = abs(face_landmarks[159].y - face_landmarks[145].y)

            if eye_distance < 0.01:  # หลับตา
                st.session_state.COUNTER += 1
                if st.session_state.COUNTER >= EYE_CLOSED_LIMIT:
                    current_status = "!!! DROWSY ALERT !!!"
                    if not st.session_state.ALARM_ON and alarm_sound_enabled:
                        play_alarm_sound("warningsound.mp3.mp3")
                        st.session_state.ALARM_ON = True
                else:
                    current_status = "Eyes Closed"
            else:  # ลืมตา
                st.session_state.COUNTER = 0
                st.session_state.ALARM_ON = False
                current_status = "EYES OPEN"

        status_placeholder.markdown(f"Status: **{current_status}**")
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(0.01)


if start_btn:
    st.session_state.running = True
    process_video_stream()

if stop_btn:
    st.session_state.running = False
    if 'cap' in st.session_state and st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.rerun()

# --- ตกแต่งหน้าเว็บด้วย CSS ---
st.set_page_config(page_title="SafeDrive AI", layout="wide") # ตั้งชื่อ Tab และขยายหน้าจอ

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stCheckbox {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
    .video-container {
        border: 5px solid #262730;
        border-radius: 15px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ส่วนหัวของแอป ---
st.title("🛡️ SafeDrive AI")
st.subheader("ระบบ AI ตรวจจับอาการง่วงนอนอัจฉริยะ")
st.divider()

# สร้าง Column แบ่งหน้าจอ
col1, col2 = st.columns([1, 2])  # col1 กว้าง 1 ส่วน, col2 กว้าง 2 ส่วน

with col1:
    st.info(
        "💡 **วิธีใช้งาน**\n1. ปรับค่าความไว (Sensitivity)\n2. กด Start เพื่อเริ่มตรวจจับ\n3. ระบบจะเตือนเมื่อคุณหลับตานานเกินไป")



