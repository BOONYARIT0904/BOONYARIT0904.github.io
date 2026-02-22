import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import base64
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- ตั้งค่าหน้าเว็บ (ต้องอยู่บรรทัดแรกของโค้ด) ---
st.set_page_config(page_title="SafeDrive AI", layout="centered")

# --- 1. ฟังก์ชันเล่นเสียง ---
def play_alarm_sound(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)

# --- 2. ตั้งค่าโมเดล AI ---
model_path = "face_landmarker.task"

@st.cache_resource
def get_mediapipe_detector():
    base_options = tasks.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

detector = get_mediapipe_detector()

# --- 3. UI สำหรับผู้สูงอายุ ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 80px;
        font-size: 30px !important;
        border-radius: 20px;
        font-weight: bold;
    }
    .status-box {
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ ระบบช่วยเตือนคนขับ")

if 'COUNTER' not in st.session_state: st.session_state.COUNTER = 0
if 'ALARM_ON' not in st.session_state: st.session_state.ALARM_ON = False

# ส่วนแสดงสถานะตัวโตๆ
status_placeholder = st.empty()

# --- 4. ฟังก์ชันประมวลผลวิดีโอ (WebRTC) ---
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # แปลงเป็นภาพสำหรับ Mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results = detector.detect(mp_image)
        
        current_status = "NORMAL"

        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]
            
            # คำนวณระยะห่างเปลือกตา
            eye_distance = abs(face_landmarks[159].y - face_landmarks[145].y)

            if eye_distance < 0.015: # ค่าความไว (ปรับตามความเหมาะสม)
                st.session_state.COUNTER += 1
                if st.session_state.COUNTER > 10:
                    current_status = "DROWSY"
            else:
                st.session_state.COUNTER = 0
                st.session_state.ALARM_ON = False

        # วาดข้อความแจ้งเตือนลงในวิดีโอ
        if current_status == "DROWSY":
            cv2.putText(img, "!!! WARNING !!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. ปุ่มเริ่ม/หยุด และช่องวิดีโอ ---
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="driver-monitor",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# จัดการเสียงเตือน
if webrtc_ctx.state.playing and st.session_state.COUNTER > 10:
    status_placeholder.markdown('<div class="status-box" style="background-color: red;">⚠️ ง่วงนอน! หยุดพักด่วน ⚠️</div>', unsafe_allow_html=True)
    if not st.session_state.ALARM_ON:
        play_alarm_sound("warningsound.mp3")
        st.session_state.ALARM_ON = True
elif webrtc_ctx.state.playing:
    status_placeholder.markdown('<div class="status-box" style="background-color: green;">✅ ปกติ (ขับขี่ปลอดภัย)</div>', unsafe_allow_html=True)
else:
    status_placeholder.info("กดปุ่ม 'Start' ด้านล่างวิดีโอเพื่อเริ่มใช้งาน")

st.info("💡 คำแนะนำ: วางมือถือให้เห็นใบหน้าชัดเจน และแตะหน้าจอ 1 ครั้งเพื่อให้เสียงทำงาน")
