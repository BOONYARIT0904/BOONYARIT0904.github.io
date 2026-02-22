import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import av
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="SafeDrive AI", layout="centered")

# --- 1. ฟังก์ชันเล่นเสียงเตือน (Base64) ---
def play_alarm_sound(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)

# --- 2. ตั้งค่าโมเดล Mediapipe ---
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

# --- 3. การตั้งค่าระบบเชื่อมต่อ (STUN Servers แบบจัดเต็ม) ---
# ช่วยแก้ปัญหา Connection taking longer โดยใช้ทางผ่านหลายช่องทาง
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", 
                  "stun:stun1.l.google.com:19302", 
                  "stun:stun2.l.google.com:19302", 
                  "stun:stun3.l.google.com:19302", 
                  "stun:stun4.l.google.com:19302",
                  "stun:global.stun.twilio.com:3478"]}
    ]}
)

# --- 4. ปรับแต่ง UI สำหรับผู้ใช้งาน/ผู้สูงอายุ ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 70px;
        font-size: 25px !important;
        border-radius: 15px;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .status-box {
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ SafeDrive AI")
st.write("ระบบตรวจจับความปลอดภัยขณะขับขี่")

if 'COUNTER' not in st.session_state: st.session_state.COUNTER = 0
if 'ALARM_ON' not in st.session_state: st.session_state.ALARM_ON = False

status_placeholder = st.empty()

# --- 5. ส่วนประมวลผลวิดีโอ ---
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # ลดขนาดภาพเล็กน้อยเพื่อให้ส่งข้อมูลเร็วขึ้นบนเน็ตมือถือ
        h, w = img.shape[:2]
        img_small = cv2.resize(img, (w//2, h//2))
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        results = detector.detect(mp_image)
        
        is_drowsy = False

        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]
            # วัดระยะห่างเปลือกตา (จุด 159 และ 145)
            eye_dist = abs(face_landmarks[159].y - face_landmarks[145].y)

            if eye_dist < 0.012: # ปรับค่าความไวตรงนี้ (น้อยลง = ต้องหลับตาเนียนขึ้น)
                st.session_state.COUNTER += 1
                if st.session_state.COUNTER > 8: # จำนวนเฟรมที่หลับตาติดต่อกัน
                    is_drowsy = True
            else:
                st.session_state.COUNTER = 0
                st.session_state.ALARM_ON = False

        # วาดสัญลักษณ์เตือนบนจอ
        if is_drowsy:
            cv2.rectangle(img, (0,0), (w,h), (0,0,255), 20)
            cv2.putText(img, "DROWSY!", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 10)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. สร้างช่องวิดีโอ WebRTC ---
webrtc_ctx = webrtc_streamer(
    key="safe-drive-v3",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False
    },
    async_processing=True,
)

# --- 7. ตรรกะการเตือนและการแสดงสถานะ ---
if webrtc_ctx.state.playing:
    if st.session_state.COUNTER > 8:
        status_placeholder.markdown('<div class="status-box" style="background-color: #FF0000; color: white;">⚠️ ตื่นๆ! ตรวจพบอาการง่วงนอน ⚠️</div>', unsafe_allow_html=True)
        if not st.session_state.ALARM_ON:
            play_alarm_sound("warningsound.mp3")
            st.session_state.ALARM_ON = True
    else:
        status_placeholder.markdown('<div class="status-box" style="background-color: #28a745; color: white;">✅ สถาณะ: ปกติ</div>', unsafe_allow_html=True)
else:
    status_placeholder.warning("กรุณากดปุ่ม START เพื่อเปิดกล้อง")

st.markdown("---")
st.info("💡 **คำแนะนำสำหรับผู้ใช้:**\n1. วางมือถือให้เห็นใบหน้าชัดเจน\n2. หากกล้องไม่ขึ้น ให้ตรวจสอบว่ากด 'อนุญาต (Allow)' หรือยัง\n3. เสียงจะทำงานเมื่อท่านแตะหน้าจอเว็บอย่างน้อย 1 ครั้ง")
