import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import av
import queue
import random
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="SafeDrive AI", layout="centered")

# สร้าง Queue เพื่อส่งค่าระหว่าง Thread (ป้องกัน AttributeError)
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# --- 2. ฟังก์ชันเล่นเสียงเตือน (ปลดล็อก Autoplay) ---
def play_alarm_sound(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            rid = random.randint(0, 9999)
            md = f"""
                <iframe src="data:audio/mp3;base64,{b64}" allow="autoplay" style="display:none" id="f_{rid}"></iframe>
                <audio autoplay="true" id="a_{rid}">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)

# --- 3. โหลด AI Model ---
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

# --- 4. ตั้งค่า WebRTC (STUN Servers เพื่อทะลุ Wi-Fi) ---
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", 
                  "stun:stun1.l.google.com:19302", 
                  "stun:global.stun.twilio.com:3478"]}
    ]}
)

# --- 5. Class ประมวลผลวิดีโอ ---
class VideoProcessor:
    def __init__(self):
        self.counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # ลดขนาดภาพเพื่อความลื่นและลดอาการค้างบน Wi-Fi
        img_small = cv2.resize(img, (w//2, h//2))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        results = detector.detect(mp_image)
        
        drowsy_detected = False

        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]
            # ระยะห่างเปลือกตา (EAR)
            eye_dist = abs(face_landmarks[159].y - face_landmarks[145].y)

            if eye_dist < 0.012: # ปรับความไว
                self.counter += 1
            else:
                self.counter = 0

            if self.counter > 6: 
                drowsy_detected = True

        st.session_state.result_queue.put(drowsy_detected)

        if drowsy_detected:
            cv2.rectangle(img, (0,0), (w,h), (0,0,255), 20)
            cv2.putText(img, "DROWSY!", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 10)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. UI และปุ่มกด ---
st.title("🛡️ SafeDrive AI")
st.write("ระบบตรวจจับความปลอดภัย (เวอร์ชันสมบูรณ์)")

# ปุ่มปลดล็อกเสียง
if 'audio_unlocked' not in st.session_state:
    st.session_state.audio_unlocked = False

if not st.session_state.audio_unlocked:
    if st.button("📢 คลิกที่นี่เพื่อเปิดระบบเสียง (ทำครั้งเดียวก่อนเริ่ม)"):
        st.session_state.audio_unlocked = True
        st.rerun()

status_placeholder = st.empty()

# ส่วนของกล้อง (ปรับ Resolution ให้ต่ำลงเพื่อป้องกันอาการค้าง)
webrtc_ctx = webrtc_streamer(
    key="safedrive-v5",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"width": 320, "height": 240, "frameRate": 15},
        "audio": False
    },
    async_processing=True,
)

# --- 7. เช็กผลและเล่นเสียง ---
if webrtc_ctx.state.playing:
    is_drowsy = False
    try:
        while not st.session_state.result_queue.empty():
            is_drowsy = st.session_state.result_queue.get_nowait()
    except:
        pass

    if is_drowsy:
        status_placeholder.markdown('<div style="background-color:#FF0000; color:white; padding:20px; border-radius:15px; text-align:center; font-size:30px; font-weight:bold;">⚠️ ตื่นๆ! ตรวจพบอาการง่วง ⚠️</div>', unsafe_allow_html=True)
        if st.session_state.audio_unlocked:
            play_alarm_sound("warningsound.mp3")
    else:
        status_placeholder.markdown('<div style="background-color:#28a745; color:white; padding:20px; border-radius:15px; text-align:center; font-size:30px; font-weight:bold;">✅ สถานะ: ปกติ</div>', unsafe_allow_html=True)
else:
    status_placeholder.warning("กรุณากด START เพื่อเปิดกล้อง")

st.divider()
st.info("💡 หากกล้องค้างหรือขึ้น Connecting นานเกินไป ให้ลองสลับจาก Wi-Fi เป็นเน็ตมือถือ (4G/5G) ครับ")
