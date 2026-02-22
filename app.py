import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import av
import queue
import random  # เพิ่มเพื่อช่วยให้เบราว์เซอร์ไม่จำแคชเสียง
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="SafeDrive AI", layout="centered")

# คิวสำหรับส่งข้อมูลสถานะง่วงนอน
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# --- 2. ฟังก์ชันเล่นเสียงเตือน (แก้ไขให้ทะลุข้อจำกัดเบราว์เซอร์มากขึ้น) ---
def play_alarm_sound(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            rid = random.randint(0, 1000) # ป้องกันเบราว์เซอร์มองว่าเป็นเสียงเดิมที่เคยเล่นแล้ว
            # ใช้ Iframe ผสม Audio เพื่อช่วยให้ระบบ Autoplay ทำงานง่ายขึ้น
            md = f"""
                <iframe src="data:audio/mp3;base64,{b64}" allow="autoplay" style="display:none"></iframe>
                <audio autoplay="true" id="audio_{rid}">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)

# --- 3. ตั้งค่าโมเดล Mediapipe ---
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

# --- 4. การตั้งค่าระบบเชื่อมต่อ (STUN Servers) ---
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", 
                  "stun:stun1.l.google.com:19302", 
                  "stun:stun2.l.google.com:19302", 
                  "stun:global.stun.twilio.com:3478"]}
    ]}
)

# --- 5. ส่วนประมวลผลวิดีโอ ---
class VideoProcessor:
    def __init__(self):
        self.counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        img_small = cv2.resize(img, (w//2, h//2))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        results = detector.detect(mp_image)
        
        drowsy_detected = False

        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]
            eye_dist = abs(face_landmarks[159].y - face_landmarks[145].y)

            if eye_dist < 0.012: 
                self.counter += 1
            else:
                self.counter = 0

            if self.counter > 6: 
                drowsy_detected = True

        # ส่งสถานะไปยัง Queue ใน session_state
        st.session_state.result_queue.put(drowsy_detected)

        if drowsy_detected:
            cv2.rectangle(img, (0,0), (w,h), (0,0,255), 20)
            cv2.putText(img, "DROWSY!", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 10)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. ส่วนการแสดงผล UI ---
st.markdown("""
    <style>
    .status-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ SafeDrive AI")
st.write("ระบบตรวจจับความปลอดภัยสำหรับผู้ขับขี่")

# --- จุดสำคัญ: ปุ่มเพื่อปลดล็อกเสียง ---
if 'audio_ready' not in st.session_state:
    st.session_state.audio_ready = False

if not st.session_state.audio_ready:
    st.warning("🔔 โปรดคลิกปุ่มด้านล่างเพื่อเปิดการแจ้งเตือนด้วยเสียง")
    if st.button("📢 ยืนยันเปิดเสียงแจ้งเตือน (คลิกที่นี่ก่อนเริ่ม)"):
        st.session_state.audio_ready = True
        st.rerun()

status_placeholder = st.empty()

# เริ่มระบบ WebRTC
webrtc_ctx = webrtc_streamer(
    key="safe-drive-v4",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- 7. ตรรกะการเช็กสถานะและเล่นเสียง ---
if webrtc_ctx.state.playing:
    is_drowsy = False
    try:
        # ดึงสถานะล่าสุด
        while not st.session_state.result_queue.empty():
            is_drowsy = st.session_state.result_queue.get_nowait()
    except:
        pass

    if is_drowsy:
        status_placeholder.markdown('<div class="status-box" style="background-color: #FF0000;">⚠️ ตรวจพบอาการง่วงนอน! ⚠️</div>', unsafe_allow_html=True)
        # เล่นเสียงเฉพาะเมื่อผู้ใช้กดยืนยันปุ่มเปิดเสียงแล้ว
        if st.session_state.audio_ready:
            play_alarm_sound("warningsound.mp3")
    else:
        status_placeholder.markdown('<div class="status-box" style="background-color: #28a745;">✅ สถานะ: ปกติ</div>', unsafe_allow_html=True)
else:
    status_placeholder.info("กรุณากดปุ่ม START ในกรอบวิดีโอเพื่อเริ่มการตรวจจับ")

st.divider()
st.info("💡 **คำแนะนำ:** หากเสียงไม่ดัง ให้ลองกดปุ่มยืนยันเปิดเสียงอีกครั้ง และวางมือถือให้เห็นใบหน้าชัดเจน")
