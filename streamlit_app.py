import os
import time
import json
import threading
from collections import deque, defaultdict
from datetime import datetime

import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from PIL import Image, ImageFont, ImageDraw
import google.generativeai as genai
import requests
import urllib.parse
import mediapipe as mp

LETTERS_FOLDER = "letters"
TRIGRAM_FILE = "trigram_dict.json"

# ---------- WebRTC ICE config ----------
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        # If you configure TURN, add it here using st.secrets["TURN_USERNAME"], etc.
    ]
}

# ---------- FONT LOADING ----------
def load_hindi_font():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    candidates = []

    static_dir = os.path.join(base_dir, "fonts", "static")
    if os.path.isdir(static_dir):
        for fname in os.listdir(static_dir):
            if fname.lower().endswith((".ttf", ".otf")):
                candidates.append(os.path.join(static_dir, fname))

    fonts_dir = os.path.join(base_dir, "fonts")
    if os.path.isdir(fonts_dir):
        for fname in os.listdir(fonts_dir):
            if fname.lower().endswith((".ttf", ".otf")):
                p = os.path.join(fonts_dir, fname)
                if p not in candidates:
                    candidates.append(p)

    system_candidates = [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\mangal.ttf",
    ]
    for p in system_candidates:
        if os.path.exists(p) and p not in candidates:
            candidates.append(p)

    test_img = Image.new("RGB", (100, 100), (0, 0, 0))
    draw = ImageDraw.Draw(test_img)

    for p in candidates:
        try:
            font = ImageFont.truetype(p, 32)
            draw.textbbox((0, 0), "अ", font=font)
            print("Loaded font:", p)
            return font
        except Exception as e:
            print("Failed font", p, e)

    print("WARNING: using default font (may not display Hindi properly)")
    return ImageFont.load_default()

FONT = load_hindi_font()

# ---------- TRIGRAMS ----------
with open(TRIGRAM_FILE, "r", encoding="utf-8") as f:
    TRIGRAM_DICT = json.load(f)

def generate_trigrams(word):
    padded = f"#{word}#"
    return [padded[i:i+3] for i in range(len(padded) - 2)]

def suggest_words(input_text, trigram_dict, top_n=3):
    input_trigrams = set(generate_trigrams(input_text))
    candidate_counts = defaultdict(int)
    for tri in input_trigrams:
        for word in trigram_dict.get(tri, []):
            candidate_counts[word] += 1
    if not candidate_counts:
        return []
    sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_candidates[:top_n]]

# ---------- GEMINI ----------
@st.cache_resource
def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.warning("GEMINI_API_KEY not set in secrets. Gemini features limited.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        return None

def extract_hindi_with_gemini(image_bgr: np.ndarray):
    """
    image_bgr: uint8 numpy array (H, W, 3) in BGR format.
    """
    model = get_gemini_model()
    if model is None:
        return "Model not initialized", []

    # BGR -> RGB without cv2
    image_rgb = image_bgr[:, :, ::-1]
    pil_image = Image.fromarray(image_rgb)

    prompt = (
        "Extract the Hindi handwritten character/word present in this image. "
        "Return ONLY the Hindi character(s) without any English text, punctuation, "
        "or explanations. Just the raw Hindi character(s)."
    )
    contents = [prompt, pil_image]

    try:
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=50,
                temperature=0.2,
                top_p=0.95,
            ),
        )
        if not response.text:
            return "No text detected", []

        import re
        cleaned = response.text.strip()
        pattern = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF]+")
        matches = pattern.findall(cleaned)
        if matches:
            hindi = "".join(matches)
        else:
            hindi = cleaned.strip()

        if not hindi:
            return "No Hindi text detected", []

        suggestions = [hindi]
        trig_sug = suggest_words(hindi, TRIGRAM_DICT, top_n=10)
        for s in trig_sug:
            if s not in suggestions:
                suggestions.append(s)

        return hindi, suggestions
    except Exception as e:
        return f"Error: {e}", []

def translate_text(text: str) -> str:
    # 1) Try Google Translate free endpoint
    try:
        encoded = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=hi&tl=en&dt=t&q={encoded}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return data[0][0][0]
    except Exception as e:
        print("Google translate failed:", e)

    # 2) Fallback Gemini
    model = get_gemini_model()
    if model:
        try:
            prompt = f"Translate the following Hindi text to English. Respond with only the English translation.\n\n{text}"
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=60,
                    temperature=0.1,
                    top_p=0.95,
                ),
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print("Gemini translate failed:", e)

    return "No translation available"

# ---------- SIMPLE LINE DRAWING WITHOUT OPENCV ----------
def draw_thick_line(img: np.ndarray, start, end, color, thickness: int):
    """
    Draw a simple thick line on img (H,W,3) using numpy only.
    start, end: (x, y)
    color: (B, G, R) or (R, G, B) – just be consistent.
    """
    if start is None or end is None:
        return

    x0, y0 = start
    x1, y1 = end
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    num = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.linspace(x0, x1, num).astype(int)
    ys = np.linspace(y0, y1, num).astype(int)

    h, w, _ = img.shape
    r = max(1, thickness // 2)

    for x, y in zip(xs, ys):
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        x_min = max(0, x - r)
        x_max = min(w, x + r)
        y_min = max(0, y - r)
        y_max = min(h, y + r)
        img[y_min:y_max, x_min:x_max] = color

# ---------- VIDEO PROCESSOR ----------
class HandwritingProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

        # 480x640 is enough and lighter than 720p
        self.camera_screen = np.zeros((480, 640, 3), dtype=np.uint8)
        self.blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        self.drawing = False
        self.last_position = None
        self.smooth_points = deque(maxlen=8)
        self.two_fingers_previous = False

        self.predicted_character = ""
        self.suggestions = []
        self.translation = ""
        self.is_processing = False

        self.processing_thread = None

        os.makedirs(LETTERS_FOLDER, exist_ok=True)

    def get_smooth_point(self, p):
        self.smooth_points.append(p)
        if len(self.smooth_points) < 3:
            return p
        x = int(sum(pt[0] for pt in self.smooth_points) / len(self.smooth_points))
        y = int(sum(pt[1] for pt in self.smooth_points) / len(self.smooth_points))
        return (x, y)

    def process_capture(self, blackboard_image: np.ndarray):
        # Save using PIL (no cv2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(LETTERS_FOLDER, f"letter_{ts}.png")
        img_rgb = blackboard_image[:, :, ::-1]  # BGR -> RGB for PIL
        Image.fromarray(img_rgb).save(filename)
        print("Saved letter:", filename)

        hindi_text, suggestions = extract_hindi_with_gemini(blackboard_image)
        self.predicted_character = hindi_text
        self.suggestions = suggestions
        self.is_processing = False

    def recv(self, frame):
        # Get frame in BGR (as ndarray)
        img = frame.to_ndarray(format="bgr24")

        # Flip horizontally (mirror)
        img = img[:, ::-1, :]

        # Resize to 480x640 region we use (simple center crop/resize if needed)
        h, w, _ = img.shape
        target_h, target_w = self.camera_screen.shape[:2]

        # Simple resize with numpy (nearest-neighbor) to avoid cv2.resize
        # Compute scale factors
        scale_y = h / target_h
        scale_x = w / target_w
        ys = (np.arange(target_h) * scale_y).astype(int)
        xs = (np.arange(target_w) * scale_x).astype(int)
        img_resized = img[ys][:, xs]

        frame_rgb = img_resized[:, :, ::-1]  # BGR -> RGB for mediapipe
        results = self.hands.process(frame_rgb)

        display_frame = img_resized.copy()
        fingers_up = 0
        index_x = index_y = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = int(index_tip.x * target_w)
                index_y = int(index_tip.y * target_h)

                fingers_up = 0
                if index_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                    fingers_up += 1
                if hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                    fingers_up += 1
                if hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y:
                    fingers_up += 1
                if hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y:
                    fingers_up += 1

                if fingers_up == 1:
                    self.drawing = True
                    smooth_pos = self.get_smooth_point((index_x, index_y))

                    if self.last_position:
                        draw_thick_line(self.camera_screen, self.last_position, smooth_pos, (0, 0, 255), 10)
                        draw_thick_line(self.blackboard, self.last_position, smooth_pos, (255, 255, 255), 10)
                        draw_thick_line(display_frame, self.last_position, smooth_pos, (0, 0, 255), 10)
                    self.last_position = smooth_pos
                    self.two_fingers_previous = False

                elif fingers_up == 2:
                    self.drawing = False
                    self.last_position = None
                    self.smooth_points.clear()

                    if not self.two_fingers_previous:
                        bb_copy = self.blackboard.copy()
                        if np.sum(bb_copy) > 0 and not self.is_processing:
                            self.is_processing = True
                            self.processing_thread = threading.Thread(
                                target=self.process_capture, args=(bb_copy,), daemon=True
                            )
                            self.processing_thread.start()
                        self.two_fingers_previous = True

                elif fingers_up == 4:
                    self.drawing = False
                    self.last_position = None
                    self.smooth_points.clear()
                    self.camera_screen[:] = 0
                    self.blackboard[:] = 0
                    self.two_fingers_previous = False
                    self.predicted_character = ""
                    self.suggestions = []

        else:
            self.two_fingers_previous = False

        # Overlay drawing: wherever camera_screen has non-zero pixels, paint red on display_frame
        mask = np.any(self.camera_screen != 0, axis=2)
        display_frame[mask] = (0, 0, 255)

        return display_frame

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Realtime Hindi Air-Writing", layout="wide")
st.title("✋✍️ Realtime Hindi Handwriting (Air-Writing)")

col_video, col_text = st.columns([2, 1])

with col_video:
    webrtc_ctx = webrtc_streamer(
        key="hindi-live",
        mode=WebRtcMode.SENDRECV,          # <- use enum, not string
        video_processor_factory=HandwritingProcessor,  # <- new API name
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,             # <- replaces async_transform
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True,
            "playsInline": True,
            "style": {"width": "100%", "height": "auto"},
        },
    )

with col_text:
    st.subheader("Recognition & Translation")
    info = st.empty()
    sugg_box = st.empty()
    translate_btn_slot = st.empty()
    translation_slot = st.empty()

    if webrtc_ctx and webrtc_ctx.video_transformer:
        vt: HandwritingProcessor = webrtc_ctx.video_transformer

        if vt.is_processing:
            info.info("Processing captured writing with Gemini...")
        elif vt.predicted_character:
            info.markdown(f"**Detected Hindi:** {vt.predicted_character}")
        else:
            info.write("Write in the air with 1 finger. Show 2 fingers to capture. 4 fingers to clear.")

        if vt.suggestions:
            sugg_box.markdown("**Suggestions:** " + ", ".join(vt.suggestions[:5]))
            chosen = st.selectbox("Choose word to translate", vt.suggestions, key="chosen_word")

            if translate_btn_slot.button("Translate selected word"):
                with st.spinner("Translating..."):
                    english = translate_text(chosen)
                translation_slot.markdown(f"**English:** {english}")
        else:
            sugg_box.write("No suggestions yet.")
