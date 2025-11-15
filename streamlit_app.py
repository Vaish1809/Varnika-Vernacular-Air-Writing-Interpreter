# streamlit_app.py
import os
import time
import json
import threading
from collections import deque, defaultdict
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image, ImageFont, ImageDraw
import google.generativeai as genai
import requests
import urllib.parse

import mediapipe as mp

LETTERS_FOLDER = "letters"
TRIGRAM_FILE = "trigram_dict.json"

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
    return [padded[i:i+3] for i in range(len(padded)-2)]

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

def extract_hindi_with_gemini(image_bgr):
    model = get_gemini_model()
    if model is None:
        return "Model not initialized", []

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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

def translate_text(text):
    try:
        encoded = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=hi&tl=en&dt=t&q={encoded}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return data[0][0][0]
    except Exception as e:
        print("Google translate failed:", e)

    # Fallback Gemini
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

# ---------- VIDEO PROCESSOR ----------
class HandwritingProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

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
        x = int(sum(pt[0] for pt in self.smooth_points)/len(self.smooth_points))
        y = int(sum(pt[1] for pt in self.smooth_points)/len(self.smooth_points))
        return (x, y)

    def draw_line(self, img, start_point, end_point, color, thickness):
        if start_point is not None and end_point is not None:
            cv2.line(img, start_point, end_point, color, thickness)

    def process_capture(self, blackboard_image):
        # save & call Gemini in background
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(LETTERS_FOLDER, f"letter_{ts}.png")
        cv2.imwrite(filename, blackboard_image)
        print("Saved letter:", filename)

        hindi_text, suggestions = extract_hindi_with_gemini(blackboard_image)
        self.predicted_character = hindi_text
        self.suggestions = suggestions
        self.is_processing = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        display_frame = img.copy()
        fingers_up = 0
        index_x = index_y = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = int(index_tip.x * img.shape[1])
                index_y = int(index_tip.y * img.shape[0])

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
                        self.draw_line(self.camera_screen, self.last_position, smooth_pos, (0, 0, 255), 10)
                        self.draw_line(self.blackboard, self.last_position, smooth_pos, (255, 255, 255), 10)
                        self.draw_line(display_frame, self.last_position, smooth_pos, (0, 0, 255), 10)
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
                    cv2.putText(display_frame, "CLEARED!", (150, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        else:
            self.two_fingers_previous = False

        cv2.putText(display_frame, f"Fingers: {fingers_up}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, "1: Write | 2: Capture | 4: Clear", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # blend drawing
        mask = cv2.cvtColor(self.camera_screen, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        display_frame[mask > 0] = (0, 0, 255)

        return display_frame

# ---------- STREAMLIT LAYOUT ----------
st.set_page_config(page_title="Realtime Hindi Air-Writing", layout="wide")
st.title("✋✍️ Realtime Hindi Handwriting (Air-Writing)")

col_video, col_text = st.columns([2, 1])

with col_video:
    webrtc_ctx = webrtc_streamer(
        key="hindi-live",
        video_transformer_factory=HandwritingProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
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
