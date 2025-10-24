# app.py
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from fer import FER
import random
import cv2

# Streamlit setup
st.set_page_config(page_title="Facial Emotion Detection", layout="centered")

st.title("😊 Facial Emotion Detection")
st.write("Upload an image or use your webcam. The app detects emotion and shows a random quote to refresh your mind.")

# --- Quotes dictionary ---
QUOTES = {
    "happy": [
        "The purpose of our lives is to be happy. — Dalai Lama",
        "Happiness is not something ready made. It comes from your own actions. — Dalai Lama",
        "For every minute you are angry you lose sixty seconds of happiness. — Emerson",
        "Happiness depends upon ourselves. — Aristotle",
        "Do more of what makes you happy.",
        "Enjoy your life—it's all that matters. — Audrey Hepburn"
    ],
    "sad": [
        "Tough times never last, but tough people do. — Schuller",
        "This too shall pass.",
        "Sometimes when you’re in a dark place you’ve been planted. — Christine Caine",
        "The darker the night, the brighter the stars. — Dostoyevsky",
        "It’s okay to not be okay.",
        "You are allowed to feel messed up and inside out."
    ],
    "angry": [
        "For every minute you remain angry, you give up sixty seconds of peace. — Emerson",
        "Speak when you are angry and you’ll make the best speech you regret. — Bierce",
        "You don’t have to control your emotions, just don’t let them control you.",
        "Anger is an acid that harms its vessel. — Mark Twain",
        "Keep calm and let wise choices lead you.",
        "The best fighter is never angry. — Lao Tzu"
    ],
    "neutral": [
        "Breathe. It's only a bad day, not a bad life.",
        "Stay present. Everything else can wait.",
        "Balance is not something you find, it's something you create.",
        "Calmness is the cradle of power. — Josiah Holland",
        "Rest, reflect, renew.",
        "Small steps every day."
    ],
}

# --- Helper functions ---
def get_random_quote(emotion_label):
    lst = QUOTES.get(emotion_label, QUOTES["neutral"])
    return random.choice(lst)

def map_emotion(raw_emotion):
    raw = raw_emotion.lower()
    if raw in ["angry", "disgust"]:
        return "angry"
    if raw in ["sad"]:
        return "sad"
    if raw in ["happy"]:
        return "happy"
    if raw in ["neutral", "surprise"]:
        return "neutral"
    if raw in ["fear"]:
        return "sad"
    return "neutral"

# --- Sidebar input method ---
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input:", ("Upload Image", "Webcam (Camera)"))

image = None
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif input_method == "Webcam (Camera)":
    cam_image = st.camera_input("Take a picture")
    if cam_image:
        image = Image.open(cam_image).convert("RGB")

# --- Detection ---
if image is not None:
    st.image(image, caption="Input image", use_container_width=True)

    st.info("Detecting emotion... please wait ⏳")
    detector = FER(mtcnn=True)

    try:
        results = detector.detect_emotions(np.array(image))
        if not results:
            st.warning("😕 No face detected. Try another image or better lighting.")
        else:
            # Pick the most prominent face
            def box_area(box): x, y, w, h = box; return w * h
            main = max(results, key=lambda r: box_area(r["box"]))
            emotions = main["emotions"]
            sorted_em = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            raw_emotion, raw_score = sorted_em[0]
            mapped = map_emotion(raw_emotion)

            st.subheader(f"Detected Emotion: **{mapped.title()}** ({raw_emotion}: {raw_score:.2f})")

            # Chart
            labels = [k.title() for k, _ in sorted_em]
            scores = [v for _, v in sorted_em]
            fig, ax = plt.subplots()
            ax.barh(labels[::-1], scores[::-1])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence")
            ax.set_title("Emotion Probabilities")
            st.pyplot(fig)

            # Quote
            quote = get_random_quote(mapped)
            st.markdown("### 💬 Quote for You")
            st.info(quote)

            # Bounding box
            x, y, w, h = main["box"]
            img_copy = np.array(image).copy()
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.image(img_copy, caption="Detected Face", use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.write("👈 Upload or capture an image to begin.")

st.sidebar.markdown("---")
st.sidebar.caption("Using pretrained FER model (mtcnn=True).")
