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
st.set_page_config(
    page_title="Facial Emotion Detection", 
    layout="centered",
    page_icon="😊",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful gradient background and glass morphism
st.markdown("""
<style>
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    /* Main container with glass morphism effect */
    .main-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Sidebar glass effect */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #ffffff, #f0f0f0, #e0e0e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2rem;
        font-weight: 500;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Card styling with glass effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Quote box styling */
    .quote-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center;
        font-size: 1.2rem;
        font-style: italic;
    }
    
    /* Emotion result boxes */
    .emotion-result {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        background: rgba(255, 255, 255, 0.2);
        border-left: 6px solid;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .happy { 
        border-color: #4CAF50; 
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(255, 255, 255, 0.1));
    }
    .sad { 
        border-color: #2196F3; 
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.2), rgba(255, 255, 255, 0.1));
    }
    .angry { 
        border-color: #f44336; 
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.2), rgba(255, 255, 255, 0.1));
    }
    .neutral { 
        border-color: #9E9E9E; 
        background: linear-gradient(135deg, rgba(158, 158, 158, 0.2), rgba(255, 255, 255, 0.1));
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Text input styling */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        border-radius: 10px;
    }
    
    /* Chart background */
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main content container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">😊 Facial Emotion Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image or use your webcam. The app detects emotion and shows a random quote to refresh your mind.</p>', unsafe_allow_html=True)

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
        "Sometimes when you're in a dark place you've been planted. — Christine Caine",
        "The darker the night, the brighter the stars. — Dostoyevsky",
        "It's okay to not be okay.",
        "You are allowed to feel messed up and inside out."
    ],
    "angry": [
        "For every minute you remain angry, you give up sixty seconds of peace. — Emerson",
        "Speak when you are angry and you'll make the best speech you regret. — Bierce",
        "You don't have to control your emotions, just don't let them control you.",
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

# --- Sidebar with enhanced styling ---
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9)); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 1rem;'>
    <h3 style='margin:0;'>🎯 Input Method</h3>
</div>
""", unsafe_allow_html=True)

input_method = st.sidebar.radio("Choose input:", ("Upload Image", "Webcam (Camera)"), label_visibility="collapsed")

# Add some spacing
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Instructions in sidebar
st.sidebar.markdown("""
<div style='background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 15px; border-left: 4px solid #667eea; backdrop-filter: blur(10px);'>
    <h4 style='margin-top:0; color: white;'>💡 Tips for best results:</h4>
    <ul style='margin-bottom:0; color: rgba(255,255,255,0.9);'>
        <li>Ensure good lighting</li>
        <li>Face the camera directly</li>
        <li>Remove sunglasses/hats</li>
        <li>Use a clear, high-quality image</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Main content area ---
image = None

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

if input_method == "Upload Image":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📁 Upload Image")
        uploaded_file = st.file_uploader("Choose an image file (jpg/png)", type=["jpg", "jpeg", "png"], 
                                        help="Choose a clear image with a visible face", label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

elif input_method == "Webcam (Camera)":
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("### 📸 Webcam Capture")
        st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.9);'>Position your face in the center and click the photo</p>", 
                   unsafe_allow_html=True)
        cam_image = st.camera_input("Take a picture", label_visibility="collapsed")
        if cam_image:
            image = Image.open(cam_image).convert("RGB")

st.markdown('</div>', unsafe_allow_html=True)

# --- Detection ---
if image is not None:
    # Display uploaded image in a card-like container
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📷 Your Image")
        st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Progress bar for detection
    with st.spinner("🔍 Analyzing facial expression..."):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        
        detector = FER(mtcnn=True)
        
        try:
            results = detector.detect_emotions(np.array(image))
            if not results:
                st.markdown("""
                <div style='background: rgba(255,255,255,0.2); padding: 2rem; border-radius: 20px; text-align: center; border-left: 5px solid #ff6b6b; backdrop-filter: blur(10px);'>
                    <h2 style='color: white;'>❌ No Face Detected</h2>
                    <p style='color: rgba(255,255,255,0.9);'>Try another image with better lighting or ensure your face is clearly visible</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Pick the most prominent face
                def box_area(box): x, y, w, h = box; return w * h
                main = max(results, key=lambda r: box_area(r["box"]))
                emotions = main["emotions"]
                sorted_em = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                raw_emotion, raw_score = sorted_em[0]
                mapped = map_emotion(raw_emotion)
                
                # Emotion result with colored styling
                emotion_icons = {
                    "happy": "😊", 
                    "sad": "😢", 
                    "angry": "😠", 
                    "neutral": "😐"
                }
                
                st.markdown(f"""
                <div class='emotion-result {mapped}'>
                    <h2 style='color: white;'>{emotion_icons[mapped]} Detected Emotion: <strong>{mapped.title()}</strong></h2>
                    <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem;'>Raw emotion: {raw_emotion} (confidence: {raw_score:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Chart with better styling
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Emotion Analysis")
                labels = [k.title() for k, _ in sorted_em]
                scores = [v for _, v in sorted_em]
                
                # Create a more attractive chart
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D', '#6A0572', '#AB83A1']
                bars = ax.barh(labels[::-1], scores[::-1], color=colors[:len(labels)])
                ax.set_xlim(0, 1)
                ax.set_xlabel("Confidence Score", fontsize=12, color='white')
                ax.set_ylabel("Emotions", fontsize=12, color='white')
                ax.set_title("Emotion Probability Distribution", fontsize=14, fontweight='bold', color='white')
                ax.grid(axis='x', alpha=0.3)
                ax.tick_params(colors='white')
                
                # Add value labels on bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width:.2f}', ha='left', va='center', fontweight='bold', color='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Quote in styled box
                st.markdown("### 💫 Quote for You")
                quote = get_random_quote(mapped)
                st.markdown(f'<div class="quote-box">{quote}</div>', unsafe_allow_html=True)
                
                # Bounding box visualization
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 👁️ Face Detection")
                x, y, w, h = main["box"]
                img_copy = np.array(image).copy()
                # Enhanced bounding box
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # Add label
                cv2.putText(img_copy, f'{mapped.title()} ({raw_score:.2f})', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(img_copy, caption="Detected Face with Emotion", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                    
        except Exception as e:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.2); padding: 2rem; border-radius: 20px; text-align: center; border-left: 5px solid #ff6b6b; backdrop-filter: blur(10px);'>
                <h2 style='color: white;'>⚠️ Processing Error</h2>
                <p style='color: rgba(255,255,255,0.9);'>Details: {e}</p>
                <p style='color: rgba(255,255,255,0.9);'>Please try with a different image.</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Welcome state with features
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem; background: rgba(255,255,255,0.15); border-radius: 25px; margin: 2rem 0; backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.2);'>
        <h2 style='color: white;'>🎭 Ready to Discover Your Emotions?</h2>
        <p style='font-size: 1.2rem; color: rgba(255,255,255,0.9); max-width: 600px; margin: 0 auto;'>
            Upload a photo or use your webcam to detect facial emotions and get personalized quotes!
        </p>
        <div style='display: flex; justify-content: center; gap: 3rem; margin-top: 3rem; flex-wrap: wrap;'>
            <div style='text-align: center; color: white;'>
                <div style='font-size: 3rem;'>📁</div>
                <strong style='font-size: 1.2rem;'>Upload Image</strong>
                <p style='font-size: 1rem; color: rgba(255,255,255,0.8);'>JPG, PNG formats</p>
            </div>
            <div style='text-align: center; color: white;'>
                <div style='font-size: 3rem;'>📸</div>
                <strong style='font-size: 1.2rem;'>Webcam Capture</strong>
                <p style='font-size: 1rem; color: rgba(255,255,255,0.8);'>Real-time detection</p>
            </div>
            <div style='text-align: center; color: white;'>
                <div style='font-size: 3rem;'>💬</div>
                <strong style='font-size: 1.2rem;'>Get Quotes</strong>
                <p style='font-size: 1rem; color: rgba(255,255,255,0.8);'>Personalized messages</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.8);'>
    <p style='font-size: 1.1rem;'>Built with ❤️ using Streamlit & FER | Emotion Detection AI</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.8); font-size: 0.9rem;'>
    <p>Using pretrained FER model (mtcnn=True)</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem;'>
    <p>✨ Magical Emotion Detection ✨</p>
</div>
""", unsafe_allow_html=True)