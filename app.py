import streamlit as st
from recognition.model import load_and_train_model, predict_digit
from recognition.image_utils import load_and_preprocess_image
from recognition.explainability import explain_prediction
from recognition.feedback import log_feedback
import plotly.express as px
import numpy as np

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import cv2

# 1. === Настройка страницы ===
st.set_page_config(
    page_title="NeuroDigits | AI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. === Стили Neomorphic UI ===
st.markdown("""
<style>
:root {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --accent: #2dd4bf;
    --text: #e2e8f0;
}

html, body, .main {
    background: var(--bg-primary) !important;
    color: var(--text) !important;
}

/* Неоморфные карточки */
.stApp, .block-container {
    background: transparent !important;
}

.custom-card {
    background: var(--bg-secondary);
    border-radius: 16px;
    box-shadow: 
        8px 8px 16px rgba(0,0,0,0.3),
        -8px -8px 16px rgba(72, 79, 96, 0.1);
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Холст с эффектом стекла */
.canvas-glass {
    border-radius: 16px;
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(4px);
    box-shadow:
        inset 2px 2px 5px rgba(0,0,0,0.2),
        inset -2px -2px 5px rgba(72, 79, 96, 0.1);
}

/* Кнопки с градиентом */
.stButton>button {
    background: linear-gradient(135deg, #2dd4bf 0%, #1e40af 100%);
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# 3. === Основной интерфейс ===
cols = st.columns([0.6, 0.4], gap="large")

with cols[0]:
    st.markdown("""
    <div class="custom-card">
        <h2 style="color: var(--accent); margin-top: 0;">DRAWING CANVAS</h2>
    """, unsafe_allow_html=True)
    
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#1E293B",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
        className="canvas-glass"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with cols[1]:
    st.markdown("""
    <div class="custom-card">
        <h2 style="color: var(--accent); margin-top: 0;">AI ANALYSIS</h2>
    """, unsafe_allow_html=True)
    
    if st.button("Recognize Digit", use_container_width=True):
        if canvas.image_data is not None:
            # Обработка изображения
            img = cv2.resize(canvas.image_data.astype('uint8'), (28, 28))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Здесь должна быть ваша модель
            prediction = np.random.rand(10)  # Заглушка
            predicted_digit = np.argmax(prediction)
            
            # Визуализация
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(range(10), prediction, color='#2dd4bf')
            ax.set_title("Model Confidence Levels", color="white")
            ax.set_facecolor("#1E293B")
            fig.patch.set_facecolor("#1E293B")
            ax.tick_params(colors='white')
            
            st.pyplot(fig)
            st.success(f"**Predicted Digit:** `{predicted_digit}`")
        else:
            st.warning("Please draw a digit first")
    
    st.markdown("</div>", unsafe_allow_html=True)
