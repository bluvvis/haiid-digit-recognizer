import streamlit as st
from recognition.model import load_and_train_model, predict_digit
from recognition.image_utils import load_and_preprocess_image
from recognition.explainability import explain_prediction
from recognition.feedback import log_feedback
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Digit Recognizer", layout="wide", initial_sidebar_state="expanded")

# Кастомный стиль: нежно-голубой фон и розовые акценты
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, .stApp {
        background-color: #e0f7fa !important;
        font-family: 'Roboto', sans-serif;
        color: #333;
    }

    .stButton>button {
        background: linear-gradient(90deg, #f472b6, #f9a8d4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        background: linear-gradient(90deg, #ec4899, #f472b6);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    .stFileUploader {
        background: rgba(255,255,255,0.8);
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(5px);
    }

    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 20px;
    }

    .main-title {
        background: rgba(255,255,255,0.9);
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }

    .result-card {
        background: rgba(255,255,255,0.9);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    st.header("📖 О приложении")
    st.write("Распознавайте рукописные цифры с помощью ИИ! Загружайте изображения и смотрите результат.")
    st.image("bg.gif", use_container_width=True)

# Заголовок
st.markdown("""
<div class='main-title'>
    <h1>✨ Распознавание рукописных цифр</h1>
    <p>Загрузите изображение (PNG, JPG) с цифрой, и модель предскажет результат!</p>
</div>
""", unsafe_allow_html=True)

# Загрузка изображения
uploaded_file = st.file_uploader(
    "📤 Выберите изображение для распознавания:",
    type=["png", "jpg", "jpeg"],
    help="Поддерживаются PNG, JPG, JPEG."
)

if uploaded_file is not None and st.button("🔄 Очистить изображение"):
    uploaded_file = None
    st.experimental_rerun()

if uploaded_file is not None:
    with st.spinner("🔍 Обработка изображения..."):
        try:
            col1, col2 = st.columns([1, 1])

            with col1:
                img, img_array = load_and_preprocess_image(uploaded_file)
                st.image(img, caption="🖼️ Загруженное изображение", use_container_width=True)

            with col2:
                model = load_and_train_model()
                prediction, confidence = predict_digit(model, img_array.reshape(1, -1))
                with st.container():
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.subheader(f"🔢 Предсказание: {prediction}")
                    st.markdown(f"**Уверенность модели**: {confidence:.2%}")
                    st.progress(confidence)
                    st.markdown("</div>", unsafe_allow_html=True)

            # Объяснение предсказания
            st.subheader("🧠 Как модель сделала выбор?")
            explanation = explain_prediction(model, img_array)
            fig = px.imshow(
                explanation.reshape(8, 8),
                color_continuous_scale="Viridis",
                title="Тепловая карта важности пикселей"
            )
            fig.update_layout(width=400, height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig)

            # Обратная связь
            with st.expander("💬 Оставить отзыв"):
                correct_digit = st.selectbox("Если предсказание неверно, выберите правильную цифру:", list(range(10)))
                if st.button("✅ Отправить отзыв"):
                    with st.spinner("Сохраняем ваш отзыв..."):
                        log_feedback(uploaded_file.name, prediction, correct_digit)
                        st.success("Спасибо за отзыв! 🎉")
                        st.balloons()

        except Exception as e:
            st.error(f"❌ Ошибка: {e}. Проверьте формат изображения.")
else:
    st.info("👉 Загрузите изображение, чтобы начать распознавание.")
