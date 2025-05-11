import streamlit as st
from recognition.model import load_and_train_model, predict_digit
from recognition.image_utils import load_and_preprocess_image
from recognition.explainability import explain_prediction
from recognition.feedback import log_feedback
import plotly.express as px
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Digit Recognizer", layout="wide", initial_sidebar_state="expanded")

# Кастомные стили с бледно-розовым градиентом
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body {
        background: linear-gradient(135deg, #ffe4e6, #f3e8ff) !important;
        font-family: 'Roboto', sans-serif;
        color: #333333;
        overflow: hidden;
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
        background: rgba(255,255,255,0.7);
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        background: rgba(255,255,255,0.9);
        transform: scale(1.02);
    }
    .stSelectbox {
        background: rgba(255,255,255,0.7);
        border-radius: 8px;
        color: #333333;
    }
    h1, h2, h3 {
        color: #333333;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-in;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f9a8d4, #fed7e2);
        color: #333333;
        border-radius: 10px;
        padding: 20px;
        animation: slideIn 0.5s ease-out;
    }
    .stProgress .st-bo {
        background: linear-gradient(90deg, #f472b6, #f9a8d4);
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    .animated-icon {
        animation: bounce 1.5s infinite;
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    </style>
""", unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    st.header("📖 О приложении")
    st.write("Распознавайте рукописные цифры с помощью ИИ! Загружайте изображения (PNG, JPG) и смотрите результат.")
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://media.giphy.com/media/3o7TKrHrTLiH0zE0HC/giphy.gif' alt='Animated Digit' width='150' class='animated-icon'>
            <p style='color: #333333;'>Анимированная цифра</p>
        </div>
    """, unsafe_allow_html=True)

# Основной контент
st.title("✨ Распознавание рукописных цифр")
st.write("Загрузите изображение (PNG, JPG) с цифрой, и модель предскажет результат!")

# Загрузка изображения
uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["png", "jpg", "jpeg"],
    help="Поддерживаются PNG, JPG, JPEG."
)

# Кнопка очистки
if uploaded_file is not None:
    if st.button("Очистить"):
        uploaded_file = None
        st.experimental_rerun()

if uploaded_file is not None:
    with st.spinner("Обработка..."):
        try:
            col1, col2 = st.columns([1, 1])

            with col1:
                # Показываем загруженное изображение
                img, img_array = load_and_preprocess_image(uploaded_file)
                st.image(img, caption="Загруженное изображение", use_container_width=True)

            with col2:
                # Предсказание
                model = load_and_train_model()
                prediction, confidence = predict_digit(model, img_array.reshape(1, -1))
                st.subheader(f"Предсказание: **{prediction}**")
                st.markdown(f"**Уверенность**: {confidence:.2%}")
                
                # Индикатор уверенности
                st.progress(confidence)

            # Объяснение
            st.subheader("🔍 Как модель сделала выбор?")
            explanation = explain_prediction(model, img_array)
            fig = px.imshow(
                explanation.reshape(8, 8),
                color_continuous_scale="Viridis",
                title="Тепловая карта важности"
            )
            fig.update_layout(width=400, height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig)

            # Обратная связь
            with st.expander("💬 Оставить отзыв"):
                correct_digit = st.selectbox("Если предсказание неверно, выберите цифру:", list(range(10)))
                if st.button("Отправить"):
                    with st.spinner("Сохранение..."):
                        log_feedback(uploaded_file.name, prediction, correct_digit)
                        st.success("Спасибо за отзыв! 🎉")
                        st.balloons()

        except Exception as e:
            st.error(f"Ошибка: {e}. Проверьте формат изображения.")

# Подсказка
st.info("👉 Попробуйте загрузить изображение или создайте своё!")
