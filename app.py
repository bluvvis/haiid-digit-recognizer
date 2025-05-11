import streamlit as st
from recognition.model import load_and_train_model, predict_digit
from recognition.image_utils import load_and_preprocess_image
from recognition.explainability import explain_prediction
from recognition.feedback import log_feedback
import plotly.express as px
import numpy as np

# Настройка темы и стилей
st.set_page_config(page_title="Digit Recognizer", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    .main {
        background: url('https://www.transparenttextures.com/patterns/dark-mosaic.png'), linear-gradient(to right, #6b7280, #9ca3af);
        background-size: cover, auto;
        animation: subtle-move 10s infinite alternate;
    }
    .stButton>button {
        background-color: #f97316; 
        color: white; 
        border-radius: 10px; 
        padding: 10px;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button:hover {
        background-color: #ea580c; 
        transform: scale(1.05); 
        transition: 0.2s;
    }
    .stSelectbox {
        background-color: #ffffff; 
        border-radius: 5px;
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3 {
        color: #fefcbf; 
        font-family: 'Roboto', sans-serif; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    .sidebar .sidebar-content {
        background: #1f2937; 
        color: #ffffff;
        border-right: 2px solid #4b5563;
    }
    .stSpinner .spinner {
        border: 3px solid #f97316; 
        border-top: 3px solid #ffffff; 
        border-radius: 50%; 
        width: 30px; 
        height: 30px; 
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes subtle-move {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    </style>
""", unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    st.header("О проекте")
    st.write("Распознаём рукописные цифры и показываем, как модель думает!")
    # GIF в боковой панели
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://media.giphy.com/media/3o7TKrHrTLiH0zE0HC/giphy.gif' alt='Animated Digit' width='150'>
            <p>Анимированная цифра</p>
        </div>
    """, unsafe_allow_html=True)

# Основной контент
st.title("🎨 Распознавание рукописных цифр")
st.write("Загрузи изображение и узнай, что видит модель!")

# Загрузка изображения
uploaded_file = st.file_uploader("Выбери изображение с цифрой", type=["png", "jpg", "jpeg"], help="Поддерживаются PNG, JPG, JPEG")

if uploaded_file is not None:
    with st.spinner("Обработка изображения..."):
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                # Показываем загруженное изображение
                img, img_array = load_and_preprocess_image(uploaded_file)
                st.image(img, caption="Твоё изображение", use_column_width=True)
            
            with col2:
                # Предсказание
                model = load_and_train_model()
                prediction, confidence = predict_digit(model, img_array.reshape(1, -1))
                st.subheader(f"Предсказание: **{prediction}**")
                st.write(f"Уверенность: **{confidence:.2%}**")
                
                # Индикатор уверенности
                st.progress(confidence)
    
            # Объяснение
            st.subheader("🔍 Почему модель так решила?")
            explanation = explain_prediction(model, img_array)
            fig = px.imshow(explanation.reshape(8, 8), color_continuous_scale="Inferno", title="Тепловая карта важности пикселей")
            fig.update_layout(width=400, height=400)
            st.plotly_chart(fig)
    
            # Обратная связь
            with st.expander("💬 Дай обратную связь"):
                correct_digit = st.selectbox("Если предсказание неверно, выбери правильную цифру:", list(range(10)))
                if st.button("Отправить"):
                    with st.spinner("Сохранение отзыва..."):
                        log_feedback(uploaded_file.name, prediction, correct_digit)
                        st.success("Спасибо! Твой отзыв сохранён.")
                        st.balloons()
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Подсказка для новых пользователей
st.info("👉 Загрузи изображение из папки `samples/` или своё собственное, чтобы начать!")
