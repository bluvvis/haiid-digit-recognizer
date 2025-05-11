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
    .main {background: url('https://www.transparenttextures.com/patterns/cubes.png'), linear-gradient(to right, #a1c4fd, #c2e9fb);}
    .stButton>button:hover {background-color: #45a049; transform: scale(1.05); transition: 0.2s;}
    h1 {font-family: 'Roboto', sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);}
    .stSpinner {border: 3px solid #4CAF50; border-radius: 50%; animation: spin 1s linear infinite;}
    @keyframes spin {0% {transform: rotate(0deg);} 100% {transform: rotate(360deg);}}
    </style>
""", unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    st.header("О проекте")
    st.write("Это приложение распознаёт рукописные цифры и объясняет, как модель принимает решение. Загрузи изображение и попробуй!")
    st.image("https://via.placeholder.com/150", caption="Пример цифры")

# Основной контент
st.title("🎨 Распознавание рукописных цифр")
st.write("Загрузи изображение, получи предсказание и узнай, как модель думает!")

# Загрузка изображения
uploaded_file = st.file_uploader("Выбери изображение с цифрой", type=["png", "jpg", "jpeg"], help="Поддерживаются PNG, JPG, JPEG")

if uploaded_file is not None:
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
        fig = px.imshow(explanation.reshape(8, 8), color_continuous_scale="Viridis", title="Тепловая карта важности пикселей")
        fig.update_layout(width=400, height=400)
        st.plotly_chart(fig)

        # Обратная связь
        with st.expander("💬 Дай обратную связь"):
            correct_digit = st.selectbox("Если предсказание неверно, выбери правильную цифру:", list(range(10)))
            if st.button("Отправить"):
                log_feedback(uploaded_file.name, prediction, correct_digit)
                st.success("Спасибо! Твой отзыв сохранён.")
                st.balloons()
    except Exception as e:
        st.error(f"Ошибка: {e}")

# Подсказка для новых пользователей
st.info("👉 Загрузи изображение из папки `samples/` или своё собственное, чтобы начать!")
