import streamlit as st
from recognition.model import load_and_train_model, predict_digit
from recognition.image_utils import load_and_preprocess_image
from recognition.explainability import explain_prediction
from recognition.feedback import log_feedback
import plotly.express as px
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Digit Recognizer", layout="wide", initial_sidebar_state="expanded")

# Кастомные стили
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    .main {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #059669;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stFileUploader {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSelectbox {
        background-color: #ffffff;
        border-radius: 8px;
        font-family: 'Poppins', sans-serif;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    .sidebar .sidebar-content {
        background: #1e293b;
        color: #ffffff;
        border-radius: 8px;
        padding: 10px;
    }
    .stProgress .st-bo {
        background-color: #10b981;
    }
    .warning-box {
        background-color: #fef3c7;
        color: #b45309;
        padding: 10px;
        border-radius: 8px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    st.header("📖 О приложении")
    st.write("Распознавайте рукописные цифры с помощью ИИ! Загружайте квадратные изображения (PNG, JPG) для лучших результатов.")
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://media.giphy.com/media/3o7TKrHrTLiH0zE0HC/giphy.gif' alt='Animated Digit' width='150'>
            <p style='color: #ffffff;'>Анимированная цифра</p>
        </div>
    """, unsafe_allow_html=True)

# Основной контент
st.title("✨ Распознавание рукописных цифр")
st.markdown("Загрузите **квадратное** изображение (PNG, JPG) с цифрой, и модель предскажет результат!")

# Инструкция по формату
st.markdown("""
    <div class='warning-box'>
        ⚠️ Для точного распознавания используйте квадратное изображение (например, 28x28 пикселей, как в MNIST). Неквадратные изображения могут снизить точность.
    </div>
""", unsafe_allow_html=True)

# Загрузка изображения
uploaded_file = st.file_uploader(
    "Выберите изображение с цифрой",
    type=["png", "jpg", "jpeg"],
    help="Загрузите квадратное изображение (PNG, JPG, JPEG) для лучшей точности."
)

# Кнопка очистки
if uploaded_file is not None:
    if st.button("Очистить изображение"):
        uploaded_file = None
        st.experimental_rerun()

if uploaded_file is not None:
    with st.spinner("Обработка изображения..."):
        try:
            col1, col2 = st.columns([1, 1])

            with col1:
                # Показываем загруженное изображение
                img, img_array = load_and_preprocess_image(uploaded_file)
                st.image(img, caption="Загруженное изображение", use_column_width=True)

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
                title="Тепловая карта важности пикселей"
            )
            fig.update_layout(width=400, height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig)

            # Обратная связь
            with st.expander("💬 Оставить отзыв"):
                correct_digit = st.selectbox("Если предсказание неверно, выберите правильную цифру:", list(range(10)))
                if st.button("Отправить отзыв"):
                    with st.spinner("Сохранение..."):
                        log_feedback(uploaded_file.name, prediction, correct_digit)
                        st.success("Спасибо за отзыв! 🎉")
                        st.balloons()

        except Exception as e:
            st.error(f"Ошибка обработки: {e}. Убедитесь, что изображение квадратное и в правильном формате.")

# Подсказка для новых пользователей
st.info("👉 Попробуйте загрузить изображение из папки `samples/` или создайте своё квадратное изображение с цифрой!")
