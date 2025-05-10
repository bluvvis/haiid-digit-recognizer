from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_train_model():
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_digit(model, processed_input):
    """Предсказывает цифру и возвращает предсказание с уверенностью."""
    prediction = model.predict(processed_input)[0]
    confidence = model.predict_proba(processed_input).max()
    return prediction, confidence

def print_model_info(model):
    print("\n[Информация о модели]")
    print(f"Количество деревьев: {len(model.estimators_)}")
    print(f"Глубина первого дерева: {model.estimators_[0].get_depth()}")
