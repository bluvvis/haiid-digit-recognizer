import numpy as np

def explain_prediction(model, processed_input):
    """Возвращает важность пикселей для предсказания."""
    importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    return importance
