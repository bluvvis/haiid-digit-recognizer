import numpy as np
import matplotlib.pyplot as plt

def explain_prediction(model, processed_input):
    importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    img_importance = importance.reshape(8, 8)

    plt.imshow(img_importance, cmap='hot')
    plt.title("Важность пикселей для модели")
    plt.colorbar()
    plt.show()

