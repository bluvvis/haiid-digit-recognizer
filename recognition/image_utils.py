import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def select_and_process_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        img_array = np.array(img) / 16.0

        plt.imshow(img_array, cmap='gray')
        plt.title("Обработанное изображение")
        plt.show()

        return img_array.reshape(1, -1)
    except Exception as e:
        print(f"Ошибка: {e}")
        return None
