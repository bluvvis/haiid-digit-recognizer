import numpy as np
from PIL import Image, ImageOps

def load_and_preprocess_image(file):
    """Загружает и обрабатывает изображение в формат 8x8."""
    try:
        img = Image.open(file).convert('L')
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        img_array = np.array(img) / 16.0
        return img, img_array.flatten()
    except Exception as e:
        raise Exception(f"Ошибка обработки изображения: {e}")
