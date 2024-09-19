import numpy as np
import tensorflow as tf

def load_image(img_path):
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(1920, 1080))  # 이미지 크기를 1920x1080으로 조정합니다
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # 이미지를 정규화합니다
    return image
