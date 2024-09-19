import tensorflow as tf
import numpy as np

def load_image(img_path, color_mode='rgb'):
    image = tf.keras.preprocessing.image.load_img(img_path, color_mode=color_mode, target_size=(1920, 1080))
    image = tf.keras.preprocessing.image.img_to_array(image)
    return image

def determine_channels(image):
    if image.shape[-1] == 1:
        return 1
    elif image.shape[-1] == 3:
        return 3
    else:
        raise ValueError("Unsupported image format")
