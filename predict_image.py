# predict.py
import tensorflow as tf
from load_image import load_image, determine_channels
import numpy as np

def predict_image(model, img_path):
    image = load_image(img_path)
    channels = determine_channels(image)
    prediction = model.predict(np.expand_dims(image, axis=0))
    return prediction

def main():
    model = tf.keras.models.load_model('animal_model.h5')
    img_path = './data_set/TS_07.족제비/A07_H20_G070_C_211204_2085_20S_000008.010.jpg'  # 예측할 이미지 경로
    prediction = predict_image(model, img_path)
    print(f"예측 결과: {prediction}")

if __name__ == "__main__":
    main()
