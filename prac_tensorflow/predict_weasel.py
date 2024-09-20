# predict.py

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(image_path, target_size=(150, 150)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return image

def predict_image(model, image_path, threshold=0.5):
    image = load_and_preprocess_image(image_path)
    prediction = model.predict(image)
    
    # 임계값 기준으로 클래스 판단
    if prediction[0][0] >= threshold:
        return "weasel"
    else:
        return "not weasel"

if __name__ == "__main__":
    model_path = './saved_model/my_model.keras'  # 학습된 모델 경로
    image_path = './path_to_your_image.jpg'  # 예측할 이미지 경로

    # 모델 로드
    model = tf.keras.models.load_model(model_path)

    # 이미지 예측
    result = predict_image(model, image_path)
    print(f"이미지 '{image_path}'의 예측 결과: {result}")
