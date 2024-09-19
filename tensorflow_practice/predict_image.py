import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

def preprocess_image(img_path):
    # 이미지 열기
    img = Image.open(img_path)

    # 28x28 크기로 리사이즈
    img = img.resize((28, 28))

    # 흑백 변환
    img = img.convert('L')

    # NumPy 배열로 변환
    img_array = np.array(img)

    # 차원 확장 (1, 28, 28, 1) 및 정규화
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array

def predict_image(model_path, img_path):
    # 저장된 모델 로드
    model = tf.keras.models.load_model(model_path)

    # 이미지 전처리
    img_array = preprocess_image(img_path)

    # 예측
    predictions = model.predict(img_array)

    # 예측 결과 출력 (가장 높은 확률의 클래스 반환)
    predicted_class = np.argmax(predictions[0])
    print(f"이미지에 대한 예측: {predicted_class}")

# 실행 명령문 추가
if __name__ == '__main__':
    model_path = 'my_model.keras'  # 저장된 모델 파일 경로
    img_path = './image/test_predict_five.jpg'   # 예측할 이미지 파일 경로
    predict_image(model_path, img_path)
  