import tensorflow as tf
import numpy as np
import os

# 이미지 전처리 함수
def preprocess_image(img_path):
    # 이미지 로드 및 전처리
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')  # 흑백 이미지를 28x28로 로드
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # 이미지를 NumPy 배열로 변환 (28, 28, 1)

    # 차원 확장 및 정규화
    img_array = np.expand_dims(img_array, axis=0)  # (28, 28, 1) -> (1, 28, 28, 1)
    img_array = img_array / 255.0  # 0-255 사이 값들을 0-1 사이로 정규화

    return img_array

# 예측 함수
def predict_image(model_path, img_path):
    # 저장된 모델 로드
    print("모델을 로드 중...")
    model = tf.keras.models.load_model(model_path)
    print("모델 로드 완료.")

    # 이미지 전처리
    img_array = preprocess_image(img_path)
    print(f"이미지 전처리 완료: {img_array.shape}")

    # 예측
    predictions = model.predict(img_array)
    print(f"모델 예측 결과: {predictions}")

    # 예측 결과 출력 (가장 높은 확률의 클래스 반환)
    predicted_class = np.argmax(predictions[0])
    print(f"이미지에 대한 예측: {predicted_class}")
    
    # 각 클래스에 대한 확률 출력
    print(f"각 클래스에 대한 확률 분포: {predictions[0]}")

# 실행 명령문 추가
if __name__ == '__main__':
    model_path = 'my_model.keras'  # 저장된 모델 파일 경로
    img_path = os.path.join(os.path.dirname(__file__), "image", "test_predict_five.jpg")  # 예측할 이미지 파일 경로
    predict_image(model_path, img_path)
