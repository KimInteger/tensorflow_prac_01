import tensorflow as tf
from predict_image import predict_image, display_image

def main():
    model_path = './saved_model/my_model'  # 저장된 모델 경로
    img_path = './data_set/TS_07.족제비/A07_H20_G070_C_211204_2085_20S_000008.010.jpg'  # 예측할 이미지 경로

    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 이미지 예측
    prediction = predict_image(model, img_path)
    print(f'예측 결과: {prediction}')

    # 이미지 출력
    display_image(img_path)

if __name__ == "__main__":
    main()
