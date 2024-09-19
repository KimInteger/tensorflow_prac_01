import tensorflow as tf
import numpy as np

# 모델 로드
model = tf.keras.models.load_model('model.keras')

# 이미지 로드 및 전처리
img_path = './image/mococo.png'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))  # 모델 입력 크기에 맞게 조정
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
img_array /= 255.0  # 정규화

# 예측
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

print(f'Predicted class: {predicted_class}')
