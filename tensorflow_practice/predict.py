import tensorflow as tf
import numpy as np

# 저장된 모델 로드
model = tf.keras.models.load_model('my_model.keras')

# MNIST 테스트 데이터셋 로드
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

# 데이터 정규화
x_test = x_test / 255.0

# 첫 번째 테스트 이미지를 사용해 예측
predictions = model.predict(x_test)
print(f"첫 번째 이미지에 대한 예측: {np.argmax(predictions[0])}")
