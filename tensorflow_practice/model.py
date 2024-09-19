import tensorflow as tf

# Sequential 모델을 정의하는 함수
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # 이미지 입력: 28x28 크기
        tf.keras.layers.Dense(128, activation='relu'),  # 은닉층
        tf.keras.layers.Dropout(0.2),                   # 드롭아웃으로 과적합 방지
        tf.keras.layers.Dense(10, activation='softmax') # 출력층 (숫자 0~9를 분류)
    ])
    return model
