import tensorflow as tf
from model import create_model  # 모델 파일에서 함수 임포트

def train_model():
    # MNIST 데이터셋 로드
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 데이터 정규화
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 모델 생성
    model = create_model()

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    model.fit(x_train, y_train, epochs=5)

    # 모델 저장
    model.save('my_model.keras')

    # 모델 평가
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"테스트 정확도: {test_acc}")

if __name__ == '__main__':
    train_model()