import os
import numpy as np
import tensorflow as tf

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증가 방지 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # 2GB 메모리 제한 설정
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
        )
        
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def create_model(input_shape):
    print("모델 생성 중...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류를 위한 출력
    ])
    print("모델 생성 완료.")
    return model

def compile_model(model):
    print("모델 컴파일 중...")
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("모델 컴파일 완료.")

def load_data(data_dir, image_size=(75, 75)):
    print("데이터 로드 중...")
    images = []
    labels = []

    # 'weasel' 이미지를 로드
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(data_dir, filename)
                image = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
                image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # 정규화
                images.append(image)
                labels.append(1)  # 'weasel' 클래스를 1로 설정
    except Exception as e:
        print("데이터 로드 중 오류 발생:", e)

    print(f"로드된 이미지 수: {len(images)}")
    return np.array(images), np.array(labels)

def save_model(model, model_path):
    print("모델 저장 중...")
    # 모델 저장
    try:
        model.save(model_path)
        print(f"모델이 {model_path}에 저장되었습니다.")
    except Exception as e:
        print("모델 저장 중 오류 발생:", e)

if __name__ == "__main__":
    data_dir = '../data_set/output'  # 데이터 디렉토리 설정
    model_path = 'weasel_model.keras'  # 모델 저장 경로

    # 데이터 로드
    train_images, train_labels = load_data(data_dir)

    # 모델이 로드된 이미지가 있는지 확인
    if len(train_images) == 0:
        print("로드된 이미지가 없습니다. 프로그램을 종료합니다.")
    else:
        # 모델 생성 및 컴파일
        model = create_model(train_images.shape[1:])
        compile_model(model)

        # 모델 학습
        try:
            print("모델 학습 중...")
            model.fit(train_images, train_labels, epochs=10, batch_size=8)
            print("모델 학습 완료.")
        except Exception as e:
            print("모델 학습 중 오류 발생:", e)

        # 모델 저장
        save_model(model, model_path)
