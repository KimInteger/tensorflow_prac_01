# train_model.py
import numpy as np
from load_image import load_image, determine_channels
from create_and_train_model import create_and_train_model

def load_data(image_paths, labels):
    X_train = []
    y_train = []
    for img_path, label in zip(image_paths, labels):
        image = load_image(img_path)
        channels = determine_channels(image)
        X_train.append(image)
        y_train.append(label)
    return np.array(X_train), np.array(y_train), channels

def main():
    image_paths = ['./data_set/TS_07.족제비/A07_G01_G001_G_171229_2001_29S_000013.967.jpg']  # 예제 이미지 경로
    labels = [1]  # 예제 레이블 (0 또는 1)
    
    X_train, y_train, input_channels = load_data(image_paths, labels)
    model = create_and_train_model(X_train, y_train, input_channels)
    model.save('animal_model.h5')

if __name__ == "__main__":
    main()
