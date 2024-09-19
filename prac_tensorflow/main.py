import os
import numpy as np
import cv2  # 추가된 임포트
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from json_parser import load_json, get_image_info, get_annotations
from image_processor import load_image, crop_bbox, save_cropped_image, resize_image
from train_model import create_model, compile_model, train_model

def process_files(json_dir, image_dir, output_dir, image_size=(256, 256)):
    all_train_images = []
    all_train_labels = []

    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            print(f"Processing JSON file: {json_path}")
            data = load_json(json_path)

            images_info = get_image_info(data)
            annotations = get_annotations(data)

            for image_info in images_info:
                image_path = os.path.join(image_dir, image_info['file_name'])
                print(f"Loading image: {image_path}")

                image = load_image(image_path)
                if image is None:
                    print(f"Error: Image at {image_path} could not be loaded.")
                    continue

                # 이미지 크기 조정
                original_size = (image.shape[1], image.shape[0])
                resized_image = resize_image(image, size=image_size)

                for annotation in annotations:
                    bbox = annotation['bbox']
                    print(f"Annotation bbox: {bbox}")

                    # 바운딩 박스 비율 조정
                    x_min, y_min = bbox[0]
                    x_max, y_max = bbox[1]

                    # 비율 조정
                    x_min_resized = int(x_min / original_size[0] * image_size[0])
                    y_min_resized = int(y_min / original_size[1] * image_size[1])
                    x_max_resized = int(x_max / original_size[0] * image_size[0])
                    y_max_resized = int(y_max / original_size[1] * image_size[1])
                    
                    resized_bbox = [[x_min_resized, y_min_resized], [x_max_resized, y_max_resized]]

                    cropped_image = crop_bbox(resized_image, resized_bbox)
                    
                    if cropped_image.size == 0:
                        print(f"Error: Cropped image is empty for {image_path}")
                        continue

                    output_image_path = os.path.join(output_dir, f"cropped_{image_info['file_name']}")
                    save_cropped_image(cropped_image, output_image_path)

                    all_train_images.append(cropped_image)
                    all_train_labels.append(annotation['category_name'])

    # Convert lists to numpy arrays with consistent shape
    all_train_images = np.array([cv2.resize(img, image_size) for img in all_train_images])
    all_train_labels = np.array(all_train_labels)

    return all_train_images, all_train_labels

def main(json_dir, image_dir, output_dir):
    train_images, train_labels = process_files(json_dir, image_dir, output_dir)

    if len(train_images) == 0:
        print("No images were loaded. Check file paths and formats.")
        return

    # 레이블을 정수형으로 변환
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)

    # 레이블을 원-핫 인코딩으로 변환
    label_binarizer = LabelBinarizer()
    train_labels = label_binarizer.fit_transform(train_labels)

    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")

    input_shape = (train_images.shape[1], train_images.shape[2], train_images.shape[3])
    num_classes = len(label_binarizer.classes_)
    model = create_model(input_shape)
    compile_model(model)
    train_model(model, train_images, train_labels)

    # 모델 저장
    model.save('model.keras')
    print("Model saved to model.keras")

if __name__ == '__main__':
    json_dir = '../data_set/TL_07_weasel/'
    image_dir = '../data_set/TS_07_weasel/'
    output_dir = '../data_set/output/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(json_dir, image_dir, output_dir)
