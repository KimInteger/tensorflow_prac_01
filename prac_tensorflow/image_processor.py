# image_processor.py

import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} could not be loaded.")
    return image

def crop_bbox(image, bbox):
    """
    Crop the image based on the bounding box.

    :param image: The input image as a NumPy array.
    :param bbox: Bounding box coordinates in the format [[x_min, y_min], [x_max, y_max]].
    :return: Cropped image as a NumPy array.
    """
    x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
    x_max, y_max = int(bbox[1][0]), int(bbox[1][1])

    # 이미지 크기 범위 내로 조정
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, image.shape[1]), min(y_max, image.shape[0])

    return image[y_min:y_max, x_min:x_max]

def save_cropped_image(cropped_image, output_path):
    if cropped_image is not None:
        cv2.imwrite(output_path, cropped_image)
    else:
        print(f"Error: Cropped image is None for path {output_path}")
