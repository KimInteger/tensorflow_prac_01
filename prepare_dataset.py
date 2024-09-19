import os
import numpy as np
from load_json import load_json
from load_image import load_image

def prepare_dataset(json_path, img_dir):
    data = load_json(json_path)
    img_files = [entry['file_name'] for entry in data['images']]
    images = []

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        image = load_image(img_path)
        images.append(image)

    return np.vstack(images), img_files
