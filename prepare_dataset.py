import os
import numpy as np
from load_json import load_json
from load_image import load_image

def prepare_dataset(json_path, img_dir):
    data = load_json(json_path)
    img_files = [entry['file_name'] for entry in data['images']]
    image_types = [entry['type'] for entry in data['images']]
    
    images = []
    channels = None

    for img_file, img_type in zip(img_files, image_types):
        img_path = os.path.join(img_dir, img_file)
        color_mode = 'rgb' if img_type == 'RGB' else 'grayscale'
        image = load_image(img_path, color_mode=color_mode)
        
        if channels is None:
            channels = image.shape[-1]
        elif image.shape[-1] != channels:
            raise ValueError("Inconsistent channel size among images.")
        
        images.append(image)

    return np.array(images), img_files, channels
