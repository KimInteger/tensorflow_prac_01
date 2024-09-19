# json_parser.py

import json

def load_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except UnicodeDecodeError as e:
        print(f"Error decoding JSON file {json_path}: {e}")
        raise
    except FileNotFoundError as e:
        print(f"File not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON data from file {json_path}: {e}")
        raise

def get_image_info(data):
    return data['images']

def get_annotations(data):
    return data['annotations']
