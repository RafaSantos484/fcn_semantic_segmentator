import json
import os
import shutil

import cv2
import numpy as np

from params import img_size


def clear_folder(folder: str):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def copy_files_content(src: str, dest: str):
    for filename in os.listdir(src):
        shutil.copy(os.path.join(src, filename), dest)


def copy_folders_content(src: str, dest: str):
    for foldername in os.listdir(src):
        shutil.copytree(os.path.join(src, foldername),
                        os.path.join(dest, foldername))


def json_to_mask(path: str):
    with open(path, 'r') as f:
        data = json.load(f)

    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    for shape in data['shapes']:
        points = shape['points']
        points = np.array(points, dtype=np.int32)
        # points = points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 1)

    '''
    print(mask)
    img = ImageOps.autocontrast(Image.fromarray(mask))
    img.show()
    exit()
    '''
    return np.array(mask)
