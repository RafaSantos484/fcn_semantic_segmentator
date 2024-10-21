import os
from PIL import Image
import numpy as np
from tensorflow import keras

from params import img_size, color_mode
from .utils import clear_folder


def run():
    model = keras.models.load_model('tmp/model')
    imgs_arrs = []
    imgs_names = []
    imgs_shapes = []
    for img_path in os.listdir('test_imgs'):
        img = Image.open(f'test_imgs/{img_path}')
        imgs_shapes.append(img.size)
        img = img.convert(color_mode)
        img = img.resize((img_size, img_size))
        img_arr = np.array(img) / 255.0
        # img_arr = np.expand_dims(img_arr, axis=0)
        imgs_arrs.append(img_arr)
        imgs_names.append(os.path.splitext(img_path)[0])

    clear_folder('tmp/test_results')
    imgs_arrs = np.array(imgs_arrs)
    preds = model.predict(imgs_arrs)
    for pred, img_name, img_shape in zip(preds, imgs_names, imgs_shapes):
        mask = np.where(pred > 0.5, 255, 0).astype(np.uint8)
        mask = np.squeeze(mask)
        img = Image.fromarray(mask)
        img = img.resize(img_shape)
        img.save(f'tmp/test_results/{img_name}.png')
        # img.show()
