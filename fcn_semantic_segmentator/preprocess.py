import os
from PIL import Image
from params import img_size, color_mode
from .utils import clear_folder


def preprocess_imgs(src: str, dest: str):
    for filename in os.listdir(src):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(src, filename)
            with Image.open(img_path) as img:
                img = img.convert(color_mode)
                img = img.resize((img_size, img_size))

                output_path = os.path.join(
                    dest, f'{os.path.splitext(filename)[0]}.jpg')

                img.save(output_path, 'JPEG')


def run():
    clear_folder('tmp')

    os.mkdir('tmp/labels')
    os.mkdir('tmp/processed_imgs')
    os.mkdir('tmp/test_results')
    '''
    for folder in os.listdir('imgs'):
        os.mkdir(f'tmp/processed_imgs/{folder}')
        preprocess_imgs(f'imgs/{folder}', f'tmp/processed_imgs/{folder}')
    '''
    preprocess_imgs('imgs', 'tmp/processed_imgs')
