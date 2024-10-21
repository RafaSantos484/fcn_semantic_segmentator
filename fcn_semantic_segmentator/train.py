import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras import layers, models

from params import epochs, img_size, color_mode
from .utils import json_to_mask


def load_imgs_and_masks_arrs():
    imgs_arr = []
    masks_arr = []
    for img_path in os.listdir('tmp/processed_imgs'):
        img = Image.open(f'tmp/processed_imgs/{img_path}')
        img_arr = np.array(img)
        imgs_arr.append(img_arr)

        filename = img_path.split('.')[0]
        mask = json_to_mask(f'tmp/labels/{filename}.json')
        masks_arr.append(mask)

    return np.array(imgs_arr), np.array(masks_arr)


def get_fcn_model():
    input_shape = (img_size, img_size, 1 if color_mode == 'L' else 3)
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def run():
    print('Loading images and masks...')
    imgs_arr, masks_arr = load_imgs_and_masks_arrs()
    imgs_arr = imgs_arr / 255.0
    masks_arr = np.expand_dims(masks_arr, axis=-1)
    print(imgs_arr.shape)
    print(masks_arr.shape)

    print('Splitting data...')
    train_imgs_arr, test_imgs_arr, train_labels, test_labels = train_test_split(
        imgs_arr, masks_arr, test_size=0.2)

    model = get_fcn_model()
    model.summary()
    print('Compiling model...')
    opt = keras.optimizers.Adam(1e-4)
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    print('Training model...')
    model.fit(train_imgs_arr, train_labels, epochs=epochs, batch_size=8)
    _, test_acc = model.evaluate(test_imgs_arr, test_labels)
    print('\nTest accuracy:', test_acc)

    print('Exporting model...')
    model.save('tmp/model')
