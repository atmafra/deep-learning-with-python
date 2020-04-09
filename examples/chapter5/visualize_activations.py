import os
import matplotlib.pyplot as plt
import numpy as np
from keras import Model, models
from keras.models import load_model
from keras_preprocessing import image

from core.network import create_model_from_file


def get_model(model_path: str = 'models/cats_and_dogs',
              model_filename: str = 'cats-and-dogs-convolutional-with-dropout.json',
              weights_filename: str = 'cats-and-dogs-convolutional-with-dropout.h5'):
    model_filepath = os.path.join(model_path, model_filename)
    model = create_model_from_file(model_filepath)
    weights_filepath = os.path.join(model_path, weights_filename)
    model.load_weights(weights_filepath)
    return model


def get_image_tensor(image_path: str = 'data/cats_and_dogs_small/test/cats',
                     image_filename: str = 'cat.1700.jpg'):
    image_filepath = os.path.join(image_path, image_filename)
    img = image.load_img(image_filepath, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    print(img_tensor.shape)
    return img_tensor


def plot_image_tensor(img_tensor):
    plt.imshow(img_tensor[0])
    plt.show()


def get_activation_model(model: Model):
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    return activation_model


def get_activations(activations, layer_number, img_tensor):
    return activations[layer_number]


def plot_channel(activation, channel):
    plt.matshow(activation[0, :, :, channel], cmap='viridis')
    plt.show()


def plot_all_layers_all_channels(model: Model,
                                 activations,
                                 images_per_row: int = 16):
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, size * images_per_row))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                std = channel_image.std()
                if std > 0.:
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


def run():
    model = get_model()
    model.summary()
    img_tensor = get_image_tensor()
    plot_image_tensor(img_tensor)
    activation_model = get_activation_model(model)
    activations = activation_model.predict(img_tensor)
    first_layer_activation = activations[0]
    # plot_channel(activation=first_layer_activation, channel=4)
    # plot_channel(activation=first_layer_activation, channel=7)
    plot_all_layers_all_channels(model, activations)
