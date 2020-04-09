import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import backend as K


def get_model():
    model = VGG16(weights='imagenet', include_top=False)
    return model


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(model, layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128

    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


def plot_filter_grid(model, layer_name: str):
    filter_images_per_row = 8
    size = 64
    margin = 2
    side = filter_images_per_row * size + (filter_images_per_row - 1) * margin
    results = np.zeros((side, side, 3), dtype=int)

    for i in range(filter_images_per_row):
        for j in range(filter_images_per_row):
            filter_index = j + (i * filter_images_per_row)
            print('filter index = {}'.format(filter_index))
            filter_img = generate_pattern(model, layer_name, filter_index, size=size)
            horizontal_start = j * (size + margin)
            horizontal_end = horizontal_start + size
            vertical_start = i * (size + margin)
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    plt.figure(figsize=(10, 10))
    plt.imshow(results)


def run():
    model = get_model()
    layer_name = 'block5_conv2'
    # dep_img = generate_pattern(model=model, layer_name=layer_name, filter_index=0, size=150)
    # plt.imshow(dep_img)
    plot_filter_grid(model, layer_name)
    plt.show()
