import cv2
from io import BytesIO

import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt

from keras import backend as K
from keras.applications import VGG16
from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

image_url = 'https://s3.amazonaws.com/book.keras.io/img/ch5/creative_commons_elephant.jpg'
# image_url = 'https://upload.wikimedia.orgo/wikipedia/commons/6/6a/Indian_Elephant.jpeg'
# image_url = 'https://ca-times.brightspotcdn.com/dims4/default/2f4e1bf/2147483647/strip/true/crop/2910x2021+0+0/resize/840x583!/quality/90/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2Fe4%2Fe8%2F0ee997f74398b400c5ea65244f70%2Fzoo-elephant-euthanized-67478.jpg'
# image_url = 'https://www.thefactsite.com/wp-content/uploads/2018/07/facts-about-parrots.jpg'


def download_image(url: str = image_url):
    response = requests.get(url=image_url)
    return Image.open(BytesIO(response.content))


def prepare_image(img):
    x = img.resize((224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def get_image(url: str = image_url):
    image = download_image()
    image = prepare_image(image)
    return image


def get_heatmap(model, image):
    african_e66elephant_output = model.output[:, 386]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(african_e66elephant_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([image])

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


def superimpose_heatmap_to_image(image, heatmap):
    cv2_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (cv2_image.shape[1], cv2_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + cv2_image
    cv2.imwrite('images/superimposed_image.jpg', superimposed_img)


def run():
    image = download_image()
    prepared_image = prepare_image(image)
    model = VGG16(weights='imagenet')
    preds = model.predict(prepared_image)
    print('Predicted:', decode_predictions(preds, top=3))
    heatmap = get_heatmap(model, prepared_image)
    plt.matshow(heatmap)
    superimpose_heatmap_to_image(image, heatmap)
