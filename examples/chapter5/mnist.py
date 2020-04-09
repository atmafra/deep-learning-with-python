import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from core.corpus import Corpus
from examples.chapter5.mnist_configurations import load_experiment_plan


def to_array(image_set: np.array) -> np.array:
    image_set_size = len(image_set)
    image_shape = image_set[0].shape
    array_dimensions = image_shape[0] * image_shape[1]
    return image_set.reshape(image_set_size, array_dimensions)


def normalize(image_set: np.array) -> np.array:
    return image_set.astype('float32') / 255


def load_corpus(input_type: str, verbose: bool = True) -> Corpus:
    """Loads the MNIST corpus from public repositories
    """
    if verbose:
        print("Loading MNIST dataset...")

    corpus = Corpus.from_tuple(mnist.load_data())
    train_set_size = corpus.training_set.length
    test_set_size = corpus.test_set.length

    image_shape = corpus.training_set.input_data.shape
    height = image_shape[1]
    width = image_shape[2]
    channels = 1

    train_images = []
    test_images = []

    # Prepare the images
    if input_type == 'array':
        train_images = normalize(to_array(corpus.training_set.input_data))
        test_images = normalize(to_array(corpus.test_set.input_data))

    elif input_type == 'image':
        train_images = normalize(corpus.training_set.input_data.reshape(train_set_size, width, height, channels))
        test_images = normalize(corpus.test_set.input_data.reshape(test_set_size, width, height, channels))

    # Convert the labels to categorical
    train_labels = to_categorical(corpus.training_set.output_data)
    test_labels = to_categorical(corpus.test_set.output_data)

    if verbose:
        print("image dimensions :", image_shape)
        print("train set size   :", train_set_size, "images")
        print("test set size    :", test_set_size, "images")

    corpus = Corpus.from_datasets(train_images, train_labels, test_images, test_labels)
    return corpus


def run():
    """Runs the MNIST digit recognition example
    """
    corpus = load_corpus(input_type='image')
    experiment_plan = load_experiment_plan(corpus=corpus)
    experiment_plan.run(print_results=True,
                        plot_training_loss=True,
                        plot_training_accuracy=True,
                        display_progress_bars=True,
                        save_models=True,
                        models_path='models/mnist')


