import numpy as np
from keras import Model, optimizers
from keras.applications import VGG16
from keras.utils import Progbar

from core.corpus import Corpus, CorpusType
from core.datasets import Dataset
from core.experiment import Experiment
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.training_configuration import TrainingConfiguration
from examples.chapter5.cats_and_dogs_files import *

classifier = [
    {'layer_type': 'Dense', 'units': 256, 'activation': 'relu', 'input_dim': 4 * 4 * 512},
    {'layer_type': 'Dropout', 'rate': 0.5},
    {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}]

learning_rate = 2e-5
training_parameters = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=learning_rate),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit': {
            'epochs': 30,
            'batch_size': 20}},
    'validation': {
        'strategy': ValidationStrategy.CROSS_VALIDATION}}


def load_convolutional_base():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    return conv_base


def load_experiment(corpus: Corpus):
    classifier_network = NeuralNetwork.from_configurations(name='Cats & Dogs Classifier',
                                                           layers_configuration=classifier)

    training_configuration = TrainingConfiguration(training_parameters)

    experiment = Experiment(name='Cats & Dogs - Feature Extraction',
                            neural_network=classifier_network,
                            training_configuration=training_configuration,
                            corpus_type=CorpusType.CORPUS_DATASET,
                            corpus=corpus)
    return experiment


def extract_features(set_files: DatasetFileIterator,
                     conv_base: Model):
    """Extract features from a set of files using a convolutional base model as feature extractor
    """
    sample_count = set_files.length
    batch_size = set_files.batch_size
    total_batches = sample_count // batch_size
    output_shape = (sample_count,) + conv_base.output_shape[1:]

    features = np.zeros(shape=(output_shape))
    labels = np.zeros(shape=(sample_count,), dtype=np.int)
    i = 0
    progress_bar = Progbar(target=sample_count)
    for inputs_batch, labels_batch in set_files.directory_iterator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch

        progress_bar.update(current=i * batch_size)
        i += 1
        if i * batch_size > sample_count:
            progress_bar.update(current=sample_count)
            break

    return Dataset(input_data=features, output_data=labels)


def build_corpus(corpus_file_structure: CorpusFileStructure,
                 name: str):
    """Builds the corpus by processing the source files

    Args:
        corpus_file_structure (CorpusFileStructure): corpus file structure
        name (str): corpus name

    """
    dirs = prepare_directories()
    copy_files(dirs, check=True)
    conv_base = load_convolutional_base()
    corpus_files = load_corpus_files(dirs=dirs, use_augmented=False, check=True)

    print('\nExtracting features from training set ({} files)'.format(corpus_files.training_set_files.length))
    training_set = extract_features(set_files=corpus_files.training_set_files, conv_base=conv_base)
    training_set.flatten_input_data()

    print('\nExtracting features from validation set ({} files)'.format(corpus_files.validation_set_files.length))
    validation_set = extract_features(set_files=corpus_files.validation_set_files, conv_base=conv_base)
    validation_set.flatten_input_data()

    print('\nExtracting features from test set ({} files)'.format(corpus_files.test_set_files.length))
    test_set = extract_features(set_files=corpus_files.test_set_files, conv_base=conv_base)
    test_set.flatten_input_data()

    corpus = Corpus(training_set=training_set,
                    test_set=test_set,
                    validation_set=validation_set,
                    name=name)

    corpus_file_structure.save_corpus(corpus=corpus)
    return corpus


def run(build_dataset: bool = False):
    corpus = None
    corpus_name = 'Cats and Dogs'
    base_path = 'data/features'
    corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus_name,
                                                              base_path=base_path)

    if build_dataset:
        corpus = build_corpus(corpus_file_structure=corpus_file_structure,
                              name=corpus_name)
    else:
        corpus = corpus_file_structure.load_corpus(corpus_name=corpus_name,
                                                   datasets_base_name=corpus_name)

    experiment = load_experiment(corpus=corpus)
    experiment.run(print_results=True, plot_history=False, display_progress_bars=True)
    experiment.plot_loss()
    experiment.plot_accuracy()
