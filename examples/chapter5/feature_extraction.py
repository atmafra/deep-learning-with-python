import numpy as np
from keras import Model, optimizers
from keras.applications import VGG16

from core.corpus import Corpus, CorpusType
from core.experiment import Experiment
from core.network import ValidationStrategy
from core.neural_network import NeuralNetwork
from core.sets import SetFiles, Set
from core.training_configuration import TrainingConfiguration
from examples.chapter5.cats_and_dogs_files import prepare_directories, load_corpus_files, copy_files

classifier = [
    {'layer_type': 'Dense', 'units': 512, 'activation': 'relu', 'input_shape': (4 * 4 * 512, 1)},
    {'layer_type': 'Dropout', 'rate': 0.5},
    {'layer_type': 'Dense', 'units': 1, 'activation': 'sigmoid'}]

learning_rate = 1e-4
training_parameters = {
    'keras': {
        'compile': {
            'optimizer': optimizers.RMSprop(lr=learning_rate),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']},
        'fit_generator': {
            'epochs': 30,
            'steps_per_epoch': 100,
            'validation_steps': 50}},
    'validation': {
        'strategy': ValidationStrategy.NO_VALIDATION}}


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


def extract_features(set_files: SetFiles,
                     conv_base: Model):
    sample_count = set_files.length
    batch_size = set_files.batch_size
    output_shape = (sample_count,) + conv_base.output_shape[1:]

    features = np.zeros(shape=(output_shape))
    labels = np.zeros(shape=(sample_count,))

    i = 0
    for inputs_batch, labels_batch in set_files.directory_iterator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        i += 1
        if i * batch_size > sample_count:
            break

    return Set(input_data=features, output_data=labels)


def run():
    dirs = prepare_directories()
    copy_files(dirs, check=True)
    corpus_files = load_corpus_files(dirs=dirs, use_augmented=False, check=True)
    conv_base = load_convolutional_base()

    training_set = extract_features(set_files=corpus_files.training_set_files, conv_base=conv_base)
    validation_set = extract_features(set_files=corpus_files.validation_set_files, conv_base=conv_base)
    test_set = extract_features(set_files=corpus_files.test_set_files, conv_base=conv_base)

    training_set.flatten_input_data()
    validation_set.flatten_input_data()
    test_set.flatten_input_data()
    corpus = Corpus(training_set=training_set, test_set=test_set)
    experiment = load_experiment(corpus=corpus)
    experiment.run(print_results=True, plot_history=True, display_progress_bars=True)
