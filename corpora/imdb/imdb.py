import os

import numpy as np
from keras import preprocessing
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from core.corpus import Corpus
from core.file_structures import CorpusFileStructure
from utils.dataset_utils import one_hot_encode, separate_corpus

tokenizer = None
word_index = None


# def download_raw_data(url: str = 'http://mng.bz/0tIo',
#                       download_path: str = ''):
#     print('Trying to download IMDB dataset from "{}"...'.format(url))
#     response = requests.get(url=url)
#     print('HTTP status code: {}'.format(response.status_code))
#     print('content-type: {}'.format(response.headers['content-type']))
#     print('encoding: {}'.format(response.encoding))
#     with open(download_path, 'wb') as f:
#         f.write(response.content)
#     print('File saved to "{}"'.format(download_path))

def __load_raw_data(imdb_path: str, subdir: str):
    """ Loads raw IMDB reviews data into train and labels list

    :param imdb_path: system path for the IMDB reviews dataset root dir
    :param subdir: train/test subdirectory
    """
    source_dir = os.path.join(imdb_path, subdir)
    print('Loading IMDB {} reviews files from \'{}\'...'.format(subdir, source_dir))
    texts = []
    labels = []

    for label_type in ['neg', 'pos']:
        current_dir = os.path.join(source_dir, label_type)
        dirlist = os.listdir(current_dir)
        print('Found {} files in directory \'{}\''.format(len(dirlist), current_dir))

        for filename in dirlist:
            if filename[-4:] == '.txt':
                f = open(os.path.join(current_dir, filename))
                texts.append(f.read())
                f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

    return texts, labels


def __get_tokenizer(num_words: int):
    """ Gets an instance of a tokenizer
    """
    global tokenizer
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=num_words)
    return tokenizer


def __tokenize_text(texts: list,
                    fit: bool,
                    max_words: int):
    """ Splits text reviews into tokens (words, in this particular case)
    """
    global tokenizer
    tokenizer = __get_tokenizer(num_words=max_words)
    print('Tokenizing (maximum {} words)...'.format(max_words))
    if fit:
        print('Fitting tokenizer to the current dataset')
        tokenizer.fit_on_texts(texts=texts)
        global word_index
        word_index = tokenizer.word_index
        print('Word index has {} unique tokens'.format(len(word_index)))

    sequences = tokenizer.texts_to_sequences(texts=texts)
    return sequences


def __randomize_order(data, labels):
    """ Randomize the order of data (inputs) and labels (outputs)
    """
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], labels[indices]


def __get_samples(data: list,
                  labels: np.ndarray,
                  training_samples: int,
                  validation_samples: int):
    """ Samples the data for training and validation datasets
    """
    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples:training_samples + validation_samples]
    y_val = labels[training_samples:training_samples + validation_samples]
    return x_train, y_train, x_val, y_val


def build_corpus(imdb_path: str = 'corpora/imdb/data/aclImdb',
                 corpus_name: str = 'IMDB',
                 maximum_tokens_per_text: int = 100,
                 vocabulary_size: int = 10000,
                 randomize: bool = True,
                 training_samples: int = -1,
                 validation_samples: int = 0):
    """ Build the IMDB sentiment analysis corpus from the original text files

    :param imdb_path: system path of the IMDB data directory
    :param corpus_name: corpus name
    :param maximum_tokens_per_text: maximum tokens per text
    :param vocabulary_size: maximum vocabulary size
    :param randomize: randomize sample order in the set
    :param training_samples: size of the training set (default: all available samples)
    :param validation_samples: size of the validation set (default: 0)
    :return: the IMDB corpus
    """
    # Training and Validation Data
    training_texts, training_labels = __load_raw_data(imdb_path=imdb_path, subdir='train')
    training_sequences = __tokenize_text(texts=training_texts, fit=True, max_words=vocabulary_size)
    training_data = pad_sequences(training_sequences, maxlen=maximum_tokens_per_text)
    training_labels = np.asarray(training_labels)
    set_size = len(training_data)

    if randomize:
        training_data, training_labels = __randomize_order(training_data, training_labels)

    if training_samples < 0:
        training_samples = set_size - validation_samples

    if training_samples > 0 and validation_samples > 0 and training_samples + validation_samples > set_size:
        raise RuntimeError('Error building IMDB Dataset: number of training samples requestes ({}) '
                           'plus validation samples ({}) is larger than the Dataset size ({})'
                           .format(training_samples, validation_samples, set_size))

    x_train, y_train, x_val, y_val = __get_samples(data=training_data,
                                                   labels=training_labels,
                                                   training_samples=training_samples,
                                                   validation_samples=validation_samples)

    # Test Data
    test_texts, test_labels = __load_raw_data(imdb_path=imdb_path, subdir='test')
    test_sequences = __tokenize_text(texts=test_texts, fit=False, max_words=vocabulary_size)
    test_data = pad_sequences(test_sequences, maxlen=maximum_tokens_per_text)
    test_labels = np.asarray(test_labels)

    corpus = Corpus.from_datasets(training_input=x_train,
                                  training_output=y_train,
                                  validation_input=x_val,
                                  validation_output=y_val,
                                  test_input=test_data,
                                  test_output=test_labels,
                                  name=corpus_name)

    return corpus


def load_preprocessed_corpus(name: str,
                             encoding: str = 'int-array',
                             words: int = 10000,
                             maxlen: int = 20,
                             save: bool = True,
                             verbose: bool = True) -> Corpus:
    """ Loads the IMDB dataset into a corpus object

    :param name: corpus name
    :param encoding: how to encode the data, as an array of integers (int-array) or as a boolean vector (one-hot)
    :param words: word limit in the reverse index
    :param maxlen: maximum number of words per sentence
    :param save: save the pre-processed datafiles
    :param verbose: outputs progress messages
    """
    if verbose:
        print("Loading IMDB dataset...")

    corpus_datasets = separate_corpus(imdb.load_data(num_words=words))
    train_samples = corpus_datasets[0][0]
    train_labels = corpus_datasets[0][1]
    test_samples = corpus_datasets[1][0]
    test_labels = corpus_datasets[1][1]

    if encoding == 'int-array':
        training_inputs = preprocessing.sequence.pad_sequences(train_samples, maxlen=maxlen)
        test_inputs = preprocessing.sequence.pad_sequences(test_samples, maxlen=maxlen)

    elif encoding == 'one-hot':
        training_inputs = one_hot_encode(train_samples, words)
        test_inputs = one_hot_encode(test_samples, words)

    else:
        raise RuntimeError('Invalid encoding schema: {}'.format(encoding))

    # vectorize the labels
    training_outputs = np.asarray(train_labels).astype('float32')
    test_outputs = np.asarray(test_labels).astype('float32')

    # create the corpus
    corpus = Corpus.from_datasets(training_input=training_inputs,
                                  training_output=training_outputs,
                                  test_input=test_inputs,
                                  test_output=test_outputs,
                                  name=name)

    if verbose:
        print("{} train reviews loaded".format(corpus.training_set.length))
        print("{} test reviews loaded".format(corpus.test_set.length))

    if save:
        save_corpus(corpus)

    return corpus


def save_corpus(corpus: Corpus, corpus_file_structure: CorpusFileStructure = None):
    """ Saves the corpus according to the file structure

    :param corpus: corpus to be saved
    :param corpus_file_structure: file structure map
    """
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus.name, base_path='data')
    corpus_file_structure.save_corpus(corpus)


def load_corpus(corpus_name: str, corpus_file_structure: CorpusFileStructure = None):
    """ Loads a previously saved corpus according to the file structure

    :param corpus_name: corpus name (not in the file)
    :param corpus_file_structure: file structure map
    :return: new corpus, loaded from files
    """
    if corpus_file_structure is None:
        corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus_name, base_path='data')
    return corpus_file_structure.load_corpus(corpus_name=corpus_name, datasets_base_name=corpus_name)
