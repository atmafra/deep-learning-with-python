import sys
import os
import numpy as np

glove_base_dir = os.path.dirname(sys.modules[__name__].__file__)
default_glove_dir = os.path.join(glove_base_dir, 'data/glove.6B')
default_glove_embeddings_filename_template = 'glove.6B.{}d.txt'
default_embeddings_dimension = 100
default_vocabulary_size = 10000


def __load_embeddings_from_files(glove_path: str,
                                 glove_embeddings_filename_template: str,
                                 embeddings_dimension: int) -> dict:
    """ Loads the GloVe embeddings from downloaded files

    :param glove_path: system path of the GloVe embeddings files
    :param glove_embeddings_filename_template: filename of the specific version of the GloVe embeddings
    :param embeddings_dimension: embeddings vector dimension (ex.: 50, 100, 200, 300)
    :return: embeddings index
    """
    embeddings_index = {}
    glove_embeddings_filename = glove_embeddings_filename_template.format(embeddings_dimension)
    print('Loading GloVe embeddings from file \'{}\''.format(glove_embeddings_filename))
    f = open(os.path.join(glove_path, glove_embeddings_filename))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found {} word vectors'.format(len(embeddings_index)))
    return embeddings_index


def __prepare_embeddings_matrix(embeddings_index: dict,
                                vocabulary_size: int,
                                embeddings_dim: int):
    """ Converts the embeddings index into an embeddings matrix of fixed dimensions

    :param embeddings_index: previously loaded embeddings index
    :param vocabulary_size: maximum number of tokens to be considered (for one-hot encoding)
    :param embeddings_dim: dimension of the embeddings vector
    :return: GloVe embeddings matrix
    """
    embeddings_matrix = np.zeros(shape=(vocabulary_size, embeddings_dim))
    i = 1
    for word, embedding_vector in embeddings_index.items():
        if i < vocabulary_size:
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector
        i += 1

    return embeddings_matrix


def load_embeddings_matrix(glove_path: str = default_glove_dir,
                           glove_filename_template: str = default_glove_embeddings_filename_template,
                           embeddings_dimension: int = default_embeddings_dimension,
                           vocabulary_size: int = default_vocabulary_size):
    """ Loads the GloVe embeddings from files and returns the embeddings matrix

    :param glove_path: system path of the GloVe embeddings files
    :param glove_filename_template: filename of the specific version of the GloVe embeddings
    :param embeddings_dimension: dimension of the embeddings vector
    :param vocabulary_size: maximum number of tokens to be considered (for one-hot encoding)
    :return: GloVe embeddings matrix
    """
    embeddings_index = __load_embeddings_from_files(glove_path=glove_path,
                                                    glove_embeddings_filename_template=glove_filename_template,
                                                    embeddings_dimension=embeddings_dimension)

    embeddings_matrix = __prepare_embeddings_matrix(embeddings_index=embeddings_index,
                                                    vocabulary_size=vocabulary_size,
                                                    embeddings_dim=embeddings_dimension)

    return embeddings_matrix
