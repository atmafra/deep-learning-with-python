import json
import os
import re
import nltk
from keras_preprocessing.text import Tokenizer, tokenizer_from_json

""" Preprocessing code copied from

Title: Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks
Author: Nathan Hartmann, Erick Fonseca, Christopher Shulby, Marcos Treviso, Jessica Rodrigues, Sandra Aluisio
Date: June 15, 2020
Availability:  https://github.com/nathanshartmann/portuguese_word_embeddings/blob/master/preprocessing.py
"""
sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')

# Punctuation list
punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')

re_remove_brackets = re.compile(r'\{.*\}')
re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
re_transform_numbers = re.compile(r'\d', re.UNICODE)
re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
# Different quotes are used.
re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
re_tree_dots = re.compile(u'…', re.UNICODE)
# Differents punctuation patterns are used.
re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                       (punctuations, punctuations), re.UNICODE)
re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                         (punctuations, punctuations), re.UNICODE)
re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
re_changehyphen = re.compile(u'–')
re_doublequotes_1 = re.compile(r'(\"\")')
re_doublequotes_2 = re.compile(r'(\'\')')
re_trim = re.compile(r' +', re.UNICODE)


def build_tokenizer(phrase_list: list,
                    vocabulary_size: int,
                    verbose: bool = True):
    """ Builds the global tokenizer

    :param phrase_list: list of input phrases
    :param vocabulary_size: maximum vocabulary size
    :param verbose: display messages during processing
    :return: tokenizer object, fit to input data
    """
    if verbose:
        print('Tokenizing (maximum {} words)...'.format(vocabulary_size))
        print('Fitting tokenizer to the current dataset')

    tokenizer = Tokenizer(num_words=vocabulary_size,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=' ',
                          char_level=False,
                          oov_token=None,
                          document_count=0)

    tokenizer.fit_on_texts(texts=phrase_list)
    if verbose:
        word_index = tokenizer.word_index
        print('Word index has {} unique tokens'.format(len(word_index)))

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer,
                   path: str,
                   filename: str,
                   verbose: bool = True):
    """ Saves a tokenizer configuration to file

    :param tokenizer: tokenizer
    :param path: system path of the tokenizer file
    :param filename: tokenizer filename
    :param verbose: output progress messages in the terminal
    """
    if tokenizer is None:
        raise RuntimeError('Cannot save tokenizer: no tokenizer object passed')

    filepath = os.path.join(path, filename)
    if verbose:
        print('Saving tokenizer to file "{}"'.format(filepath))

    with open(filepath, 'w') as file:
        json.dump(tokenizer.to_json(), file)
    file.close()


def load_tokenizer(path: str,
                   filename: str,
                   verbose: bool = True):
    """ Creates a new Tokenizer object from the configurations saved on a json file


    :param path: system path of the tokenizer file
    :param filename: tokenizer filename
    :param verbose: output progress messages in the terminal
    :return: loaded tokenizer
    """
    if not filename:
        raise RuntimeError('Cannot load tokenizer: no file path passed')

    filepath = os.path.join(path, filename)
    if verbose:
        print('Loading tokenizer from file "{}"'.format(filepath))

    with open(filepath, 'r') as file:
        tokenizer_configuration = json.load(file)
    file.close()

    return tokenizer_from_json(tokenizer_configuration)


def tokenize_text(texts: list,
                  tokenizer: Tokenizer):
    """ Tokenizes the list of phrases using the given tokenizer

    :param texts: list of phrases to tokenize
    :param tokenizer: pre-built tokenizer
    :return: list of tokenized phrases
    """
    """ Splits text reviews into tokens (words, in this particular case)
    """
    if tokenizer is None:
        raise RuntimeError('Cannot tokenize text: no tokenizer passed')

    sequences = tokenizer.texts_to_sequences(texts=texts)
    return sequences


def clean_text(text):
    """ Apply all regex above to a given string

    :param text: input text
    :return: clean text
    """
    text = text.lower()
    text = text.replace('\xa0', ' ')
    text = re_tree_dots.sub('...', text)
    text = re.sub('\.\.\.', '', text)
    text = re_remove_brackets.sub('', text)
    text = re_changehyphen.sub('-', text)
    text = re_remove_html.sub(' ', text)
    text = re_transform_numbers.sub('0', text)
    text = re_transform_url.sub('URL', text)
    text = re_transform_emails.sub('EMAIL', text)
    text = re_quotes_1.sub(r'\1"', text)
    text = re_quotes_2.sub(r'"\1', text)
    text = re_quotes_3.sub('"', text)
    text = re.sub('"', '', text)
    text = re_dots.sub('.', text)
    text = re_punctuation.sub(r'\1', text)
    text = re_hiphen.sub(' - ', text)
    text = re_punkts.sub(r'\1 \2 \3', text)
    text = re_punkts_b.sub(r'\1 \2 \3', text)
    text = re_punkts_c.sub(r'\1 \2', text)
    text = re_doublequotes_1.sub('\"', text)
    text = re_doublequotes_2.sub('\'', text)
    text = re_trim.sub(' ', text)
    return text.strip()
