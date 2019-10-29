dataset = None
word_index = dict()
reverse_word_index = dict()


def load_word_index(dataset):
    """ Loads the word index into the global variable
    """
    global word_index
    word_index = dataset.get_word_index()


def load_reverse_word_index():
    """ Creates the global reverse index from the word index
    """
    global word_index
    global reverse_word_index
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def get_word_indexes(dataset):
    """ Gets the direct and reverse word indices
    """
    load_word_index(dataset)
    load_reverse_word_index()
    return word_index, reverse_word_index


def decode(phrase_set, review_index: int):
    """ Decodes a phrase according to the """
    decoded = ' '.join([reverse_word_index.get(i - 3, '?') for i in phrase_set[review_index]])
    return decoded
