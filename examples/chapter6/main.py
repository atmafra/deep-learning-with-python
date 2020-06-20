from examples.chapter6 import one_hot_encoding, word_embeddings, recurrent_toy, recurrent_imdb, temperature_forecasting, \
    rutger

if __name__ == '__main__':

    experiments = [
        # 'one_hot_encoding',
        # 'word_embeddings',
        # 'recurrent_toy',
        # 'recurrent_imdb',
        # 'temperature_forecasting'
        'rutger'
    ]

    if 'one_hot_encoding' in experiments:
        one_hot_encoding.run()

    if 'word_embeddings' in experiments:
        word_embeddings.run(build=True,
                            maximum_tokens_per_text=100,
                            vocabulary_size=10000)

    if 'recurrent_toy' in experiments:
        recurrent_toy.run()

    if 'recurrent_imdb' in experiments:
        recurrent_imdb.run()

    if 'temperature_forecasting' in experiments:
        temperature_forecasting.run()

    if 'rutger' in experiments:
        rutger.run(build_corpus=True, train_model=True)
