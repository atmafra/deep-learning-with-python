from examples.chapter3 import mnist, imdb, reuters, boston

if __name__ == '__main__':

    experiments = [
        'mnist',
        'imdb',
        'reuters1',
        'reuters2',
        'boston'
    ]

    if 'mnist' in experiments:
        mnist.run(build=True, load_model_configuration=True)

    if 'imdb' in experiments:
        imdb.run(build=True)

    if 'reuters1' in experiments:
        reuters.run(num_words=10000, encoding_schema='one-hot', build=True)

    if 'reuters2' in experiments:
        reuters.run(num_words=10000, encoding_schema='int-array', build=True)

    if 'boston' in experiments:
        boston.run(build=False)
