from examples.chapter3 import mnist, imdb, reuters, boston

if __name__ == '__main__':
    mnist.run(build=False)
    # imdb.run()
    # reuters.run(num_words=10000, encoding_schema='one-hot')
    # reuters.run(num_words=10000, encoding_schema='int-array')
    # boston.run()
