from corpora.jena_climate import jena_climate


def run():
    jena_climate.build_corpus(jena_path='../../corpora/jena_climate/data')
