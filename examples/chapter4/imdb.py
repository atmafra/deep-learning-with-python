from corpora.imdb.imdb import load_preprocessed_corpus, load_corpus
from examples.chapter4.imdb_configurations import *


def run(plan: str = 'comparison', build: bool = True):
    """ Runs the selected experiment plan

    :param plan: key of the experiment to run. Possible values are:
                 'dropout', 'weight_regularization_l1', 'weight_regularization_l2', 'comparison'
    :param build: force rebuild of the datasets
    """
    # loads the corpus and the experiment plans
    corpus_name = 'IMDB'
    if build:
        corpus = load_preprocessed_corpus(name=corpus_name,
                                          encoding='one-hot',
                                          words=num_words,
                                          maxlen=20,
                                          save=True,
                                          verbose=True)
    else:
        corpus = load_corpus(corpus_name=corpus_name)

    experiments = load_experiments(corpus=corpus)

    # runs the selected experiment plan
    experiment_plan = experiments[plan]

    experiment_plan.run(train=True,
                        test=True,
                        print_results=True,
                        plot_training_loss=True,
                        plot_validation_loss=True,
                        plot_training_accuracy=True,
                        plot_validation_accuracy=True,
                        display_progress_bars=True,
                        save_models=True,
                        models_path='models')
