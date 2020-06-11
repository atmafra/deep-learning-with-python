from core.file_structures import CorpusFileStructure
from corpora.rutger import rutger
from examples.chapter6.rutger_configurations import load_experiment_plan

default_rutger_path = '../../corpora/rutger/data'
default_filename = 'rutger-2020-06-03.csv'
default_input_length = 100
default_sampling_rate = .1
default_training_set_share = .6
default_validation_set_share = .2
default_test_set_share = .2
default_vocabulary_size = 10000
default_embeddings_dimension = 100


def run(build_corpus: bool = True,
        sampling_rate: float = default_sampling_rate,
        training_set_share: float = default_training_set_share,
        validation_set_share: float = default_validation_set_share,
        test_set_share: float = default_test_set_share,
        input_length: int = default_input_length,
        vocabulary_size: int = default_vocabulary_size,
        embeddings_dimension: int = default_embeddings_dimension):
    corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name='Rutger', base_path=default_rutger_path)
    if build_corpus:
        corpus = rutger.build_corpus(rutger_path=default_rutger_path,
                                     filename=default_filename,
                                     vocabulary_size=vocabulary_size,
                                     input_length=default_input_length,
                                     sampling_rate=sampling_rate,
                                     training_set_share=training_set_share,
                                     validation_set_share=validation_set_share,
                                     test_set_share=test_set_share,
                                     verbose=True)

        # corpus_file_structure.save_corpus(corpus)
    else:
        corpus = corpus_file_structure.load_corpus(corpus_name='Rutger', datasets_base_name='Rutger')

    # rutger_embeddings = load_embeddings_experiment(corpus=corpus,
    #                                                input_length=default_input_length,
    #                                                vocabulary_size=default_vocabulary_size,
    #                                                embeddings_dimension=embeddings_dimension)

    rutger_experiment_plan = load_experiment_plan(corpus=corpus,
                                                  input_length=input_length,
                                                  vocabulary_size=vocabulary_size,
                                                  embeddings_dimension=embeddings_dimension)

    rutger_experiment_plan.run(train=True,
                               test=True,
                               print_results=True,
                               plot_training_loss=True,
                               plot_validation_loss=True,
                               plot_training_accuracy=True,
                               plot_validation_accuracy=True,
                               display_progress_bars=True,
                               save_models=True,
                               models_path='models/rutger')

