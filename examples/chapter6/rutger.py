from core.file_structures import CorpusFileStructure
from corpora.rutger import rutger
from examples.chapter6.rutger_configurations import load_experiment_plan

default_rutger_path = '../../corpora/rutger/data'
default_rutger_golden_path = default_rutger_path + '/golden'
default_filename = 'rutger-2020-06-03.csv'
default_golden_filename = 'Version33.csv'
default_input_length = 100
default_sampling_rate = 1.
default_training_set_share = .8
default_validation_set_share = .1
default_test_set_share = .1
default_clean = True
default_vocabulary_size = 10000
default_embeddings_dimension = 100


def run(build_corpus: bool = True,
        clean: bool = default_clean,
        sampling_rate: float = default_sampling_rate,
        training_set_share: float = default_training_set_share,
        validation_set_share: float = default_validation_set_share,
        test_set_share: float = default_test_set_share,
        input_length: int = default_input_length,
        vocabulary_size: int = default_vocabulary_size,
        embeddings_dimension: int = default_embeddings_dimension,
        train_model: bool = True):
    """ Runs the Rutger Intent Detection experiments

    :param build_corpus: build corpus (tries to load corpus if False)
    :param clean: clean phrases before tokenization
    :param sampling_rate: corpus data sampling rate
    :param training_set_share: samples share to be used for training
    :param validation_set_share: samples share to be used as validation
    :param test_set_share: samples share to be used as test
    :param input_length: maximum tokens per phrase (padding/truncation possible)
    :param vocabulary_size: maximum number of tokes to be kept by the tokenizer
    :param embeddings_dimension: number of units in the embeddings layer
    :param train_model: execute the training of the neural network model
    """
    corpus_name = rutger.get_name(clean=clean,
                                  vocabulary_size=vocabulary_size,
                                  input_length=input_length,
                                  sampling_rate=sampling_rate,
                                  training_set_share=training_set_share,
                                  validation_set_share=validation_set_share,
                                  test_set_share=test_set_share)

    corpus_file_structure = CorpusFileStructure.get_canonical(corpus_name=corpus_name, base_path=default_rutger_path)
    output_encoder = None
    if build_corpus:
        corpus, tokenizer, output_encoder = \
            rutger.build_corpus(rutger_path=default_rutger_path,
                                filename=default_filename,
                                clean=clean,
                                vocabulary_size=vocabulary_size,
                                input_length=input_length,
                                sampling_rate=sampling_rate,
                                training_set_share=training_set_share,
                                validation_set_share=validation_set_share,
                                test_set_share=test_set_share,
                                verbose=True)

        print('Saving corpus "{}"'.format(corpus_name))
        corpus_file_structure.save_corpus(corpus)
    else:
        print('Loading corpus "{}"'.format(corpus_name))
        corpus = corpus_file_structure.load_corpus(corpus_name=corpus_name, datasets_base_name=corpus_name)
        tokenizer = None
        print('Corpus has {} samples'.format(corpus.length))

    if train_model:
        rutger_experiment_plan = load_experiment_plan(corpus=corpus,
                                                      vocabulary_size=vocabulary_size,
                                                      embeddings_dimension=embeddings_dimension)

        rutger_experiment_plan.run(train=True,
                                   test=True,
                                   use_sample_weights=True,
                                   print_results=True,
                                   plot_training_loss=True,
                                   plot_validation_loss=True,
                                   plot_training_accuracy=True,
                                   plot_validation_accuracy=True,
                                   display_progress_bars=True,
                                   save_models=True,
                                   models_path='models/rutger')

        golden_dataset = rutger.build_golden_dataset(rutger_golden_path=default_rutger_golden_path,
                                                     rutger_golden_filename=default_golden_filename,
                                                     clean=clean,
                                                     vocabulary_size=vocabulary_size,
                                                     input_length=input_length,
                                                     sampling_rate=1.,
                                                     tokenizer=tokenizer,
                                                     output_encoder=output_encoder,
                                                     verbose=True)

        golden_results = rutger_experiment_plan.evaluate(dataset=golden_dataset,
                                                         use_sample_weights=False,
                                                         display_progress_bars=True)

        with open('golden_results.txt', 'w') as f:
            for item in golden_results:
                f.write("{}".format(item))

        golden_predictions = rutger_experiment_plan.predict(dataset=golden_dataset,
                                                            display_progress_bars=True)

        golden_categories = output_encoder.decode(encoded_sequence=golden_predictions)[0]

        with open('golden_categories.txt', 'w') as f:
            for category in golden_categories:
                f.write("{}\n".format(category))
            f.writelines(golden_categories)
