from examples.chapter5.cats_and_dogs_configurations import *
from examples.chapter5.cats_and_dogs_files import *


def run(prepare_files: bool = True,
        train_models: bool = True,
        visualize_activations: bool = False):

    if prepare_files:
        dirs = prepare_directories()
        copy_files(dirs=dirs, check=True)

    corpus_generator = load_corpus_files(dirs=dirs,
                                         use_augmented=False,
                                         check=True)

    corpus_generator_augmented = load_corpus_files(dirs=dirs, use_augmented=True, check=True)

    experiment_plan = load_experiment_plan(corpus_generator=corpus_generator,
                                           corpus_generator_augmented=corpus_generator_augmented)

    experiment_plan.run(train=True,
                        test=True,
                        print_results=True,
                        plot_training_loss=True,
                        plot_training_accuracy=True,
                        display_progress_bars=True,
                        save_models=True,
                        models_path='models/cats_and_dogs')
