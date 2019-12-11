import os
import shutil

import utils.parameter_utils as putl
from core.sets import SetGenerator, CorpusGenerator
from examples.chapter5.cats_and_dogs_configurations import load_experiment_plan
from utils.image_utils import get_image_generator

root_dir = '/Users/alexandre.mafra/Documents/projetos/neural-network/deep-learning-with-python/'
data_dir = os.path.join(root_dir, 'examples/chapter5/data/')


def prepare_directories() -> dict:
    """Prepare the files and organizes the directories
       Returns a dictionary of directory keys and their respective locations

    """
    original_dataset_dir = os.path.join(data_dir, 'original')
    original_train_dir = os.path.join(original_dataset_dir, 'train')
    original_test_dir = os.path.join(original_dataset_dir, 'test1')

    base_dir = os.path.join(data_dir, 'cats_and_dogs_small')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    test_cats_dir = os.path.join(test_dir, 'cats')
    test_dogs_dir = os.path.join(test_dir, 'dogs')

    # creates the directories
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)

    os.mkdir(base_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)

    os.mkdir(validation_cats_dir)
    os.mkdir(validation_dogs_dir)

    os.mkdir(test_dogs_dir)
    os.mkdir(test_cats_dir)

    dirs = {'original_dataset': original_dataset_dir,
            'original_train': original_train_dir,
            'original_test': original_test_dir,
            'train': train_dir,
            'validation': validation_dir,
            'test': test_dir,
            'train_cats': train_cats_dir,
            'validation_cats': validation_cats_dir,
            'test_cats': test_cats_dir,
            'train_dogs': train_dogs_dir,
            'validation_dogs': validation_dogs_dir,
            'test_dogs': test_dogs_dir}

    return dirs


def copy_files_mask(mask: str,
                    start: int,
                    end: int,
                    from_dir: str,
                    to_dir: str):
    """Copy batches files from one directory to another
       The file names to be copied must fit a mask with numbers (start and end)

    Args:
        mask (str): file mask to be applied in the origin directory
        start (int): start number to be injected in the file name mask
        end (int): end number to be injected in the file name mask
        from_dir (str): source directory
        to_dir (str): destination directory

    """
    fnames = [mask.format(i) for i in range(start, end)]
    for fname in fnames:
        src = os.path.join(from_dir, fname)
        dst = os.path.join(to_dir, fname)
        shutil.copyfile(src, dst)


def copy_files(dirs: dict, check: bool = True):
    """Copies files from original dir to structured dirs
    """
    original_dataset_dir = putl.get_parameter(parameters=dirs, key='original_dataset', mandatory=True)
    original_train_dir = putl.get_parameter(parameters=dirs, key='original_train', mandatory=True)
    original_test_dir = putl.get_parameter(parameters=dirs, key='original_test', mandatory=True)
    train_cats_dir = putl.get_parameter(parameters=dirs, key='train_cats', mandatory=True)
    validation_cats_dir = putl.get_parameter(parameters=dirs, key='validation_cats', mandatory=True)
    test_cats_dir = putl.get_parameter(parameters=dirs, key='test_cats', mandatory=True)
    train_dogs_dir = putl.get_parameter(parameters=dirs, key='train_dogs', mandatory=True)
    validation_dogs_dir = putl.get_parameter(parameters=dirs, key='validation_dogs', mandatory=True)
    test_dogs_dir = putl.get_parameter(parameters=dirs, key='test_dogs', mandatory=True)

    copy_files_mask(mask='cat.{}.jpg', start=0, end=1000, from_dir=original_train_dir, to_dir=train_cats_dir)
    copy_files_mask(mask='cat.{}.jpg', start=1000, end=1500, from_dir=original_train_dir, to_dir=validation_cats_dir)
    copy_files_mask(mask='cat.{}.jpg', start=1500, end=2000, from_dir=original_train_dir, to_dir=test_cats_dir)

    copy_files_mask(mask='dog.{}.jpg', start=0, end=1000, from_dir=original_train_dir, to_dir=train_dogs_dir)
    copy_files_mask(mask='dog.{}.jpg', start=1000, end=1500, from_dir=original_train_dir, to_dir=validation_dogs_dir)
    copy_files_mask(mask='dog.{}.jpg', start=1500, end=2000, from_dir=original_train_dir, to_dir=test_dogs_dir)

    if check:
        check_files(dirs=dirs)


def check_files(dirs: dict):
    """Sanity checks if file copying was ok
    """
    print('Training cat images  :', len(os.listdir(dirs['train_cats'])))
    print('Training dog images  :', len(os.listdir(dirs['train_dogs'])))
    print('Validation cat images:', len(os.listdir(dirs['validation_cats'])))
    print('Validation dog images:', len(os.listdir(dirs['validation_dogs'])))
    print('Test cat images      :', len(os.listdir(dirs['test_cats'])))
    print('Test dog images:     :', len(os.listdir(dirs['test_dogs'])))


def load_corpus_generator(dirs: dict, use_augmented: bool, check: bool = False) -> CorpusGenerator:
    """Loads the corpus from files in the directory structure
    """
    train_dir = putl.get_parameter(parameters=dirs, key='train')
    validation_dir = putl.get_parameter(parameters=dirs, key='validation')
    test_dir = putl.get_parameter(parameters=dirs, key='test')

    rescale_factor = 1. / 255
    target_size = (150, 150)
    batch_size = 20
    class_mode = 'binary'

    training_generator = get_image_generator(rescale_factor=rescale_factor,
                                             source_dir=train_dir,
                                             target_size=target_size,
                                             batch_size=batch_size,
                                             class_mode=class_mode,
                                             use_augmented=use_augmented)

    validation_generator = get_image_generator(rescale_factor=rescale_factor,
                                               source_dir=validation_dir,
                                               target_size=target_size,
                                               batch_size=batch_size,
                                               class_mode=class_mode)

    test_generator = get_image_generator(rescale_factor=rescale_factor,
                                         source_dir=test_dir,
                                         target_size=target_size,
                                         batch_size=batch_size,
                                         class_mode=class_mode)

    training_set_generator = SetGenerator(generator=training_generator)
    validation_set_generator = SetGenerator(generator=validation_generator)
    test_set_generator = SetGenerator(generator=test_generator)

    corpus_generator = CorpusGenerator(training_set_generator=training_set_generator,
                                       validation_set_generator=validation_set_generator,
                                       test_set_generator=test_set_generator)

    return corpus_generator


def check_generator(set_generator: SetGenerator):
    for data_batch, labels_batch in set_generator.generator:
        print('data batch shape  :', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break


# def test_data_augmentation(dirs: dict):
#     train_cats_dir = putl.get_parameter(parameters=dirs, key='train_cats', mandatory=True)
#     fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
#     img_path = fnames[3]
#     img = image.load_img(img_path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#     datagen = get_augmented_image_generator()
#     i = 0
#     for batch in datagen.flow(x, batch_size=1):
#         plt.figure(i)
#         imgplot = plt.imshow(image.array_to_img(batch[0]))
#         i += 1
#         if i % 10 == 0:
#             break
#         plt.show()


def run():
    dirs = prepare_directories()
    copy_files(dirs=dirs, check=True)
    corpus_generator = load_corpus_generator(dirs=dirs, use_augmented=False, check=True)
    corpus_generator_augmented = load_corpus_generator(dirs=dirs, use_augmented=True, check=True)

    experiment_plan = load_experiment_plan(corpus_generator=corpus_generator,
                                           corpus_generator_augmented=corpus_generator_augmented)

    experiment_plan.run(print_results=False,
                        plot_training_loss=True,
                        plot_training_accuracy=True,
                        display_progress_bars=True)

    experiment_plan.save_models('models/json')
