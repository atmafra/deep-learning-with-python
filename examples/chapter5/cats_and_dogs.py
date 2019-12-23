from examples.chapter5.cats_and_dogs_configurations import *
from examples.chapter5.cats_and_dogs_files import *


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
    corpus_generator = load_corpus_files(dirs=dirs, use_augmented=False, check=True)
    corpus_generator_augmented = load_corpus_files(dirs=dirs, use_augmented=True, check=True)

    experiment_plan = load_experiment_plan(corpus_generator=corpus_generator,
                                           corpus_generator_augmented=corpus_generator_augmented)

    experiment_plan.run(print_results=False,
                        plot_training_loss=True,
                        plot_training_accuracy=True,
                        display_progress_bars=True)

    experiment_plan.save_models('models/cats_and_dogs')
