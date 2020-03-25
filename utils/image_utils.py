import matplotlib.pyplot as plt
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator


def get_image_directory_iterator(source_dir: str,
                                 target_size: tuple,
                                 batch_size: int,
                                 class_mode: str,
                                 rescale_factor: float,
                                 use_augmented: bool,
                                 rotation_range: int = 0,
                                 width_shift_range: float = 0.,
                                 height_shift_range: float = 0.,
                                 shear_range: float = 0.,
                                 zoom_range: float = 0.,
                                 horizontal_flip: bool = False,
                                 fill_mode: str = 'nearest'):
    """
    Args:
        source_dir (str): image source directory path
        target_size (tuple): target image shape
        batch_size (int): size of the image batch that is read from disk
        class_mode (str):
        rescale_factor (float): multiply element values by this scale factor
        use_augmented (bool): apply transformations to the image to implement data augmentation
        rotation_range (int): degree range for random rotations
        width_shift_range (float): part of width for random horizontal shift
        height_shift_range (float): part of width for random vertical shift
        shear_range (float): angle in degrees for random shear
        zoom_range (float): random zoom range
        horizontal_flip (boolean): allow random horizontal flip
        fill_mode (str): how points outside the boundaries are filled("constant", "nearest", "reflect" or "wrap")

    """
    datagen = None

    if not use_augmented:
        datagen = ImageDataGenerator(rescale=rescale_factor)
    else:
        datagen = ImageDataGenerator(
            # parameters
            rescale=rescale_factor,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode,
            # additional attributes (default values)
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            brightness_range=None,
            channel_shift_range=0.,
            cval=0.,
            vertical_flip=False,
            preprocessing_function=None,
            data_format='channels_last',
            validation_split=0.0,
            interpolation_order=1,
            featurewise_center=False,
            dtype='float32')

    if datagen is None:
        raise RuntimeError('Error creating ImageDataGenerator')

    return datagen.flow_from_directory(directory=source_dir,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       class_mode=class_mode)


def show_sample(generator: DirectoryIterator,
                num_samples: int = 20,
                rows: int = 4,
                columns: int = 5):
    """Displays samples from the first batch of the image generator
    """
    batch = next(generator)
    images_list = batch[0]
    classification_list = batch[1]
    if num_samples > len(images_list):
        num_samples = len(images_list)
    fig = plt.figure(figsize=(8, 6))

    for i in range(0, num_samples):
        current_image = image.array_to_img(images_list[i])
        #current_classification = generator.class_indices[classification_list[i]]
        subplot = fig.add_subplot(rows, columns, i + 1)
        #subplot.title.set_text(current_classification)
        plt.imshow(current_image)

    plt.show()
