from keras_preprocessing.image import ImageDataGenerator


def get_image_generator(rescale_factor: float,
                        source_dir: str,
                        target_size: tuple,
                        batch_size: int,
                        class_mode: str,
                        use_augmented: bool = False,
                        width_shift_range: float = 0.2,
                        height_shift_range: float = 0.2,
                        rotation_range: int = 40,
                        shear_range: float = 0.2,
                        zoom_range: float = 0.2,
                        horizontal_flip: bool = True,
                        fill_mode: str = 'nearest'):
    """
    Args:
        rescale_factor (float): multiply element values by this scale factor
        source_dir (str): image source directory path
        target_size (tuple): target image shape
        width_shift_range(float): part of width for random horizontal shift
        height_shift_range(float): part of width for random vertical shift
        rotation_range(int): degree range for random rotations
        shear_range(float): angle in degrees for random shear
        zoom_range(float): random zoom range
        horizontal_flip(boolean): allow random horizontal flip
        fill_mode(str): how points outside the boundaries are filled("constant", "nearest", "reflect" or "wrap")

    """
    datagen = None

    if not use_augmented:
        datagen = ImageDataGenerator(rescale=rescale_factor)
    else:
        datagen = ImageDataGenerator(rotation_range=rotation_range,
                                     width_shift_range=width_shift_range,
                                     height_shift_range=height_shift_range,
                                     shear_range=shear_range,
                                     zoom_range=zoom_range,
                                     horizontal_flip=horizontal_flip,
                                     fill_mode=fill_mode,
                                     samplewise_center=False,
                                     featurewise_std_normalization=False,
                                     samplewise_std_normalization=False,
                                     zca_whitening=False,
                                     zca_epsilon=1e-6,
                                     brightness_range=None,
                                     channel_shift_range=0.,
                                     cval=0.,
                                     vertical_flip=False,
                                     rescale=rescale_factor,
                                     preprocessing_function=None,
                                     data_format='channels_last',
                                     validation_split=0.0,
                                     interpolation_order=1,
                                     featurewise_center=False,
                                     dtype='float32')

    if datagen is not None:
        generator = datagen.flow_from_directory(directory=source_dir,
                                                target_size=target_size,
                                                batch_size=batch_size,
                                                class_mode=class_mode)

        return generator
    else:
        return None


def get_augmented_image_generator(rescale_factor: float = 1. / 255,
                                  rotation_range: int = 40,
                                  width_shift_range: float = 0.2,
                                  height_shift_range: float = 0.2,
                                  shear_range: float = 0.2,
                                  zoom_range: float = 0.2,
                                  horizontal_flip: bool = True,
                                  fill_mode: str = 'nearest'):
    """
    Args:
        rescale_factor (float): rescale factor for pixel values
        rotation_range (int): degree range for random rotations
        width_shift_range (float): part of width for random horizontal shift
        height_shift_range (float): part of width for random vertical shift
        shear_range (float): angle in degrees for random shear
        zoom_range (float): random zoom range
        horizontal_flip (boolean): allow random horizontal flip
        fill_mode (str): how points outside the boundaries are filled("constant", "nearest", "reflect" or "wrap")

    """
    datagen = ImageDataGenerator(rotation_range=rotation_range,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range,
                                 shear_range=shear_range,
                                 zoom_range=zoom_range,
                                 horizontal_flip=horizontal_flip,
                                 fill_mode=fill_mode,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 zca_epsilon=1e-6,
                                 brightness_range=None,
                                 channel_shift_range=0.,
                                 cval=0.,
                                 vertical_flip=False,
                                 rescale=rescale_factor,
                                 preprocessing_function=None,
                                 data_format='channels_last',
                                 validation_split=0.0,
                                 interpolation_order=1,
                                 featurewise_center=False,
                                 dtype='float32')

    return datagen
