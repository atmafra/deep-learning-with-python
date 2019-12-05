from keras_preprocessing.image import ImageDataGenerator


def get_image_generator(rescale_factor: float,
                        source_dir: str,
                        target_size: tuple,
                        batch_size: int,
                        class_mode: str):
    """
    """
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = \
        datagen.flow_from_directory(directory=source_dir,
                                    target_size=target_size,
                                    batch_size=batch_size,
                                    class_mode=class_mode)

    return generator
