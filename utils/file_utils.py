import os.path

from slugify import slugify


def str_to_filename(input_string: str):
    """Converts a generic string to a valid filename

    Args:
        input_string (str): string to be converted to filename
    """
    return slugify(input_string)


def root_name(filename: str):
    """Returns the root of a filename (name without extension)

    Args:
        filename: file name to extract the root

    """
    basename = os.path.basename(filename)
    basename_split = os.path.splitext(basename)
    root = None
    if len(basename_split) == 2:
        root = basename_split[0]
    return root
