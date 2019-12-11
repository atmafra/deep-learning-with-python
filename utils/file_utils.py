import os.path

from slugify import slugify


def str_to_filename(filename_str: str, extension: str = ''):
    return slugify(filename_str) + extension


def root_name(filename: str):
    basename = os.path.basename(filename)
    basename_split = os.path.splitext(basename)
    root = None
    if len(basename_split) == 2:
        root = basename_split[0]
    return root
