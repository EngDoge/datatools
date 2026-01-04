import os


def exists_or_make(path: str, make_dir: bool = True) -> str:
    if not os.path.exists(path):
        if make_dir:
            os.makedirs(path, exist_ok=make_dir)
        else:
            raise FileNotFoundError(f"No such directory: '{path}'")
    return path


def check_file_exist(filename, msg_tmpl='File "{}" does not exist'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def remove_empty_directory(root_dir: str):
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in dirs:
            if not os.listdir(empty_dir := os.path.join(root, name)):
                print('remove empty dir: ', empty_dir)
                os.rmdir(empty_dir)


def is_none(obj):
    return obj is None


def is_not_none(obj):
    return obj is not None


def convert2map(src: dict):
    return {label: k for k, v in src.items() for label in v}
