import os
from typing import Callable


def rename_folder(root: str, fn: Callable):
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if os.path.isdir(folder_path):
            rename = fn(folder)
            os.rename(folder_path, os.path.join(root, rename))



if __name__ == '__main__':
    tgt = '/data/dataset2/Workshop/sunjianyao/general_outer/task_export/ntsn_20230719'

    def swap(folder):
        c = folder.split('-')
        c.reverse()
        return '_'.join(c)


    rename_folder(tgt, swap)



