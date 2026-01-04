from datatools.utils.format_tools import PathFormatter
from typing import Optional, List
from tqdm import tqdm
import shutil
import zipfile
import tarfile
import os


class ArchiveManager:
    FILE_EXT = {
        'tar': ['gz', 'tgz', 'tar'],
        'zip': ['zip', '7z'],
    }

    def __init__(self, ):
        pass

    @staticmethod
    def _get_archive(file_path: str):
        assert os.path.isfile(file_path), f'File not found: {file_path}'
        file_root, file_name = os.path.split(file_path)
        name, *mid, ext = file_name.split('.')
        if ext in ArchiveManager.FILE_EXT['tar']:
            return tarfile.open(file_path), file_root, name
        elif ext in ArchiveManager.FILE_EXT['zip']:
            return zipfile.ZipFile(file_path), file_root, name
        else:
            raise AttributeError(f'Unknown file extension: {ext}, please register in "ArchiveManager.FILE_EXT"')

    @staticmethod
    def extract(file_path: str,
                dst: Optional[str] = None,
                force: bool = True,
                individual_dir: bool = False,
                delete_after_extraction: bool = False):
        file_path = PathFormatter.format(file_path)
        print(f'> Extracting: {file_path}')
        archive, file_root, name = ArchiveManager._get_archive(file_path)
        dst = PathFormatter.format(dst) if dst is not None else file_root
        if individual_dir:
            dst = os.path.join(dst, name)
        if force and not os.path.exists(dst):
            os.makedirs(dst)
        archive.extractall(dst)
        archive.close()

        print(f'> Archive Extracted to: {dst}')
        if delete_after_extraction:
            os.remove(file_path)
            print(f'> Archive File Deleted: {file_path}')


    @staticmethod
    def extract_all(root: str,
                    exceptions: Optional[List[str]] = None):
        pass

    @staticmethod
    def make_archive(src: str,
                     name: Optional[str] = None,
                     dst: Optional[str] = None,
                     force: bool = False,
                     target: Optional[List[str]] = None,
                     exceptions: Optional[List[str]] = None):
        src = PathFormatter.format(src)
        src_root, src_name = os.path.split(src)
        name = src_name if name is None else name
        dst = src_root if dst is None else dst
        if target is None and exceptions is None:
            file_path = shutil.make_archive(os.path.join(dst, name), 'zip', src)
        else:
            file_path = os.path.join(dst, '.'.join([name, 'zip']))
            if os.path.exists(file_path):
                if force:
                    print('> Original Archive File Removed!')
                    os.remove(file_path)
                else:
                    raise FileExistsError(f'Archive File Exists: {file_path}\n'
                                          f'Use "force = True" to replace the original archive file.')
            with zipfile.ZipFile(file_path, 'a') as archive:
                dir_list = os.listdir(src) if target is None else target
                os.chdir(src)
                for tgt_dir in tqdm(dir_list):
                    if exceptions is None or tgt_dir not in exceptions:
                        for root, _, files in os.walk(tgt_dir):
                            for file in files:
                                archive.write(os.path.join(root, file), compress_type=zipfile.ZIP_DEFLATED)

        print(f'> Archive Created: {file_path}')



if __name__ == '__main__':
    ArchiveManager.make_archive(r'\data\dataset2\Workshop\sunjianyao\general\redetect_result\20230802_kshl_extract_100')
