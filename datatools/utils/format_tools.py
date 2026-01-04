import os
import re
import cv2
import platform
from typing import NoReturn
from datatools.utils.misc import is_not_none


class PathFormatter:
    __slots__ = ['_dir']

    WEIGHT_DIRECTORY_STRUCTURE = ['work_dir', 'project', 'target_model', 'update']

    def __init__(self, path=None):
        assert isinstance(path, str), 'Input must be [str]'
        self._dir = path

    @property
    def windows(self) -> str:
        return self._dir.replace('/', '\\')

    @property
    def linux(self) -> str:
        return self._dir.replace('\\', '/')

    @property
    def path(self) -> str:
        return self._dir.replace('\\', os.sep).replace('/', os.sep)

    @staticmethod
    def to_window_format(path: str) -> str:
        assert isinstance(path, str), 'Input must be [str]'
        return path.replace('/', '\\')

    @staticmethod
    def to_linux_format(path: str) -> str:
        assert isinstance(path, str), 'Input must be [str]'
        return path.replace('\\', '/')

    @staticmethod
    def format(path: str) -> str:
        assert isinstance(path, str), 'Input must be [str]'
        if platform.system() != 'Windows':
            path = path.split(':')[-1]
        return path.replace('\\', os.sep).replace('/', os.sep)

    @staticmethod
    def review_dir(path: str, indicator=True) -> NoReturn:
        if indicator:
            print('Linux path:', PathFormatter.to_linux_format(path))
            print('Windows path:', PathFormatter.to_window_format(path))
        else:
            print(PathFormatter.to_linux_format(path))
            print(PathFormatter.to_window_format(path))


class SuffixFormatter:

    ENCRYPT_FORMAT = ['raw', "RAW"]
    SUPPORT_FORMAT = ['png', 'jpg', 'bmp', 'jpeg', 'PNG', 'JPG', 'JPEG', 'BMP'] + ENCRYPT_FORMAT
    _support_format = "|".join(SUPPORT_FORMAT)

    REGEX = {
        'ref': re.compile(f'_(?P<suffix>std|ref).(?P<ext>{_support_format})'),
        'mask': re.compile(f'_(?P<suffix>mask).(?P<ext>{_support_format})'),
        'comp': re.compile(f'_(?P<suffix>comp).(?P<ext>{_support_format})'),
        'gerb': re.compile(f'_(?P<suffix>gerb).(?P<ext>{_support_format})'),
        'id': re.compile(f'_(?P<suffix>id).(?P<ext>{_support_format})'),
        'support_format': re.compile(f'.(?P<ext>{_support_format})')
    }

    MAPPER = {
        'std': 'Ref',
        'ref': 'Ref',
        'mask': 'Mask',
        'gerb': 'Gerb',
        'comp': 'Comp',
        'cam': 'Cam',
        'refcomp': 'Refcomp',
        'gerbcomp': 'gerbcomp',
        'speccomp': 'Speccomp',
        'infrared': 'Infrared',
        'id': 'Id'
    }

    @staticmethod
    def format_suffix(src: str, target: str = 'ref') -> NoReturn:
        clean_root = PathFormatter.format(src)
        for root, _, files in os.walk(clean_root):
            print(f'Working on {root}')
            for file in files:
                SuffixFormatter.format_filename(root, file, target)

    @staticmethod
    def format_filename(file_root: str, file_name: str, file_type: str = 'ref') -> NoReturn:
        file_type = file_type.lower()
        # search_result = None
        if file_type in ['ref', 'Ref']:
            search_result = re.search(SuffixFormatter.REGEX['ref'], file_name)
            #TODO: modify the code to fileio adapter
            if search_result is not None:
                if search_result['ext'] in ['png'] and search_result['suffix'] in ['ref']:
                    print(f'Formated \'ref\' Case: {os.path.join(file_root, file_name)}')
                    rename = file_name.replace('_ref', '_std')
                    os.chdir(file_root)
                    os.rename(file_name, rename)
                elif search_result['ext'] in ['jpg']:
                    img = cv2.imread(os.path.join(file_root, file_name))
                    print(f'Formated \'.jpg\' Case: {os.path.join(file_root, file_name)}')
                    rename = file_name.replace('_ref', '_std').replace('.jpg', '.png')
                    cv2.imwrite(os.path.join(file_root, rename), img)
                elif search_result['ext'] in ['png'] and search_result['suffix'] in ['std']:
                    pass
                else:
                    print(f'Error Case: {os.path.join(file_root, file_name)}')

    @staticmethod
    def is_file_type(file_name: str, file_type: str) -> bool:
        assert isinstance(file_type, str), f"file_type must be str, while {type(file_type)} is given."
        file_type = file_type.lower()
        if file_type in SuffixFormatter.REGEX.keys():
            search_result = re.search(SuffixFormatter.REGEX['file_type'], file_name)
            if search_result is not None:
                return True
        return False

    @staticmethod
    def get_suffix(file):
        _, ext = os.path.splitext(file)
        suffix_pattern = re.compile('_(?P<suffix>[a-zA-Z]+)' + ext)
        res = re.search(suffix_pattern, file)
        return res['suffix'] if res is not None and res['suffix'] in SuffixFormatter.MAPPER.keys() else None

    @staticmethod
    def is_cur(file: str) -> bool:
        _, ext = os.path.splitext(file)
        suffix_case = '|'.join(SuffixFormatter.MAPPER.keys())
        suffix_pattern = re.compile(f'_(?P<suffix>{suffix_case})( ?\(\d+\))?' + ext)
        res = re.search(suffix_pattern, file)
        if res is not None or ext[1:] not in SuffixFormatter.SUPPORT_FORMAT:
            return False
        return True

    @staticmethod
    def is_attr(file: str, attr: str) -> bool:
        attr = attr.lower()
        if attr in ['cur']:
            return SuffixFormatter.is_cur(file)

        suffix_pattern = SuffixFormatter.REGEX[attr]
        res = re.search(suffix_pattern, file)
        return is_not_none(res)

    @staticmethod
    def is_encrypted_format(file: str) -> bool:
        return file.endswith(tuple(SuffixFormatter.ENCRYPT_FORMAT))

    @staticmethod
    def is_supported_format(file: str) -> bool:
        res = re.search(SuffixFormatter.REGEX['support_format'], file)
        return is_not_none(res) and file.endswith(res['ext'])

    @staticmethod
    def separate_by_suffix(work_dir, inplace=True):
        from datatools.image import SingleImage
        work_dir = PathFormatter.format(work_dir)
        files = os.listdir(work_dir)
        for file in files:
            file_path = os.path.join(work_dir, file)
            if os.path.isfile(file_path):
                target_folder = None
                if SuffixFormatter.is_cur(file_path):
                    target_folder = 'Cur'
                else:
                    suffix = SuffixFormatter.get_suffix(file_path)
                    if suffix in SuffixFormatter.MAPPER.keys():
                        target_folder = SuffixFormatter.MAPPER[suffix]
                if target_folder is not None and target_folder not in work_dir:
                    img = SingleImage(file_path)
                    img.copy_to(os.path.join(work_dir, target_folder), force=True)
                    if inplace:
                        os.remove(file_path)

    @staticmethod
    def move_by_type(root: str, log=True):
        root = PathFormatter.format(root)
        for work_dir, _, _ in os.walk(root):
            if log:
                print('> Working on:', work_dir)
            SuffixFormatter.separate_by_suffix(work_dir)



if __name__ == '__main__':
    # raw_dir = r"abc_ref.png"
    # PathFormatter.review_dir(raw_dir, indicator=False)
    print(SuffixFormatter.is_supported_format('.jpg.x'))
    print(os.path.splitext('_gerb.bmp'))
    print('.mbp'[1:])
    # test_root = r"\data\dataset2\Workshop\sunjianyao\test\7_34359738368,0已分类"
    # test_root = PathFormatter.format(test_root)
    # SuffixFormatter.move_by_type(test_root)
