import os
import math
import random
import numpy as np

from typing import Optional, List, Union, Tuple, NoReturn

from datatools.image.data import ImageData
from datatools.dataset.container import DataContainer
from datatools.utils import PathFormatter, SuffixFormatter, scandir


class DataCluster(list):
    __slots__ = ['_path', '_root', '_duplicates',
                 '_separated', '_clean_raw_label', '_hard_samples']


    def __init__(self,
                 path: str,
                 auto_load: bool = True,
                 separated: bool = True,
                 ignore_ref: bool = False,
                 ignore_gerb: bool = False,
                 hard_samples: bool = False,
                 skip_cur_check: bool = False,
                 clean_raw_label: bool = True,
                 use_single_image: bool = False,
                 strict_inspection: bool = False,
                 duplicates: Optional[int] = None,
                 required: Union[str, List[str], None] = None,
                 require_all: bool = True,
                 prohibit_all: bool = True,
                 prohibited: Union[str, List[str], None] = None,
                 **kwargs):
        """ List of ImageData.

        Args:
            path (str): Path to the data cluster.
            auto_load (bool): Whether to load the data automatically.
            separated (bool): Whether the data is separated into different folders.
            ignore_ref (bool): Whether to ignore the reference image.
            ignore_gerb (bool): Whether to ignore the gerber image.
            hard_samples (bool): Whether the data is hard samples.
            skip_cur_check (bool): Whether to skip the cur image check.
            clean_raw_label (bool): Whether to clean the raw label.
            use_single_image (bool): Whether to use single image.
            strict_inspection (bool): Whether to use strict inspection.
            duplicates (int): Number of duplicates.
            required (str, List[str], optional): Required attributes.
            require_all (bool): Whether to require all attributes.
        """

        super(DataCluster, self).__init__()

        self._path = PathFormatter.format(path)

        self._separated = separated
        self._hard_samples = hard_samples
        self._clean_raw_label = clean_raw_label
        self._duplicates = 1 if duplicates is None else duplicates

        if auto_load:
            self.load(ignore_ref=ignore_ref,
                      ignore_gerb=ignore_gerb,
                      skip_cur_check=skip_cur_check,
                      use_single_image=use_single_image,
                      strict_inspection=strict_inspection,
                      required=required,
                      require_all=require_all,
                      prohibited=prohibited,
                      prohibit_all=prohibit_all
                      )


    @property
    def path(self) -> str:
        return self._path

    @property
    def raw_label(self) -> str:
        return os.path.basename(self._path)

    @property
    def label(self) -> str:
        return DataCluster.clean_label(self.raw_label) if self._clean_raw_label else self.raw_label

    @property
    def data(self) -> DataContainer:
        data = DataContainer(allow_duplicates=(True if self._duplicates > 1 else False))
        data[self.label] = self.copy() * self._duplicates
        return data

    @property
    def raw_data(self) -> List:
        return self.copy()

    @classmethod
    def from_path(cls,
                  path: str,
                  separated: bool = True,
                  ignore_ref: bool = True,
                  ignore_gerb: bool = True,
                  hard_samples: bool = False,
                  skip_cur_check: bool = False,
                  clean_raw_label: bool = True,
                  use_single_image: bool = False,
                  prohibit_all: bool = True,
                  prohibited: Union[str, List[str], None] = None,
                  strict_inspection: bool = False,
                  duplicates: Optional[int] = None,
                  required: Union[str, List[str], None] = None,
                  require_all: bool = True,
                  **kwargs):

        return cls(path=path,
                   auto_load=True,
                   separated=separated,
                   ignore_ref=ignore_ref,
                   ignore_gerb=ignore_gerb,
                   hard_samples=hard_samples,
                   skip_cur_check=skip_cur_check,
                   clean_raw_label=clean_raw_label,
                   use_single_image=use_single_image,
                   strict_inspection=strict_inspection,
                   prohibit_all=prohibit_all,
                   prohibited=prohibited,
                   duplicates=duplicates,
                   required=required,
                   require_all=require_all,
                   **kwargs)

    @staticmethod
    def clean_label(label: str) -> str:
        assert type(label) == str, 'Raw label must be Str!'
        return label.split('_')[0]

    @staticmethod
    def _separated_file_name_check(file_path: str,
                                   skip: bool) -> bool:
        is_cur = SuffixFormatter.is_cur(file_path)
        if not is_cur:
            if SuffixFormatter.is_supported_format(file_path):
                if not skip:
                    print(f'Unidentified Cur Name: {file_path}\n> Use "skip_cur_check = True" to skip the name check')
            else:
                if skip:
                    print(f'File identified as Cur image due to skip cur check: {file_path}')
                else:
                    return False
        return is_cur or skip

    @staticmethod
    def _nonseparated_file_name_check(file_path: str,
                                      ignore_ref: bool,
                                      ignore_gerb: bool) -> bool:
        return (SuffixFormatter.is_cur(file_path) or
                DataCluster._include_or_ignore_attr(file_path=file_path,
                                                    attr='ref',
                                                    ignore=ignore_ref) or
                DataCluster._include_or_ignore_attr(file_path=file_path,
                                                    attr='gerb',
                                                    ignore=ignore_gerb))

    @staticmethod
    def _include_or_ignore_attr(file_path: str,
                                attr: str,
                                ignore: bool) -> bool:
        is_attr = SuffixFormatter.is_attr(file_path, attr)
        if is_attr and ignore:
            print(f'Ignored "{attr}" file: {file_path}\n> Use "ignore_{attr} = False" to include the file')
        return not ignore and is_attr

    def file_name_check(self,
                        file_path: str,
                        ignore_ref: bool,
                        ignore_gerb: bool,
                        skip_cur_check: bool) -> bool:
        return ((self._separated and DataCluster._separated_file_name_check(file_path=file_path,
                                                                            skip=skip_cur_check)) or
                (not self._separated and DataCluster._nonseparated_file_name_check(file_path=file_path,
                                                                                   ignore_ref=ignore_ref,
                                                                                   ignore_gerb=ignore_gerb)))


    def is_empty(self) -> bool:
        if not self:
            print(f'Cluster is empty: {self.path}')
        return not self

    def load(self,
             ignore_ref: bool = False,
             ignore_gerb: bool = False,
             skip_cur_check: bool = False,
             use_single_image: bool = False,
             required: Union[str, List[str], None] = None,
             require_all: bool = True,
             strict_inspection: bool = False,
             prohibit_all: bool = True,
             prohibited: Union[str, List[str], None] = None,
             **kwargs) -> NoReturn:

        cur_path = self.path if not self._separated else os.path.join(self.path, 'Cur')
        data = [ImageData(file_path=os.path.join(cur_path, img_path),
                          use_single_image=use_single_image,
                          separated=self._separated,
                          strict_inspection=strict_inspection,
                          hard_sample=self._hard_samples)
                for img_path in os.listdir(cur_path)
                if self.file_name_check(file_path=img_path,
                                        ignore_ref=ignore_ref,
                                        ignore_gerb=ignore_gerb,
                                        skip_cur_check=skip_cur_check)]

        if required is not None:
            if isinstance(required, str):
                required = [required]
            elif isinstance(required, list):
                required = [attr for attr in required if isinstance(attr, str)]
            else:
                raise f'required must be str or List[str], {type(required)} is given'
            if require_all:
                for requirement in required:
                    data = [img for img in data if getattr(img, requirement) is not None]
            else:
                data = [img for img in data for requirement in required if getattr(img, requirement) is not None]

        if prohibited is not None:
            if isinstance(prohibited, str):
                prohibited = [prohibited]
            elif isinstance(prohibited, list):
                prohibited = [attr for attr in prohibited if isinstance(attr, str)]
            else:
                raise f'required must be str or List[str], {type(required)} is given'
            if prohibit_all:
                for prohibition in prohibited:
                    data = [img for img in data if getattr(img, prohibition) is None]
            else:
                data = [img for img in data for prohibition in prohibited if getattr(img, prohibition) is None]

        self.extend(data)

    def split(self, split_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List, List]:
        split_ratio = 1 if self._hard_samples else split_ratio
        offset = math.ceil(len(self) * split_ratio)
        data = self.copy()
        random.seed(random_seed)
        random.shuffle(data)
        return data[:offset] * self._duplicates, data[offset:]

    @classmethod
    def validate_datacluster(cls,
                             path: str,
                             auto_load: bool = True,
                             separated: bool = True,
                             ignore_ref: bool = False,
                             ignore_gerb: bool = False,
                             hard_samples: bool = False,
                             skip_cur_check: bool = False,
                             clean_raw_label: bool = True,
                             use_single_image: bool = False,
                             strict_inspection: bool = False,
                             duplicates: Optional[int] = None,
                             required: Union[str, List[str], None] = None,
                             require_all: bool = True,
                             **kwargs):
        try:
            if duplicates is None:
                duplicates = 1
            dc = cls(path=path,
                     auto_load=auto_load,
                     separated=separated,
                     ignore_ref=ignore_ref,
                     ignore_gerb=ignore_gerb,
                     require_all=require_all,
                     clean_labels=clean_raw_label,
                     use_single_img=use_single_image,
                     skip_cur_check=skip_cur_check,
                     strict_inspection=strict_inspection,
                     duplicates=duplicates,
                     required=required,
                     hard_samples=hard_samples, )
        except FileNotFoundError as e:
            print(e)
            raise IndexError(f'Invalid file structure: {path}')

        path = PathFormatter.format(path)
        scanned = DataContainer()
        exclude_suffix = tuple('_' + suffix for suffix in SuffixFormatter.MAPPER.keys()
                               if suffix not in ['ref', 'std', 'gerb'])

        if separated:
            ignore_ref = True
            ignore_gerb = True

        if ignore_ref:
            exclude_suffix += ('_ref', '_std')
        if ignore_gerb:
            exclude_suffix += ('_gerb',)
        for ret in scandir(path,
                           recursive=True,
                           exclude_suffix=exclude_suffix,
                           with_extension=tuple('.' + ext for ext in SuffixFormatter.SUPPORT_FORMAT)):
            img = ImageData(ret, separated='Cur' in ret)
            scanned[img.label].append(img)
        if scanned.total_num != dc.data.total_num:
            raise IndexError(
                f'Invalid file structure: {path}, scanned: {scanned.total_num}, loaded: {dc.data.total_num}')

        return dc


if __name__ == '__main__':
    # t = r'\data\data_cold\Workshop\general_PI\osphj\other_ink\seg_withref\NG\kswus_20230516_ng_cls_mask_cleaned\001'
    # test_label = DataCluster.from_path(t, duplicate=2)
    x = DataCluster.validate_datacluster(r'\data\dataset2\Example\compseg\hzsh_20230908\semislot_bg', separated=False)
    print(x)
    # test_label.data.size
    # train, val = test_label.split(0.8)
    # print(len(train), len(val))


