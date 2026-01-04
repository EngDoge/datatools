import os
import concurrent.futures
from functools import wraps
from typing import Optional, List, Union, Dict, Tuple

from datatools.image.data import ImageData
from datatools.dataset.cluster import DataCluster
from datatools.dataset.container import DataContainer
from datatools.utils import PathFormatter, SuffixFormatter, scandir


class DataPatch(object):
    __slots__ = ['_root', '_data', '_modified', '_raw_data',
                 '_separated', '_duplicates', '_exceptions', '_use_single_img',
                 '_hard_samples', '_ignore_ref', '_ignore_gerb', '_exception_by_class',
                 '_num_workers', '_required', '_require_all', '_prohibited', '_prohibit_all',
                 '_skip_cur_check', '_strict_inspection']

    def __init__(self,
                 path: str,
                 separated: bool = True,
                 ignore_ref: bool = True,
                 ignore_gerb: bool = True,
                 require_all: bool = True,
                 prohibit_all: bool = True,
                 clean_labels: bool = True,
                 sort_raw_data: bool = False,
                 use_single_img: bool = False,
                 skip_cur_check: bool = False,
                 strict_inspection: bool = False,
                 num_workers: Optional[int] = 8,
                 exception_by_class: bool = False,
                 exceptions: Optional[List] = None,
                 duplicates: Union[Dict, int, None] = None,
                 required: Union[str, List[str], None] = None,
                 prohibited: Union[str, List[str], None] = None,
                 hard_samples: Union[List, bool, None] = False,
                 ):

        if duplicates is not None:
            assert isinstance(duplicates, Dict) or isinstance(duplicates, int), \
                "duplicates must be a Dictionary or Int!"
        if hard_samples is not None:
            assert isinstance(hard_samples, List) or isinstance(hard_samples, bool), \
                "hard_samples must be a Dictionary or Boolean!"

        self._raw_data = []
        self._modified = True
        self._required = required

        self._separated = separated
        self._prohibited = prohibited
        self._ignore_ref = ignore_ref
        self._ignore_gerb = ignore_gerb
        self._require_all = require_all
        self._num_workers = num_workers
        self._prohibit_all = prohibit_all
        self._hard_samples = hard_samples
        self._skip_cur_check = skip_cur_check
        self._use_single_img = use_single_img
        self._strict_inspection = strict_inspection
        self._exception_by_class = exception_by_class

        self._duplicates = self._set_default_dict(duplicates, int, 1)

        self._data = DataContainer(allow_duplicates=False if duplicates is None else True)

        if exceptions is None:
            exceptions = []
        else:
            assert isinstance(exceptions, list)
        self._exceptions = exceptions

        self._root = PathFormatter.format(path)
        if os.path.exists(self._root):
            self.load(clean_labels=clean_labels, sort_raw_data=sort_raw_data)
        else:
            raise FileNotFoundError(f'Path does not exist: {self._root}')

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __str__(self) -> str:
        return '{' + ', '.join('\'{}\':{}'.format(cluster.raw_label, len(cluster)) for cluster in self.raw_data) + '}'

    def __repr__(self) -> str:
        return f'DataPatch: {self._root}'

    def __getitem__(self, index):
        return self.data[index]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    @property
    def root(self):
        return self._root

    @property
    def data(self) -> DataContainer:
        if not (self._data.is_empty() or self._modified):
            return self._data
        self._modified = False
        self._data.clear()
        for cluster in self.raw_data:
            self._data += cluster.data
        if self._data.is_empty():
            print(f'{self._root} is empty!')
        return self._data

    @property
    def raw_cluster_data(self) -> DataContainer:
        result = DataContainer(allow_duplicates=False)
        for cluster in self.raw_data:
            result[cluster.label] += cluster.raw_data
        return result

    @property
    def raw_data(self) -> List[DataCluster]:
        return self._raw_data

    @property
    def size(self) -> Dict:
        return {k: len(v) for k, v in self.items()}

    @property
    def duplicates(self):
        return self._duplicates

    def modify_data(fn):
        @wraps(fn)
        def wrapped_fn(self, *args, **kwargs):
            self._modified = True
            return fn(self, *args, **kwargs)
        return wrapped_fn

    @modify_data
    def load(self,
             clean_labels: bool = True,
             sort_raw_data: bool = False) -> None:
        clusters = [os.path.join(self._root, name)
                    for name in os.listdir(self._root)
                    if not (name in self._exceptions or os.path.isfile(os.path.join(self._root, name)) or
                            (self._exception_by_class and DataCluster.clean_label(name) in self._exceptions))]
        if self._num_workers is not None and self._num_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_workers) as exe:
                cluster_list = []
                for cluster in clusters:
                    name = os.path.basename(cluster)
                    name = DataCluster.clean_label(name) if clean_labels else name
                    hard_sample = self._hard_samples if isinstance(self._hard_samples, bool) \
                        else True if self._hard_samples is not None and name in self._hard_samples else False
                    multi = self._duplicates[name] if name in self._duplicates else self._duplicates["all"]
                    cluster_list.append(exe.submit(DataCluster.from_path,
                                                   path=cluster,
                                                   separated=self._separated,
                                                   ignore_ref=self._ignore_ref,
                                                   ignore_gerb=self._ignore_gerb,
                                                   skip_cur_check=self._skip_cur_check,
                                                   clean_raw_label=clean_labels,
                                                   use_single_image=self._use_single_img,
                                                   strict_inspection=self._strict_inspection,
                                                   required=self._required,
                                                   require_all=self._require_all,
                                                   prohibited=self._prohibited,
                                                   prohibit_all=self._prohibit_all,
                                                   hard_samples=hard_sample,
                                                   duplicates=multi,))
                results = [future.result() for future in concurrent.futures.as_completed(cluster_list)]
        else:
            results = []
            for cluster in clusters:
                name = os.path.basename(cluster)
                name = DataCluster.clean_label(name) if clean_labels else name
                hard_sample = self._hard_samples if isinstance(self._hard_samples, bool) \
                    else True if self._hard_samples is not None and name in self._hard_samples else False
                multi = self._duplicates[name] if name in self._duplicates else self._duplicates["all"]
                results.append(DataCluster.from_path(path=cluster,
                                                     separated=self._separated,
                                                     ignore_ref=self._ignore_ref,
                                                     ignore_gerb=self._ignore_gerb,
                                                     skip_cur_check=self._skip_cur_check,
                                                     clean_raw_label=clean_labels,
                                                     use_single_image=self._use_single_img,
                                                     strict_inspection=self._strict_inspection,
                                                     required=self._required,
                                                     require_all=self._require_all,
                                                     hard_samples=hard_sample,
                                                     duplicates=multi,))

        self._raw_data = [cluster for cluster in results if not cluster.is_empty()]
        if sort_raw_data:
            self._raw_data.sort(key=lambda x: len(x.data), reverse=True)

    def split(self,
              split_ratio: float = 0.8,
              merge_labels: Optional[Dict] = None,
              top_labels: Optional[List] = None,
              random_seed: int = 42) -> Tuple[DataContainer, DataContainer]:

        allow_duplicates = self._data.allow_duplicates
        train = DataContainer(allow_duplicates=allow_duplicates)
        val = DataContainer(allow_duplicates=allow_duplicates)
        for cluster in self.raw_data:
            cluster_train, cluster_val = cluster.split(split_ratio=split_ratio, random_seed=random_seed)
            label = cluster.label
            label = merge_labels[label] if merge_labels is not None and label in merge_labels.keys() else label
            if top_labels is not None and label not in top_labels:
                continue
            train[label] += cluster_train
            val[label] += cluster_val
        return train, val

    def is_empty(self) -> bool:
        return not self._raw_data

    @classmethod
    def validate_datapatch(cls,
                           path: str,
                           separated: bool = True,
                           ignore_ref: bool = True,
                           ignore_gerb: bool = True,
                           require_all: bool = True,
                           clean_labels: bool = True,
                           sort_raw_data: bool = False,
                           use_single_img: bool = False,
                           skip_cur_check: bool = False,
                           strict_inspection: bool = False,
                           num_workers: Optional[int] = 8,
                           exception_by_class: bool = False,
                           exceptions: Optional[List] = None,
                           duplicates: Union[Dict, int, None] = None,
                           required: Union[str, List[str], None] = None,
                           hard_samples: Union[List, bool, None] = False,
                           ):
        try:
            if duplicates is None:
                duplicates = {'all': 1}
            dp = cls(path=path,
                     separated=separated,
                     ignore_ref=ignore_ref,
                     ignore_gerb=ignore_gerb,
                     require_all=require_all,
                     clean_labels=clean_labels,
                     sort_raw_data=sort_raw_data,
                     use_single_img=use_single_img,
                     skip_cur_check=skip_cur_check,
                     strict_inspection=strict_inspection,
                     num_workers=num_workers,
                     exception_by_class=exception_by_class,
                     exceptions=exceptions,
                     duplicates=duplicates,
                     required=required,
                     hard_samples=hard_samples,)
        except FileNotFoundError as e:
            print(e)
            raise IndexError(f'Invalid file structure: {path}')

        path = PathFormatter.format(path)
        scanned = DataContainer()
        exclude_suffix = tuple('_'+suffix for suffix in SuffixFormatter.MAPPER.keys()
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
                           with_extension=tuple('.'+ext for ext in SuffixFormatter.SUPPORT_FORMAT)):
            img = ImageData(ret, separated='Cur' in ret)
            scanned[img.label].append(img)
        if scanned.total_num != dp.data.total_num:
            raise IndexError(f'Invalid file structure: {path}, scanned: {scanned.total_num}, loaded: {dp.data.total_num}')

        return dp


    @staticmethod
    def _set_default_dict(config,
                          required_type,
                          default_value) -> Dict:
        assert isinstance(default_value, required_type), \
            f'Required Type [{required_type}], {type(default_value)} is given.'
        if isinstance(config, required_type):
            return dict(all=config)
        config = dict(all=default_value) if config is None else config
        if "all" not in config:
            config["all"] = default_value
        return config


if __name__ == '__main__':
    # data_path = r'\data\dataset2\Workshop\sunjianyao\test\Removal'
    patch = DataPatch(r'\data\dataset2\Example\compseg\hzsh_20230908\semislot_bg')
    # print(patch['CD000'])
