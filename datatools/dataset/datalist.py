import os
import json
import numpy as np
import pandas as pd
from functools import wraps
from inspect import isfunction
from collections import defaultdict
from typing import Optional, Dict, List, Union, Tuple, Callable, NoReturn

from datatools.dataset.datapatch import DataPatch
from datatools.dataset.container import DataContainer
from datatools.image.data import ImageData, SingleImage
from datatools.analyzer.utils import CUR_COL, CLUSTER_COL, MASK_COL, GERBER_COL
from datatools.utils import PathFormatter, ActionRecorder, is_not_none, is_none, convert2map



class DataListGenerator(object):
    __slots__ = ['save_path', 'sep', 'target_model', 'include_null_path', 'label_exceptions', 'positive_labels',
                 'top_labels', 'merge_labels', '_raw_datasets', '_top_label_weights', 'exception_by_class',
                 'clean_labels', 'allow_duplicates', 'dataset_exceptions', 'removal', 'required', 'allow_empty',
                 'ignore_ref', 'ignore_gerb', 'require_all', 'separated', 'skip_cur_check', '_record_actions', '_recorder']

    CONFIG = {
        'target_model_map': {
            'cls': 'defect_cls',
            'cls_outer': 'defect_cls',
            'seg': 'seg_withref',
            'seg_withref': 'seg_withref',
            'seg_noref': 'seg_noref',
            'noref': 'seg_noref',
            'compseg': 'compseg',
            'comp_seg': 'compseg',
            'zone_cls': 'zone_cls',
            'sk': 'sk_compseg'
        },
        'default_gen_files': {
            'train': 'train',
            'trainVal': ('train', 'val'),
            'val': 'val',
            'test': 'test',
            'general': 'dataset_info',
            'actions': 'actions'
        }
    }

    def __init__(self,
                 save_path: Optional[str] = None,
                 sep: str = '|',
                 separated: bool = True,
                 target_model: str = 'cls',
                 clean_labels: bool = True,
                 skip_cur_check: bool = False,
                 allow_duplicates: bool = False,
                 include_null_path: bool = True,
                 merge_labels: Optional[Dict] = None,
                 top_labels: Optional[List[str]] = None,
                 dataset_exceptions: Optional[List[str]] = None,
                 label_exceptions: Optional[List[str]] = None,
                 positive_labels: Optional[List[str]] = None,
                 exception_by_class: bool = False,
                 required: Union[str, List[str], None] = None,
                 allow_empty: Optional[bool] = None,
                 ignore_ref: bool = True,
                 ignore_gerb: bool = True,
                 require_all: bool = True,
                 **kwargs):

        if save_path is not None:
            self.save_path = PathFormatter.format(save_path)

        self.set_top_labels(top_labels)
        self.set_merge_labels(merge_labels)
        self.set_label_exceptions(label_exceptions)
        self.set_dataset_exceptions(dataset_exceptions)

        self._raw_datasets = []

        duplicates = 1 if allow_duplicates else None

        self.sep = sep
        self.removal = None
        self.required = required
        self.separated = separated
        self.ignore_ref = ignore_ref
        self.ignore_gerb = ignore_gerb
        self.require_all = require_all
        self.clean_labels = clean_labels
        self.target_model = target_model
        self.allow_duplicates = duplicates
        self.skip_cur_check = skip_cur_check
        self.positive_labels = positive_labels
        self.include_null_path = include_null_path
        self.exception_by_class = exception_by_class
        self.allow_empty = allow_empty if allow_empty is not None else True if required is not None else False

        self._record_actions = True
        self._recorder = ActionRecorder(DataListGenerator)

        self._top_label_weights = None

    @property
    def raw_datasets(self) -> List[DataPatch]:
        return self._raw_datasets

    @property
    def merged_dataset(self) -> DataContainer:
        train, _ = self._split(split_ratio=1)
        return train

    @property
    def dataset(self) -> DataContainer:
        dataset = DataContainer(allow_duplicates=True)
        for data_patch in self.raw_datasets:
            dataset += data_patch.raw_cluster_data
        return dataset

    def record(fn):
        @wraps(fn)
        def wrapped_fn(self, *args, **kwargs):
            if self._record_actions:
                self._recorder.actions[fn.__name__].args.append(args)
                self._recorder.actions[fn.__name__].kwargs.append(kwargs)
            return fn(self, *args, **kwargs)

        return wrapped_fn

    # @record
    # def __setattr__(self, key, value):
    #     setattr(self, key, value)

    @classmethod
    def set_project_update(cls,
                           work_dir: Optional[str] = None,
                           project: Optional[str] = None,
                           update: Optional[str] = None,
                           surface: Optional[str] = None,
                           target_model: Optional[str] = None,
                           config: Optional[Dict] = None,
                           force: bool = True):

        if config is not None:
            save_path = DataListGenerator.parse_config(config)
        else:
            work_dir = PathFormatter.format(work_dir)
            if surface is not None:
                save_path = os.path.join(work_dir, project, surface, target_model, update, 'dataset')
            else:
                save_path = os.path.join(work_dir, project, target_model, update, 'dataset')

        if force and not os.path.exists(save_path):
            os.makedirs(save_path)

        if config is None:
            return cls(save_path=work_dir)

        return cls(save_path=save_path, **config)

    @record
    def load_dataset(self,
                     folder_path: str,
                     num_workers: int = 8,
                     force_load: bool = False,
                     separated: Optional[bool] = None,
                     exceptions: Optional[List] = None,
                     ignore_ref: Optional[bool] = None,
                     ignore_gerb: Optional[bool] = None,
                     require_all: Optional[bool] = None,
                     clean_labels: Optional[bool] = None,
                     skip_cur_check: Optional[bool] = None,
                     required: Union[List, str, None] = None,
                     exception_by_class: Optional[bool] = None,
                     duplicates: Union[Dict, int, None] = None,
                     hard_samples: Union[List, bool, None] = None,):

        assert (not self.allow_duplicates and duplicates is None) or self.allow_duplicates, \
            'Duplication of datasets is not allowed!'

        folder_path = PathFormatter.format(folder_path)

        separated = self.separated if separated is None else separated
        required_attr = self.required if required is None else required
        ignore_ref = self.ignore_ref if ignore_ref is None else ignore_ref
        ignore_gerb = self.ignore_gerb if ignore_gerb is None else ignore_gerb
        require_all = self.require_all if require_all is None else require_all
        clean_labels = self.clean_labels if clean_labels is None else clean_labels
        dataset_exception = self.label_exceptions if exceptions is None else exceptions
        skip_cur_check = self.skip_cur_check if skip_cur_check is None else skip_cur_check
        exception_by_class = self.exception_by_class if exception_by_class is None else exception_by_class

        if force_load:
            dataset_exception = None if exceptions is None else exceptions

        dataset = DataPatch(path=folder_path,
                            sort_raw_data=False,
                            separated=separated,
                            duplicates=duplicates,
                            ignore_ref=ignore_ref,
                            ignore_gerb=ignore_gerb,
                            skip_cur_check=skip_cur_check,
                            num_workers=num_workers,
                            hard_samples=hard_samples,
                            required=required_attr,
                            require_all=require_all,
                            clean_labels=clean_labels,
                            exceptions=dataset_exception,
                            exception_by_class=exception_by_class,)

        assert not dataset.is_empty() or self.allow_empty, 'Empty Dataset: ' + folder_path
        self._raw_datasets.append(dataset)

    def load_from(self,
                  root_dir: str,
                  num_workders=8,
                  force_load: bool = False,
                  search_tree: bool = False,
                  required: Optional[Dict] = None,
                  separated: Optional[bool] = None,
                  ignore_ref: Optional[bool] = None,
                  ignore_gerb: Optional[bool] = None,
                  skip_cur_check: Optional[bool] = None,
                  clean_labels: Optional[bool] = None,
                  require_all: Optional[bool] = None,
                  exceptions: Optional[List] = None,
                  duplicates: Union[Dict, int, None] = None,
                  hard_samples: Optional[Dict] = None,
                  exception_by_class: Optional[Dict] = None,) -> NoReturn:

        root_dir = PathFormatter.format(root_dir)

        datasets = self.find_datasets(root_dir) if search_tree \
            else [os.path.join(root_dir, dataset) for dataset in os.listdir(root_dir)]

        exceptions = self.dataset_exceptions if exceptions is None else exceptions
        for dataset in datasets:
            dataset_name = os.path.basename(dataset)

            if (not os.path.isdir(dataset)) or (exceptions is not None and dataset_name in exceptions):
                continue

            if isinstance(duplicates, int):
                multi = duplicates
            elif isinstance(duplicates, dict):
                multi = duplicates.get(dataset_name, None)
            else:
                multi = None

            required_attr = required[dataset_name] \
                if required is not None and dataset_name in required \
                else None

            use_split = hard_samples[dataset_name] \
                if hard_samples is not None and dataset_name in hard_samples \
                else False

            dataset_exception_by_class = True \
                if exception_by_class is not None and dataset_name in exception_by_class \
                else self.exception_by_class

            self.load_dataset(dataset,
                              num_workers=num_workders,
                              force_load=force_load,
                              separated=separated,
                              ignore_ref=ignore_ref,
                              ignore_gerb=ignore_gerb,
                              skip_cur_check=skip_cur_check,
                              require_all=require_all,
                              clean_labels=clean_labels,
                              duplicates=multi,
                              hard_samples=use_split,
                              required=required_attr,
                              exceptions=self.label_exceptions,
                              exception_by_class=dataset_exception_by_class,)

    def get_top_label_weights(self,
                              dataset: Optional[DataContainer] = None,
                              review: bool = True) -> defaultdict:

        dataset = self.merged_dataset if dataset is None else dataset
        nums = np.array(list(dataset.size.values()))

        top_label_weights = np.apply_along_axis(lambda x: pow(nums.sum() / x, 0.5), 0, nums)
        self._top_label_weights = defaultdict(float, zip(dataset.size.keys(), top_label_weights))

        if review:
            print('{' + ', '.join('\'{}\':{}'.format(k, v) for k, v in self._top_label_weights.items()) + '}')
        return self._top_label_weights

    def from_json(self, file_path, review=False):
        # TODO: load directly from dataset_info.json
        pass

    def to_json(self,
                save_path: str,
                suffix: Optional[str] = None,
                review: bool = True,
                dataset: Optional[DataContainer] = None) -> NoReturn:
        if self.raw_datasets:
            suffix = self.format_suffix(suffix)
            save_path = PathFormatter.format(save_path)
            json_path = os.path.join(save_path, self.CONFIG['default_gen_files']['general'] + suffix + '.json')
            data_patches = {data_patch.root: data_patch.duplicates for data_patch in self.raw_datasets}
            top_label_weights = self._top_label_weights \
                if self._top_label_weights is not None \
                else self.get_top_label_weights(review=False)
            info = {'datasets': data_patches,
                    'cls_mapper': list(top_label_weights.keys()),
                    'top_label_weights': top_label_weights,
                    'merge_labels': self.merge_labels,
                    }
            if dataset is not None:
                info['train_num'] = dataset.size
            with open(json_path, 'w') as f:
                json.dump(info, f, indent=4)

            if review:
                print('Exported: %s' % json_path)
        else:
            print('Empty Dataset')

    @staticmethod
    def format_str(input_var):
        if isinstance(input_var, str):
            input_var = '"' + input_var + '"'
        return input_var

    def save_action_record(self,
                           save_path: str,
                           suffix: Optional[str] = None,
                           review: bool = True) -> NoReturn:
        suffix = self.format_suffix(suffix)
        save_path = PathFormatter.format(save_path)
        save_file = os.path.join(save_path, self.CONFIG['default_gen_files']['actions'] + suffix + '.txt')
        output = [f'gen = {type(self).__name__}()\n']
        output += [f'gen.{attr} = {DataListGenerator.format_str(getattr(self, attr))}\n'
                   for attr in self.__slots__ if not attr.startswith('_')]
        output += [action.replace(type(self).__name__, 'gen')+'\n' for action in self._recorder.__str__().split('\n')]
        with open(save_file, 'w') as f:
            f.writelines(output)
        if review:
            print('Exported: %s' % save_file)

    @record
    def generate_datalist(self,
                          suffix: Optional[str] = None,
                          gen_mode: str = 'trainVal',
                          save_path: Optional[str] = None,
                          split_ratio: float = 0.8,
                          random_seed: int = 42,
                          review: bool = True,
                          limit_num: Optional[Dict] = None,
                          limit_num_ratio: Optional[Dict] = None,
                          num_condition: Optional[str] = None,
                          force_split_in_val: bool = False,
                          export_datset_info: bool = True,
                          addup_file: Optional[str] = None) -> NoReturn:
        assert not (is_not_none(limit_num) and is_not_none(limit_num_ratio)), 'limit_num and limit_num_ratio conflict!'
        if addup_file:
            addup_file = PathFormatter.format(addup_file)
        if save_path is not None:
            self.save_path = save_path

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            if gen_mode in ['trainVal']:
                train, val = self._split(split_ratio=split_ratio, random_seed=random_seed)
                if is_not_none(num_condition):
                    train, keys = train.num_condition(condition=num_condition, get_keys=True)
                    print(f'Removed Cluster(s): {keys}')
                    for key in keys:
                        val.pop(key)
                if is_not_none(limit_num):
                    train = train.limit_num(limit_num)
                elif is_not_none(limit_num_ratio):
                    train = train.limit_num_ratio(limit_num_ratio)
                if force_split_in_val:
                    val.pop_empty_keys()
                    for key in set(train.keys()) - set(val.keys()):
                        val[key] = list(set(train[key]))
                train_file_name, val_file_name = self.CONFIG['default_gen_files'][gen_mode]
                self.__write_datalist(dataset=train, name=train_file_name, suffix=suffix, review=review, addup=addup_file)
                self.__write_datalist(dataset=val, name=val_file_name, suffix=suffix, review=review)
            else:
                train, _ = self._split(split_ratio=1)
                if num_condition is not None:
                    train, keys = train.num_condition(condition=num_condition, get_keys=True)
                if is_not_none(limit_num):
                    train = train.limit_num(limit_num)
                elif is_not_none(limit_num_ratio):
                    train = train.limit_num_ratio(limit_num_ratio)
                file_name = self.CONFIG['default_gen_files'][gen_mode]
                self.__write_datalist(dataset=train, name=file_name, suffix=suffix, review=review, addup=addup_file)

            _ = self.get_top_label_weights(dataset=train)
            if export_datset_info:
                self.to_json(save_path=self.save_path, suffix=suffix, review=review, dataset=train)
                self.save_action_record(save_path=self.save_path, suffix=suffix, review=review)

        else:
            raise FileNotFoundError('Save Path is not given!')

    @record
    def remove(self, targets: Dict):
        assert isinstance(targets, dict)
        print('Current function is not supported')
        self.removal = targets
        # TODO: remove not supported

    def _split(self, split_ratio: float = 0.8, random_seed: int = 42) -> Tuple[DataContainer, DataContainer]:
        train = DataContainer(allow_duplicates=self.allow_duplicates)
        val = DataContainer(allow_duplicates=self.allow_duplicates)

        for data_patch in self.raw_datasets:
            temp_train, temp_val = data_patch.split(split_ratio=split_ratio,
                                                    merge_labels=self.merge_labels,
                                                    top_labels=self.top_labels,
                                                    random_seed=random_seed)
            train += temp_train
            val += temp_val

        return train, val

    def get_binary_label(self, label: str) -> str:
        return 'OK' if label in self.positive_labels else 'NG'

    def __write_datalist(self,
                         dataset: DataContainer,
                         name: str = 'train',
                         suffix: Optional[str] = None,
                         review: bool = True,
                         addup: Optional[str] = None) -> NoReturn:
        suffix = self.format_suffix(suffix)
        output_file = os.path.join(self.save_path, name + suffix + '.txt')
        with open(output_file, 'w') as f:
            for label, data in dataset.items():
                for img_data in data:
                    mask = self.get_or_null(img_data, 'mask', include_null_path=self.include_null_path)
                    cur = img_data.cur
                    if self.target_model in ['cls', 'defect_cls']:
                        write_list = [cur, label, mask]
                        line = self.sep.join(write_list) + '\n'
                        f.write(line)
                    elif self.target_model in ['cls_outer']:
                        binary_label = self.get_binary_label(label)
                        write_list = [cur, label, binary_label]
                        line = self.sep.join(write_list) + '\n'
                        f.write(line)
                    elif self.target_model in ['seg', 'seg_withref', 'seg_noref', 'withref']:
                        write_list = [cur, mask]
                        line = self.sep.join(write_list) + '\n'
                        f.write(line)
                    elif self.target_model in ['comp_seg', 'compseg']:
                        line = cur + '\n'
                        f.write(line)
                    elif self.target_model in ['zone_cls']:
                        write_list = [cur, label]
                        line = self.sep.join(write_list) + '\n'
                        f.write(line)
                    elif self.target_model in ['sk']:
                        gerb = self.get_or_null(img_data, 'gerb', include_null_path=self.include_null_path)
                        write_list = [cur, gerb, mask]
                        line = self.sep.join(write_list) + '\n'
                        f.write(line)
            if addup is not None:
                with open(addup, 'r') as add_file:
                    add_lines = add_file.readlines()
                f.writelines(add_lines)

        if review:
            print('Generated: %s' % output_file)


    def _merge_label(self, label: str) -> str:
        return self.merge_labels[label] if self.merge_labels is not None else label

    def report_dataset_info(self,
                            split_ratio: float = 0.8,
                            count_attrs: Union[str, List[str], None] = None,
                            ascending: bool = False) -> DataContainer:
        if isinstance(count_attrs, str):
            count_attrs = [count_attrs]
        train, _ = self._split(split_ratio=split_ratio)
        df = train.get_statistics(attrs=count_attrs, sort_by=DataContainer.NUM, ascending=ascending)

        print('Number of Classes:', len(train), '\n')
        show_count = [DataContainer.NUM] + count_attrs if isinstance(count_attrs, list) else DataContainer.NUM
        print(df[show_count].sum())
        print('\n')
        print(df)
        return train

    # @record
    def set_label_exceptions(self,
                             label_exceptions: List,
                             fn: Optional[Callable] = None) -> NoReturn:
        if label_exceptions is not None:
            assert type(label_exceptions) == list, 'Expected Input Type: (List)'
            label_exceptions = label_exceptions + [label + '_train' for label in label_exceptions]
        if isfunction(fn):
            label_exceptions = fn(label_exceptions)
        setattr(self, 'label_exceptions', label_exceptions)

    # @record
    def set_dataset_exceptions(self,
                               dataset_exceptions: List,
                               fn: Optional[Callable] = None) -> NoReturn:
        if dataset_exceptions is not None:
            assert type(dataset_exceptions) == list, 'Expected Input Type: (List)'
        if isfunction(fn):
            dataset_exceptions = fn(dataset_exceptions)
        setattr(self, 'dataset_exceptions', dataset_exceptions)

    # @record
    def set_merge_labels(self,
                         merge_labels: Dict,
                         fn: Optional[Callable] = None) -> NoReturn:
        """
        :param merge_labels:
        example:

        a = {
            '009': ['0091', '0092'],
            '012': ['0129'],
        }
        :param fn:
        :return:
        """

        if isfunction(fn):
            merge_labels = fn(merge_labels)
        else:
            if merge_labels is not None:
                temp = dict()
                for k, v in merge_labels.items():
                    for label in v:
                        temp[label] = k
                merge_labels = temp
        setattr(self, 'merge_labels', merge_labels)

    # @record
    def set_top_labels(self,
                       top_labels: List,
                       fn: Callable = None) -> NoReturn:
        if isfunction(fn):
            top_labels = fn(top_labels)
        setattr(self, 'top_labels', top_labels)

    @staticmethod
    def find_datasets(root_dir: str) -> List:
        root_dir = PathFormatter.format(root_dir)
        datasets = [os.path.join(root_dir, dataset) for dataset in os.listdir(root_dir)]
        print('TODO: Search Dataset Tree')
        return datasets

    @staticmethod
    def format_suffix(suffix: str) -> str:
        return '_' + suffix if suffix is not None else ''

    @staticmethod
    def get_or_null(img_data: ImageData,
                    attr: str,
                    include_null_path: bool = False) -> Union[SingleImage, str, None]:
        assert isinstance(img_data, ImageData)
        ret = getattr(img_data, attr)
        if include_null_path and ret is None:
            return 'null'
        elif not include_null_path and ret is None:
            raise FileNotFoundError(f'The {attr} file of {img_data} is missing!')
        return ret

    @staticmethod
    def parse_config(config: Dict) -> str:
        save_path = config.pop('save_path', None)
        if not save_path:
            weight_root = config.pop('weight_root')
            weight_root = PathFormatter.format(weight_root)

            project = config.pop('project')
            update = config.pop('update')
            surface = config.get('surface')
            target_model = config.get('target_model')
            if target_model in DataListGenerator.CONFIG['target_model_map']:
                target_model = DataListGenerator.CONFIG['target_model_map'][target_model]

            if surface is not None:
                save_path = os.path.join(weight_root, project, surface, target_model, update, 'dataset')
            else:
                save_path = os.path.join(weight_root, project, target_model, update, 'dataset')
        return PathFormatter.format(save_path)


class DataListParser:
    CONFIG = {
        'cls': [CUR_COL, CLUSTER_COL, MASK_COL],
        'cls_outer': [CUR_COL, CLUSTER_COL, 'Binary'],
        'seg': [CUR_COL, MASK_COL],
        'compseg': [CUR_COL],
        'zone_cls': [CUR_COL, CLUSTER_COL],
        'sk': [CUR_COL, GERBER_COL, MASK_COL]
    }

    MAPPER = convert2map({
        'cls': ['cls', 'defect_cls'],
        'cls_outer': ['cls_outer'],
        'seg': ['seg', 'seg_withref', 'withref', 'seg_noref', 'noref'],
        'compseg': ['comp', 'compseg', 'comp_seg'],
        'zone_cls': ['zone', 'zone_cls'],
        'sk': ['sk', 'saikong', 'sk_compseg']
    })

    def __init__(self,
                 file_path: str,
                 target_model: str,
                 sep: str = '|',
                 separated: Optional[bool] = None,
                 use_single_image: bool = False,
                 auto_parse=False):
        self._target_file = PathFormatter.format(file_path)
        self._target_model = target_model
        self._sep = sep
        self._separated = separated

        if auto_parse:
            self._data = self.parse(use_single_image=use_single_image)


    def parse(self, use_single_image: bool = False) -> DataContainer:
        data = pd.read_csv(self._target_file, sep=self._sep, header=None)
        try:
            data.columns = DataListParser.CONFIG[DataListParser.MAPPER[self._target_model]]
        except ValueError:
            raise ValueError(f'Current separator is "{self._sep}", please check the separator in data list!')
        self._data = data.copy()
        if CLUSTER_COL not in data.columns:
            data[CLUSTER_COL] = 'Unknown'
        data[CUR_COL] = data[CUR_COL].apply(lambda x: ImageData(x,
                                                                use_single_image=use_single_image,
                                                                separated=('Cur' in x)
                                                                if is_none(self._separated)
                                                                else self._separated))
        return DataContainer.from_dataframe(src=data[CUR_COL])

    @property
    def raw_data(self) -> pd.DataFrame:
        data = pd.read_csv(self._target_file, sep=self._sep, header=None)
        try:
            data.columns = DataListParser.CONFIG[DataListParser.MAPPER[self._target_model]]
        except ValueError:
            raise ValueError(f'Current separator is "{self._sep}", please check the separator in data list!')
        self._data = data.copy()
        if CLUSTER_COL not in data.columns:
            data[CLUSTER_COL] = 'Unknown'
        data[CUR_COL] = data[CUR_COL].apply(lambda x: ImageData(x,
                                                                use_single_image=False,
                                                                separated=('Cur' in x)
                                                                if is_none(self._separated)
                                                                else self._separated))
        return data

    def get_top_label_weights(self):
        assert self._target_model in ['cls', 'cls_outer', 'defect_cls'],\
            'Only cls, cls_outer and defect_cls support top label weights!'

        datalist = self.raw_data
        groupby_count = datalist.groupby(CLUSTER_COL).count()
        total_num = groupby_count[CUR_COL].sum()
        groupby_count['weight'] = groupby_count[CUR_COL].apply(lambda x: pow(total_num / x, 0.5))
        return groupby_count['weight'].to_dict()

    @property
    def data(self):
        return self._data




if __name__ == '__main__':
    # configs = {
    #     'weight_root': r'\data\dataset\TrainLog\weights\sunjianyao',
    #     'project': 'qyjl',
    #     'target_model': 'cls',
    #     'update': 'test',
    #     'allow_duplicates': True,
    #     'sep': ' ',
    #     'top_labels': ['Edge']
    # }
    # test_gen = DataListGenerator.set_project_update(config=configs)
    # test_gen.load_dataset(r'\data\dataset2\Workshop\sunjianyao\qyjl\train_data\zone_cls\20230315_fallout_update', duplicates=2)
    a = DataListParser(r"\data\dataset2\TrainLog\weights\sunjianyao\szcd\cls\20230728_hard_samples_update\dataset\train.txt", 'cls_outer', sep='#')
    ddd = a.get_top_label_weights()
    # print(ddd.size)
