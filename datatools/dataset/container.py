import re
import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial, reduce
from collections import defaultdict
from collections.abc import Iterable
from multiprocessing import Pool
from typing import Optional, Dict, List, Union, Callable, NoReturn, Tuple, Any

from datatools.image.data import ImageData, SingleImage
from datatools.analyzer.utils import CUR_COL, NUM_COL, CLUSTER_COL, SHAPE_COL
from datatools.utils import PathFormatter, SuffixFormatter, is_none, is_not_none, scandir


class DataContainer(defaultdict):

    COLS = [CUR_COL, CLUSTER_COL]
    CUR = COLS[0]
    CLUSTER = COLS[1]
    NUM = NUM_COL
    SHAPE = SHAPE_COL

    def __init__(self, allow_duplicates=True, **kwargs):
        """ Dictionary of list of ImageData.

        Functions:
            strict_duplication_check: Check duplication of images.

            get_difference_with: Get difference with another DataContainer.

            get_statistics: Get statistics of the DataContainer.

            limit_num: Limit the number of images in each cluster.

            limit_num_ratio: Limit the number of images in each cluster by ratio.

            remove_num: Remove the number of images in each cluster.

            num_condition: Get the clusters that meet the condition.

            with_attrs: Get the clusters that have the attributes.

            without_attrs: Get the clusters that do not have the attributes.

            pop_empty_keys: Pop the empty clusters.

            is_empty: Check if the DataContainer is empty.

            class_copy: Get a copy of the DataContainer.

            count_data_with_attr: Count the number of images with the attribute.

            merge_cluster: Merge all clusters into one.

            to_dataframe: Convert to pandas DataFrame.

            to_txt: Save to txt file.

            export_to: Export the images to the destination.

            size: Get the size of each cluster.

            total_num: Get the total number of images.

        Args:
            allow_duplicates (bool, int): Whether to allow duplicates.
        """
        # if 'allow_duplicates' in kwargs:
        #     allow_duplicates = kwargs.pop('allow_duplicates')
        super(DataContainer, self).__init__(list, **kwargs)
        self.allow_duplicates = allow_duplicates
        self.statistics = None

    def __add__(self, other):
        assert isinstance(other, DataContainer), f'Cannot add DataContainer with {type(other)}'
        allow_duplicates = self.allow_duplicates or other.allow_duplicates
        ret = self.class_copy()
        for k, v in other.items():
            ret[k].extend(v)
            if not allow_duplicates:
                ret[k] = list(set(ret[k]))
        return ret

    @classmethod
    def from_txt(cls, src: str, allow_duplicates: bool = True, separated: Optional[bool] = None):
        src = PathFormatter.format(src)
        df = pd.read_csv(src, header=None)
        df.columns = DataContainer.COLS
        df[DataContainer.CUR] = df[DataContainer.CUR].apply(lambda x: ImageData(x,
                                                                                separated='Cur' in x
                                                                                if is_none(separated)
                                                                                else separated))
        return DataContainer.from_dataframe(df, allow_duplicates=allow_duplicates)

    @classmethod
    def from_dataframe(cls, src: pd.DataFrame, allow_duplicates: bool = True):
        assert (src.columns == DataContainer.COLS).all(), \
            f'Columns should be {DataContainer.COLS} instead of {list(src.columns)}'
        ret = cls(allow_duplicates=allow_duplicates)
        # for cluster in src[DataContainer.CLUSTER].sort_values().unique():
        #     ret[cluster].extend(src[src[DataContainer.CLUSTER] == cluster][DataContainer.CUR].to_list())
        for cluster, cur in zip(src[DataContainer.CLUSTER], src[DataContainer.CUR]):
            ret.append_to_cluster(cluster=cluster, data=cur)

        return ret

    @classmethod
    def from_scan_dir(cls,
                      src: str,
                      ignore_ref: bool = True,
                      ignore_gerb: bool = True,
                      strict_inspection: bool = False,
                      allow_duplicates: bool = True,
                      by_cluster: bool = True):
        src = PathFormatter.format(src)
        scanned = cls(allow_duplicates=allow_duplicates)
        exclude_suffix = tuple('_' + suffix for suffix in SuffixFormatter.MAPPER.keys()
                               if suffix not in ['ref', 'std', 'gerb'])
        if ignore_ref:
            exclude_suffix += ('_ref', '_std')
        if ignore_gerb:
            exclude_suffix += ('_gerb',)
        for ret in scandir(src,
                           recursive=True,
                           exclude_suffix=exclude_suffix,
                           with_extension=tuple('.' + ext for ext in SuffixFormatter.SUPPORT_FORMAT)):
            if (SuffixFormatter.is_cur(ret)
                    or (not ignore_ref and SuffixFormatter.is_attr(ret, 'ref'))
                    or (not ignore_gerb and SuffixFormatter.is_attr(ret, 'gerb'))):
                img = ImageData(ret, separated='Cur' in ret, strict_inspection=strict_inspection)
                scanned[img.label if by_cluster else "all"].append(img)

        return scanned

    @property
    def size(self) -> defaultdict:
        lens = defaultdict(int)
        for k in sorted(self.keys()):
            lens[k] = len(self[k])
        return lens

    @property
    def total_num(self) -> int:
        total = 0
        for k, v in self.items():
            total += len(v)
        return total

    def is_empty(self) -> bool:
        return not self

    def append_to_cluster(self, cluster: str, data: Union[ImageData, List]):
        if isinstance(data, ImageData):
            data = [data]
        self[cluster].extend(data)
        if not self.allow_duplicates:
            self[cluster] = list(set(self[cluster]))

    def class_copy(self):
        return DataContainer(allow_duplicates=self.allow_duplicates, **self)

    def count_data_with_attr(self, attr) -> Dict:
        result = dict()
        for key, imgs in self.items():
            result[key] = len([0 for img in imgs if getattr(img, attr) is not None])
        return result

    def count_by_shape(self) -> pd.DataFrame:
        df = self.to_dataframe(with_shape=True)
        return df.groupby(by=SHAPE_COL).count()[CUR_COL]

    def pop_empty_keys(self) -> NoReturn:
        check = self.class_copy()
        for key, value in check.items():
            if len(value) == 0:
                self.pop(key)

    def remove_num(self,
                   targets: Dict,
                   in_place: bool = False):
        ret = self if in_place else self.class_copy()
        for key, num in targets.items():
            if key in ret:
                np.random.shuffle(ret[key])
                ret[key] = ret[key][:-num]
        return ret

    def limit_num(self,
                  targets: Union[Dict, int],
                  in_place: bool = False):
        if isinstance(targets, int):
            targets = {key: targets for key in self}
        ret = self if in_place else self.class_copy()
        for key, num in targets.items():
            if key in ret:
                num_in_cluster = len(ret[key])
                hard_samples = []
                non_hard_samples = []
                for img in ret[key]:
                    if img.is_hard_sample:
                        hard_samples.append(img)
                    else:
                        non_hard_samples.append(img)
                num_hard_samples = len(hard_samples)
                if num_hard_samples >= num:
                    print(f'Limit Cluster "{key}" to the number of hard samples: '
                          f'{num_in_cluster} -> {num_hard_samples}')
                    ret[key] = hard_samples
                else:
                    np.random.shuffle(non_hard_samples)
                    print(f'Limit Cluster "{key}": {num_in_cluster} -> {min(num, num_in_cluster)}'
                          f' ({num_hard_samples} hard samples)')
                    ret[key] = hard_samples + non_hard_samples[:num - num_hard_samples]
        return ret

    def limit_num_ratio(self,
                        targets: Union[Dict, float],
                        in_place: bool = False):

        if isinstance(targets, float):
            num_condition = {key: int(targets * len(self[key])) for key in self.keys()}
        else:
            num_condition = {key: int(ratio * len(self[key])) for key, ratio in targets.items()}
        return self.limit_num(targets=num_condition, in_place=in_place)

    def num_condition(self, condition: str, get_keys: bool = False):
        self._condition_legal_check(condition)
        ret = self.class_copy()
        keys = []
        for key, value in self.items():
            compare = str(len(value)) + condition
            if not eval(compare):
                ret.pop(key)
                keys.append(key)
        if ret.is_empty():
            print(f'> Empty DataContainer under condition: "{condition}"!')
        if get_keys:
            return ret, keys
        return ret

    @staticmethod
    def _condition_legal_check(condition):
        condition_regex = re.compile(r'\s*[><]=?\s*[0-9]+|!=\s*[0-9]+|==\s*[0-9]+')
        assert is_not_none(re.fullmatch(condition_regex, condition)), \
            'ILLEGAL condition input! Should be [<|<=|>|>=|!=|==][0-9]+'
        return True

    @staticmethod
    def merge_cluster(container, merge_name='all', allow_duplicates=None):
        assert isinstance(container, DataContainer), f'Input expected to be DataContainer, got {type(container)}!'
        allow_duplicates = container.allow_duplicates if allow_duplicates is None else allow_duplicates
        merge_data = DataContainer(allow_duplicates=allow_duplicates)
        for cluster, data in container.items():
            ext_data = list(set(data)) if not allow_duplicates else data
            merge_data[merge_name].extend(ext_data)
        return merge_data

    def map(self, func: Callable, num_workers: int = 4) -> Dict:
        result = dict()
        with Pool(num_workers) as pool:
            for cluster, data in self.items():
                result[cluster] = pool.map(func, data)
        return result

    def map_reduce(self,
                   map_func: Callable,
                   reduce_func: Callable,
                   num_workers: int = 4,
                   by_cluster: bool = False) -> Dict:
        result = dict() if by_cluster else []
        with Pool(num_workers) as pool:
            for cluster, data in self.items():
                cluster_result: List = pool.map(map_func, data)
                cluster_result = reduce(reduce_func, cluster_result)
                if by_cluster:
                    result[cluster] = cluster_result
                else:
                    result.append(cluster_result)
        return result if by_cluster else reduce(reduce_func, result)

    def async_apply(self,
                    func: Callable,
                    num_workers: int = 6,
                    apply_by_cluster: bool = False,
                    return_result: bool = False,
                    time_out: Optional[int] = None,
                    **kwargs) -> Any:
        """

        The function will apply the func to each image in the DataContainer.

        apply_func should be defined as:
        >>> def apply_func(img: ImageData, cluster_name: str, *args, **kwargs):
                pass

        """

        result = dict()
        kwargs = dict() if kwargs is None else kwargs
        pbar = tqdm(total=self.total_num)

        def callback(*args):
            pbar.update()
        cluster_num = len(self)
        with Pool(num_workers) as pool:
            for i, (cluster, imgs) in enumerate(self.items()):
                pbar.set_description(f'[{i+1}/{cluster_num}] Cluster["{cluster}"] - ({len(imgs)})')
                apply_fn = partial(func, **kwargs)
                res = [pool.apply_async(apply_fn,
                                        args=(img, cluster) if apply_by_cluster else (img,),
                                        callback=callback) for img in imgs]
                result[cluster] = [x.get(timeout=time_out) for x in res]
        if return_result:
            return result

    def get_difference_with(self, other):
        assert isinstance(other, DataContainer), f'DataContainer cannot compare with {type(other)}!'
        difference = DataContainer(allow_duplicates=False)
        key_diff = self.keys() - other.keys()
        key_mutual = self.keys() & other.keys()
        for diff in key_diff:
            difference[diff].extend(set(self[diff]))
        for mutual in key_mutual:
            difference[mutual].extend(set(self[mutual]) - set(other[mutual]))
        return difference

    def get_union_with(self, other):
        assert isinstance(other, DataContainer), f'DataContainer cannot compare with {type(other)}!'
        union = DataContainer(allow_duplicates=False)
        key_union = self.keys() | other.keys()
        for key in key_union:
            union[key].extend(set(self[key]) | set(other[key]))
        return union

    def get_intersect_with(self, other):
        assert isinstance(other, DataContainer), f'DataContainer cannot compare with {type(other)}!'
        difference = DataContainer(allow_duplicates=False)
        key_mutual = self.keys() & other.keys()
        for mutual in key_mutual:
            difference[mutual].extend(set(self[mutual]) & set(other[mutual]))
        return difference

    def get_statistics(self,
                       attrs: Union[str, List, None] = None,
                       sort_by: Union[str, List, None] = None,
                       ascending: bool = False) -> pd.DataFrame:

        data = self.size
        if isinstance(attrs, str):
            attrs = [attrs]
        df = pd.DataFrame({DataContainer.CLUSTER: data.keys(),
                           DataContainer.NUM: data.values()})
        if attrs is not None:
            for attr in attrs:
                count = {cluster_name: len([None for img in data if getattr(img, attr) is not None])
                         for cluster_name, data in self.items()}
                df[attr] = df[DataContainer.CLUSTER].map(count)
        if sort_by is not None:
            if sort_by in df.columns:
                return df.sort_values(by=sort_by, ascending=ascending)
        return df

    @staticmethod
    def _get_img_shape(img: Union[ImageData, str],
                       with_channel: bool = True,
                       target_attr: Optional[str] = None,
                       *args,
                       **kwargs):

        if isinstance(img, str):
            img = ImageData(img)

        if target_attr is None:
            target_attr = 'cur'

        img.enable_single_image()
        attr = getattr(img, target_attr)

        try:
            shape = attr.shape
        except AttributeError:
            return {f"{target_attr.upper()}_NOT_EXIST": 1}

        return {str(shape) if with_channel else str(shape[:2]): 1}

    @staticmethod
    def _reduce_count(x, y) -> Dict:
        for key in x.keys():
            if key in y:
                if isinstance(y[key], dict):
                    DataContainer._reduce_count(x[key], y[key])
                else:
                    y[key] += x[key]
            else:
                y[key] = x[key]
        return y

    def count_shape(self,
                    num_workers: int = 4,
                    by_cluster: bool = False,
                    with_channel: bool = True,
                    target_attr: Optional[str] = None):
        map_func = partial(self._get_img_shape,
                           with_channel=with_channel,
                           target_attr=target_attr)
        count: dict = self.map_reduce(map_func=map_func,
                                      reduce_func=self._reduce_count,
                                      num_workers=num_workers,
                                      by_cluster=by_cluster)

        if by_cluster:
            count = {(cluster, shape): num for cluster, v in count.items() for shape, num in v.items()}
            df_index = pd.MultiIndex.from_tuples(count.keys())
        else:
            df_index = list(count.keys())

        return pd.DataFrame({'count' if target_attr is None else target_attr.capitalize(): count.values()},
                            index=df_index)

    def to_list(self) -> List:
        return [img for imgs in self.values() for img in imgs]

    def to_dataframe(self, with_shape: bool = False, with_channel: bool = False) -> pd.DataFrame:
        data = {DataContainer.CUR: [],
                DataContainer.CLUSTER: []}
        for k, v in self.items():
            data[DataContainer.CUR].extend(v)
            data[DataContainer.CLUSTER].extend([k] * len(v))
        self.statistics = pd.DataFrame(data)
        if with_shape:
            def map_shape(img: ImageData):
                shape = SingleImage(img.cur).shape if isinstance(img.cur, str) else img.cur.shape
                return shape if with_channel else shape[:2]

            self.statistics[SHAPE_COL] = self.statistics[CUR_COL].map(map_shape)
        return self.statistics.copy()

    def to_txt(self,
               dst: str,
               name: Optional[str] = None,
               sep: str = '|',
               with_shape: bool = False,
               force: bool = False) -> NoReturn:
        df = self.to_dataframe(with_shape=with_shape)
        dst = PathFormatter.format(dst)
        if is_none(name):
            now = datetime.datetime.now()
            name = '_'.join([str(now.year), str(now.month), str(now.day),
                             str(now.hour), str(now.minute)])
        if force:
            os.makedirs(dst)
        save_file = os.path.join(dst, f'{name}.txt')
        if not os.path.exists(dst):
            raise FileNotFoundError(f'Destination not found: {dst}')
        df.to_csv(save_file, sep=sep, header=False, index=False)
        print(f'DataContainer txt generated: {save_file}')

    def without_attrs(self,
                      attrs: Union[str, List],
                      without_all: bool = False):
        return self._conditioned_data_attrs(attrs=attrs, condition=is_none, condition_all=without_all)

    def with_attrs(self,
                   attrs: Union[str, List],
                   with_all: bool = False):
        return self._conditioned_data_attrs(attrs=attrs, condition=is_not_none, condition_all=with_all)

    def _conditioned_data_attrs(self,
                                attrs: Union[str, List[str]],
                                condition: Callable,
                                condition_all: bool = False):
        if isinstance(attrs, str):
            attrs = [attrs]
        result = DataContainer(allow_duplicates=self.allow_duplicates)
        conditioning = np.all if condition_all else np.any
        for cluster, imgs in self.items():
            for img in imgs:
                img_status = [condition(getattr(img, attr)) for attr in attrs]
                if conditioning(img_status):
                    result[cluster].append(img)

        return result

    @staticmethod
    def _export_by_cluster(img: ImageData,
                           cluster: str,
                           dst: str,
                           force: bool,
                           force_copy: bool,
                           separate: bool,
                           move: bool,
                           target_attrs: Union[List, str, None],
                           exceptions: Union[List, str, None]):
        img.copy_to(dst=os.path.join(dst, cluster),
                    force_copy=force_copy,
                    force=force,
                    move=move,
                    separate=separate,
                    target_attrs=target_attrs,
                    exceptions=exceptions)

    def export_to(self,
                  dst: str,
                  force: bool = False,
                  force_copy: bool = True,
                  move: bool = False,
                  separate: bool = True,
                  target_attrs: Union[List, str, None] = None,
                  exceptions: Union[List, str, None] = None,
                  num_workers: int = 6):
        dst = PathFormatter.format(dst)
        # for cluster, imgs in self.items():
        #     for img in tqdm(imgs):
        #         if isinstance(img, str):
        #             img = ImageData(img)
        #         img.copy_to(dst=os.path.join(dst, cluster),
        #                     force=force,
        #                     move=move,
        #                     separate=separate,
        #                     exceptions=exceptions)

        self.async_apply(func=self._export_by_cluster,
                         apply_by_cluster=True,
                         num_workers=num_workers,
                         dst=dst,
                         force=force,
                         force_copy=force_copy,
                         move=move,
                         separate=separate,
                         target_attrs=target_attrs,
                         exceptions=exceptions)

        print(f'Data Exported to: {dst}')

    def duplication_check(self, image_check: bool = False) -> 'DataContainer':
        merge_data = DataContainer.merge_cluster(self, merge_name='all', allow_duplicates=True)
        hash_table = dict()
        duplicates = dict()
        pbar = tqdm(merge_data['all'])
        dup_num = 0
        for img in pbar:
            if image_check:
                img.enable_strict_inspection()
            else:
                img.disable_strict_inspection()
            hash_value = str(hash(img))
            if hash_value not in hash_table.keys():
                hash_table[hash_value] = img
            else:
                if hash_value not in duplicates:
                    duplicates[hash_value] = [hash_table[hash_value]]
                duplicates[hash_value].append(img)
                dup_num += 1
            num_imgs = len(duplicates)
            pbar.set_postfix(Duplicates=num_imgs, Total=dup_num+num_imgs)
            pbar.update(1)
        result = DataContainer(allow_duplicates=True, **duplicates)
        return result

    def strict_duplication_check(self) -> 'DataContainer':
        return self.duplication_check(image_check=True)

    def get_error_semantic_data(self, priority_class, color_mapper) -> 'DataContainer':
        result = DataContainer(allow_duplicates=False)
        for cluster, imgs in self.items():
            for img in imgs:
                if img.mask is None:
                    continue
                ss_label = img.get_semantic_class(priority_class=priority_class, color_mapper=color_mapper)
                if ss_label != cluster:
                    result.append_to_cluster(cluster=ss_label, data=img)
        return result

    def get_shape_with(self,
                       shape: Union[Tuple[int], Iterable, int],
                       with_channel: bool = True) -> 'DataContainer':
        if isinstance(shape, Iterable):
            shape = tuple(shape)
        elif isinstance(shape, int):
            shape = (shape, shape)
        assert len(shape) == 2, f'Shape should be a tuple of length 2, got {len(shape)}'
        if self.statistics is None or (DataContainer.SHAPE not in self.statistics.columns):
            df = self.to_dataframe(with_shape=True, with_channel=with_channel)
        else:
            df = self.statistics.copy()
        return self.from_dataframe(df[df[SHAPE_COL] == shape])

    def get_marked_data(self) -> 'DataContainer':
        result = DataContainer(allow_duplicates=self.allow_duplicates)
        for cluster, imgs in self.items():
            for img in imgs:
                if img.mark is not None:
                    result.append_to_cluster(cluster=img.mark, data=img)
        return result

    def select(self, targets: Union[str, List[str]]) -> 'DataContainer':
        if isinstance(targets, str):
            targets = [targets]
        result = DataContainer(allow_duplicates=self.allow_duplicates)
        for cluster in targets:
            result[cluster] = self[cluster]
        return result

    def rename_cluster(self, clusters: Dict[str, str], in_place: bool = False):
        data_container = self if in_place else self.class_copy()
        for old, new in clusters.items():
            if old in data_container:
                self[new] = data_container.pop(old)
        if not in_place:
            return data_container

    def show_data_in_cluster(self,
                             cluster: str,
                             attrs: Union[str, List[str], None] = None,
                             destroy_all_windows: bool = False,
                             allowed_keys: Union[str, List[str], None] = ' ',
                             allowed_marks: Union[str, List[str], None] = None,):
        assert cluster in self, f"{cluster} given is not contained!"
        if attrs is None:
            attrs = ['cur']
        for img in self[cluster]:
            img.enable_single_image()
            image_attrs = [(attr, getattr(img, attr)) for attr in attrs if getattr(img, attr) is not None]
            max_index_attrs = len(attrs) - 1
            for i, (attr, attr_img) in enumerate(image_attrs):
                if i == max_index_attrs:
                    attr_img.show(wait_key=True,
                                  named_window=attr,
                                  destroy_all_windows=destroy_all_windows,
                                  allowed_keys=allowed_keys,
                                  allowed_marks=allowed_marks)
                else:
                    attr_img.show(named_window=attr)

    def get_semantic_cluster(self, priority_class, color_mapper):
        result = DataContainer(allow_duplicates=False)
        for _, imgs in self.items():
            for img in imgs:
                ss_label = img.get_semantic_cluster(priority_class=priority_class, color_mapper=color_mapper)
                result.append_to_cluster(cluster=ss_label, data=img)
        return result

    @staticmethod
    def _export_dup_func(img: Union[ImageData, str],
                         md5v: str,
                         dst: str,
                         by_cluster: bool,
                         force: bool,
                         separate: bool,
                         move: bool,
                         exceptions: Union[List, str, None],
                         *args,
                         **kwargs):
        if isinstance(img, str):
            img = ImageData(img)
        img.copy_to(dst=os.path.join(dst, md5v, img.label) if by_cluster else os.path.join(dst, md5v),
                    force=force,
                    force_copy=True,
                    overwrite=False,
                    separate=separate,
                    move=move, exceptions=exceptions, inplace=True)

    def export_duplicate_data_to(self,
                                 dst: str,
                                 force: bool = False,
                                 move: bool = False,
                                 separate: bool = True,
                                 by_cluster: bool = False,
                                 exceptions: Union[List, str, None] = None):

        dst = PathFormatter.format(dst)
        export_dup_func = partial(self._export_dup_func,
                                  dst=dst,
                                  by_cluster=by_cluster,
                                  force=force,
                                  separate=separate,
                                  move=move,
                                  exceptions=exceptions)

        self.async_apply(export_dup_func, apply_by_cluster=True)


    def fast_duplicate_check(self, attrs: Union[str, List[str]] = 'mask', destroy_all_windows: bool = False):
        attrs = [attrs] if isinstance(attrs, str) else attrs
        for md5v, imgs in self.items():
            num = len(imgs) - 1
            for i, img in tqdm(enumerate(imgs)):
                img: 'ImageData'
                img.enable_single_image()
                for attr in attrs:
                    image_attr: 'SingleImage' = getattr(img, attr)
                    try:
                        image_attr.show(f'{attr.capitalize()}_{i}')
                    except AttributeError:
                        pass
                img.cur.show(f'Cur_{i}', wait_key=i == num, destroy_all_windows=destroy_all_windows)

    def __repr__(self):
        return f'DataContainer(total_num:{self.total_num}, {dict(**self.size)})'

    def __iter__(self):
        for k, v in self.items():
            for img in v:
                yield img


if __name__ == '__main__':
    from datatools import ICSemanticMapper
    # a = DataContainer(allow_duplicates=True, a=['1', '3'], b=['7', '8'])
    # b = DataContainer(allow_duplicates=False, a=['1', '2'], c=[])
    # print(a.get_difference_with(b))
    # print()
    # b.remove_num({'a': 1})
    # c = a.num_condition('>2')
    dc = DataContainer.from_scan_dir(r'\data\dataset2\Workshop\wangyueyi\KSNY\train_add_data\
    color\noref_semantic\ng-1-2彩图无标准图语义-850\012')
    print(dc)
    error = dc.get_error_semantic_data(['012', '062', '206', '018', '0151'], color_mapper=ICSemanticMapper)
    print(error)


