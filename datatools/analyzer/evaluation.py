import os
import itertools

import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from sklearn.metrics import confusion_matrix

from datatools.image import ImageData, SingleImage
from datatools.utils import PathFormatter, is_not_none, convert2map
from .utils import select_in_col, check_img, CUR_COL, NUM_COL, exclude_by_condition, groupby_counts



class ClsEvaluation:
    def __init__(self):
        pass

    @staticmethod
    def compare_between(series_a, series_b, keys=None):
        if keys is not None:
            assert isinstance(keys, list), 'keys must be list!'
        else:
            keys = ['first', 'second']

        ret = pd.concat([series_a, series_b], axis=1, keys=keys)
        ret['diff'] = ret[keys[0]] - ret[keys[1]]
        return ret

    @staticmethod
    def check_selected_img(selected, show_mask=False, ref='gerb'):
        for _, data in selected.iterrows():
            print(f'Checking: {data[CUR_COL]}')
            check_img(data, show_mask=show_mask, ref=ref)


    @staticmethod
    def find_data_from(src, file_name):
        return src[src['Cur'].str.contains(file_name)]

    @staticmethod
    def select_defect_code(src, pred_cls_x=None, pred_cls_y=None, defect_code=None, drop_duplicates=False, **kwargs):
        if pred_cls_x is not None:
            src = select_in_col(src, 'pred_cls_x', pred_cls_x)
        if pred_cls_y is not None:
            src = select_in_col(src, 'pred_cls_y', pred_cls_y)
        if defect_code is not None:
            src = select_in_col(src, 'defect_code', defect_code)
        if drop_duplicates:
            src = src.drop_duplicates('ori', keep='first')
        return src

    @staticmethod
    def select_count_threshold(src, threshold, cols=None):
        if cols is None:
            cols = ['pred_cls_x', 'pred_cls_y']
        else:
            assert isinstance(cols, list), 'cols must be a list!'
        temp = src.groupby(cols).count().reset_index()
        temp = temp[temp['Cur'] >= threshold][cols]
        conditions = ['('+'&'.join(['=='.join([col, '"'+data[col]+'"']) for col in cols])+')' for _, data in temp.iterrows()]
        threshold_condition = '|'.join(conditions)
        return src.query(threshold_condition)


    @staticmethod
    def max_count_by(src, by=None):
        assert NUM_COL in src, f"Column '{NUM_COL}' must in src!"
        by = ClsInference.GT if by is None else by
        return src.loc[src.groupby(by)[NUM_COL].idxmax()]


    @staticmethod
    def merge(left, right):
        assert isinstance(left, ClsInference) and isinstance(right, ClsInference)
        return pd.merge(left.data, right.data, on=[ClsInference.CUR, ClsInference.GT],
                        suffixes=('_'+left.name, '_'+right.name))


    @staticmethod
    def report_merge_result(sources, matrix='precision'):
        assert matrix in ['precision', 'recall']
        results = [getattr(src, matrix) for src in sources]
        keys = [matrix+'_'+getattr(src, 'name') for src in sources]
        return pd.concat(results, axis=1, keys=keys)


class ClsInference:
    __slots__ = ['_name', '_data', 'merge_labels', '_threshold', '_file_root', 'exclude_gt']
    COLS = [CUR_COL, 'Cluster', 'Predict', 'Score']
    # COLS = ['Cur', 'defect_code', 'pred_cls', 'defect_score']
    CUR = COLS[0]
    GT = COLS[1]
    PRED = COLS[2]
    SCORE = COLS[3]

    def __init__(self, name: str, data=None, merge_labels=None, threshold=None, file_root=None, exclude_gt=None, **kwargs):
        if data is not None:
            assert isinstance(data, pd.DataFrame) and (ClsInference.COLS == data.columns).all()

        if threshold is not None:
            assert isinstance(threshold, str) and ClsInference._condition_legal_check(threshold), \
                f'threshold must be str!'

        self._name = name
        self._data = data
        self._threshold = threshold
        self.merge_labels = merge_labels
        self.exclude_gt = exclude_gt
        self._file_root = file_root


    @classmethod
    def read_csv(cls,
                 file_path,
                 name=None,
                 merge_labels=None,
                 threshold=None,
                 exclude_gt=None,
                 sep='|',
                 **kwargs):
        file_path = PathFormatter.format(file_path)
        data = pd.read_csv(file_path,
                           sep=sep,
                           dtype={ClsInference.GT: str,
                                  ClsInference.PRED: str})
        file_root, file_name = os.path.split(file_path)
        name = os.path.splitext(file_name)[0] if name is None else name
        print(f'> Loaded from: {file_path}')
        return cls(name=name, data=data, merge_labels=merge_labels, threshold=threshold,
                   file_root=file_root, exclude_gt=exclude_gt,
                   **kwargs)

    @classmethod
    def from_config(cls, config: Dict, name=None):
        weight_root = config.pop('weight_root')
        project = config.pop('project')
        trial = config.pop('trial')
        test_split = config.pop('test_split')
        suffix = config.pop('suffix')
        data_trial = config.pop('data_trial')

        result_path = os.path.join(weight_root, project, 'cls',
                                   data_trial, 'dataset', trial, test_split+'_'+suffix+'.txt')
        if name is None:
            name = trial
        return ClsInference.read_csv(file_path=result_path,
                                     name=name,
                                     **config)

    @property
    def name(self) -> str:
        return self._name

    @property
    def raw_data(self) -> pd.DataFrame:
        return self._data

    @property
    def data(self) -> pd.DataFrame:
        data = self._data \
            if self._threshold is None \
            else self._data.query(ClsInference.SCORE+self._threshold).copy()

        data = data \
            if self.exclude_gt is None \
            else exclude_by_condition(data, ClsInference.GT, self.exclude_gt)

        return ClsInference.merge_label(data, self.merge_labels)

    @property
    def precision(self) -> pd.DataFrame:
        df = self.data.rename(columns={ClsInference.CUR: 'precision'})
        return df[df[ClsInference.GT] == df[ClsInference.PRED]].groupby(ClsInference.GT)['precision'].count() / \
               df.groupby(ClsInference.PRED)['precision'].count()

    @property
    def recall(self) -> pd.DataFrame:
        df = self.data.rename(columns={ClsInference.CUR: 'recall'})
        return df[df[ClsInference.GT] == df[ClsInference.PRED]].groupby(ClsInference.GT)['recall'].count() / \
               df.groupby(ClsInference.GT)['recall'].count()


    @property
    def error(self) -> pd.DataFrame:
        df = self.data
        return df[df[ClsInference.GT] != df[ClsInference.PRED]]

    @property
    def correct(self) -> pd.DataFrame:
        df = self.data
        return df[df[ClsInference.GT] == df[ClsInference.PRED]]


    @staticmethod
    def _condition_legal_check(condition):
        condition_regex = re.compile(r'\s*([><]=?|!=|==)\s*(?:\d+(?:\.\d*)?|\.\d+)\s*')
        assert is_not_none(re.fullmatch(condition_regex, condition)), \
            r'ILLEGAL condition input! Should be [<|<=|>|>=|!=|==] [int|float]'
        return True


    @staticmethod
    def merge_label(data, merge_labels):
        if merge_labels is not None:
            maps = convert2map(merge_labels)
            data[ClsInference.PRED] = data[ClsInference.PRED].replace(maps)
            data[ClsInference.GT] = data[ClsInference.GT].replace(maps)
        return data

    @staticmethod
    def counts(src, by='ground_truth') -> pd.DataFrame:
        sort_index = [ClsInference.PRED, ClsInference.GT] \
            if by in ['pred', 'pred_cls', 'prediction'] \
            else [ClsInference.GT, ClsInference.PRED]
        return groupby_counts(src=src, cols=sort_index)

    @staticmethod
    def check_selected_img(selected: pd.DataFrame,
                           ref: Optional[str] = 'gerb',
                           show_mask: bool = False,
                           destroy_window: bool = True,
                           format_score: str = '.4f'):
        assert np.all([col in selected.columns for col in ClsInference.COLS]), \
            f'Columns Required: {ClsInference.COLS}'
        selected = selected.copy()
        selected[ClsInference.SCORE] = selected[ClsInference.SCORE].map(lambda x: format(x, format_score))
        for idx, data in selected.iterrows():
            cur_path = data[ClsInference.CUR]
            ground_truth = data[ClsInference.GT]
            pred = data[ClsInference.PRED]
            score = data[ClsInference.SCORE]
            print(f'Checking: {cur_path}\n'
                  f'{ClsInference.GT}: {ground_truth}'
                  f'\t{ClsInference.PRED}: {pred}'
                  f'\t{ClsInference.SCORE}: {score}'
                  f'\tidx: {idx}\n')
            check_img(data_path=cur_path, show_mask=show_mask, ref=ref, destroy_window=False)
        if destroy_window:
            cv2.destroyAllWindows()

    @staticmethod
    def select_gt_pred(src, defect_code=None, pred_cls=None) -> pd.DataFrame:
        if defect_code is not None:
            src = select_in_col(src=src, col=ClsInference.GT, condition=defect_code)
        if pred_cls is not None:
            src = select_in_col(src=src, col=ClsInference.PRED, condition=pred_cls)
        return src

    def set_threshold(self, threshold: Optional[str]):
        if threshold is not None:
            assert isinstance(threshold, str), f'threshold must be str!'
        self._threshold = threshold

    def export_error_by(self, dst, target_labels=None, by='precision', exceptions=None, force=True) -> None:
        dst = PathFormatter.format(dst)
        recall_modes = ['pred', 'prediction', 'recall']
        outer_folder = ClsInference.PRED if by in recall_modes else ClsInference.GT
        inner_folder = ClsInference.GT if by in recall_modes else ClsInference.PRED
        if is_not_none(target_labels):
            if isinstance(target_labels, str):
                target_labels = [target_labels]
            else:
                assert isinstance(target_labels, list), 'target_labels must be a list!'
            df = select_in_col(src=self.error, col=ClsInference.PRED, condition=target_labels)
        else:
            df = self.error
        if is_not_none(exceptions):
            df = exclude_by_condition(src=df, col=ClsInference.PRED, exceptions=exceptions)
        for _, data in tqdm(df.iterrows()):
            img = ImageData(data[ClsInference.CUR])
            img.copy_to(os.path.join(dst, data[outer_folder], data[inner_folder]), force=force)

        print(f'> Exported by {outer_folder.capitalize()}/{inner_folder.capitalize()}:\n{dst}')

    def prelabel(self, dst, target_labels=None, exceptions=None, force=True) -> None:
        dst = PathFormatter.format(dst)
        if is_not_none(target_labels):
            if isinstance(target_labels, str):
                target_labels = [target_labels]
            else:
                assert isinstance(target_labels, list), 'target_labels must be a list!'
            df = select_in_col(src=self.data, col=ClsInference.PRED, condition=target_labels)
        else:
            df = self.data
        if is_not_none(exceptions):
            df = exclude_by_condition(src=df, col=ClsInference.PRED, exceptions=exceptions)

        for _, data in tqdm(df.iterrows()):
            img = ImageData(data[ClsInference.CUR])
            img.copy_to(os.path.join(dst, data[ClsInference.PRED]), force=force)

        print(f'> Prelabelled data at: {dst}')



    def save_confusion_matrix(self, data=None, suffix=None, percentage=True, figsize=(15, 10)):
        save_name = ['cm', self.name, suffix] if suffix is not None else ['cm', self.name]
        save_file = os.path.join(self._file_root, '_'.join(save_name) + '.png')
        cmap = plt.get_cmap('Blues')
        data = self.data if data is None else data
        labels = data[ClsInference.GT].sort_values().unique()
        cm = confusion_matrix(y_true=data[ClsInference.GT],
                              y_pred=data[ClsInference.PRED],
                              labels=labels)

        if percentage:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(' - '.join(['Confusion Matrix', self.name]))
        plt.colorbar()

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        thresh = cm.max() / 1.5 if percentage else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if percentage:
                plt.text(j, i, "{:.0f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Ground Truth', size=15)
        plt.xlabel('Prediction', size=15)
        plt.savefig(
            save_file,
            dpi=300,
            format='png',
            bbox_inches='tight')

        print(f'Saved Confusion Matrix: {save_file}')

    def show_confusion_matrix(self, suffix=None):
        save_name = ['cm', self.name, suffix] if suffix is not None else ['cm', self.name]
        save_file = os.path.join(self._file_root, '_'.join(save_name) + '.png')
        img = SingleImage(save_file)
        img.show(wait_key=True, named_window=' '.join([self.name, 'Confusion Matrix']))
        # plt.show()

    def get_confusion_matrix_table(self, data=None, format_output='.0f') -> pd.DataFrame:
        data = self.data if data is None else data
        counts = groupby_counts(data, [ClsInference.GT, ClsInference.PRED]).reset_index()
        total_num = data.groupby(ClsInference.GT).count()[ClsInference.CUR]
        counts['percentage'] = (counts[NUM_COL] / counts[ClsInference.GT].map(total_num)) * 100
        counts['percentage'] = counts['percentage'].map(lambda x: format(x, format_output))
        confusion_matrix_table = pd.pivot(counts,
                                          index=ClsInference.GT,
                                          columns=ClsInference.PRED,
                                          values='percentage').fillna('0')
        return confusion_matrix_table

