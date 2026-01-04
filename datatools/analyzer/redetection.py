import os
import pandas as pd

from datatools.utils import PathFormatter
from datatools.image.data import ImageData
from datatools.dataset.datapatch import DataPatch
from datatools.analyzer.utils import select_in_col




class RedetectAnalyzer:
    TEMP_ROOT = '/data/dataset2/Workshop/sunjianyao'

    def __init__(self, data_root, redetect_root, project, redetect_result=None):
        self.data_root = PathFormatter.format(data_root)
        self.redetect_root = PathFormatter.format(redetect_root)
        self.project = project
        if redetect_result is not None:
            self.redetect_result = redetect_result
        else:
            self.redetect_result = RedetectAnalyzer.get_redetect_result(self.redetect_root)

    @classmethod
    def set_redetect_trail(cls, data_root=None, project=None, data_name=None, trial=None, config=None):
        if config is not None:
            data_root = PathFormatter.format(config.pop('data_root'))
            data_name = config.pop('data_name')
            project = config.pop('project')
            trial = config.pop('trial')

        if trial == 'single':
            redetect_root = os.path.join(data_root, project, 'redetect_result', trial)
        else:
            redetect_root = os.path.join(data_root, project, 'redetect_result', data_name, trial)
        data_root = os.path.join(data_root, project, 'data_history', data_name)
        redetect_result = RedetectAnalyzer.get_redetect_result(redetect_root)
        return cls(redetect_result=redetect_result, data_root=data_root, project=project, redetect_root=redetect_root)

    @property
    def fallout(self):
        ai_result = self.redetect_result.copy()
        fallout = ai_result[(ai_result['postproc_result'] == 1) & (ai_result['vrs_result'] == 0)]
        return fallout

    @property
    def high_risk_fallout(self):
        ai_result = self.redetect_result.copy()
        high_risk_fallout = ai_result[(ai_result['postproc_result'] == 1) &
                                      (ai_result['vrs_result'] == 0) &
                                      (ai_result['ai_result'] == 0) &
                                      (ai_result['noref'] == 0)]
        return high_risk_fallout

    @property
    def noref_fallout(self):
        ai_result = self.redetect_result.copy()
        high_risk_fallout = ai_result[(ai_result['postproc_result'] == 1) &
                                      (ai_result['vrs_result'] == 0) &
                                      (ai_result['ai_result'] == 0) &
                                      (ai_result['noref'] == 1)]
        return high_risk_fallout

    @property
    def normal_fallout(self):
        ai_result = self.redetect_result.copy()
        normal_fallout = ai_result[(ai_result['postproc_result'] == 1) &
                                      (ai_result['vrs_result'] == 0) &
                                      (ai_result['ai_result'] == 1)]
        return normal_fallout

    @property
    def correct(self):
        ai_result = self.redetect_result.copy()
        correct = ai_result[((ai_result['postproc_result'] == 1) & (ai_result['vrs_result'] == 1)) |
                            ((ai_result['postproc_result'] == 0) & (ai_result['vrs_result'] == 0))]
        return correct

    @property
    def loss(self):
        ai_result = self.redetect_result.copy()
        loss = ai_result[(ai_result['postproc_result'] == 0) & (ai_result['vrs_result'] == 1)]
        return loss

    @property
    def model_loss(self):
        ai_result = self.redetect_result.copy()
        model_loss = ai_result[(ai_result['postproc_result'] == 0) &
                               (ai_result['vrs_result'] == 1) &
                               (ai_result['ai_result'] == 0)]
        return model_loss

    @property
    def normal_loss(self):
        ai_result = self.redetect_result.copy()
        normal_loss = ai_result[(ai_result['postproc_result'] == 0) &
                                (ai_result['vrs_result'] == 1) &
                                (ai_result['ai_result'] == 1)]
        return normal_loss

    def temp_save_folder(self, folder_name='temp'):
        temp = os.path.join(RedetectAnalyzer.TEMP_ROOT, self.project, 'train_data', 'unlabelled', folder_name)
        print('Temp save to:', temp)
        return PathFormatter.format(temp)

    @staticmethod
    def get_redetect_result(redetect_root):
        ai_result_file = os.path.join(redetect_root, 'ai_results.txt')
        postproc_result = os.path.join(redetect_root, 'postproc_results.txt')
        ai_results = pd.read_csv(ai_result_file, dtype={'defect_code': 'str'})
        postproc_results = pd.read_csv(postproc_result)
        redetect_result = RedetectAnalyzer.merge_results(ai_results=ai_results, postproc_results=postproc_results)
        return redetect_result

    @staticmethod
    def merge_results(ai_results, postproc_results):
        ai_results['postproc_result'] = ai_results['ori'].map(dict(zip(postproc_results['image_name'],
                                                                       postproc_results['postproc_result'])))
        ai_results['online_ai_result'] = ai_results['ori'].map(dict(zip(postproc_results['image_name'],
                                                                        postproc_results['online_ai_result'])))
        return ai_results

    @staticmethod
    def check_selected_img(selected, show_mask=False, ref='gerb'):
        for cur in selected['Cur']:
            RedetectAnalyzer.check_img(img_data_path=cur, show_mask=show_mask, ref=ref)


    @staticmethod
    def check_img(img_data_path, show_mask=False, ref='gerb'):
        assert ref in ['ref', 'gerb', 'Ref', 'Gerb'], "ref must be 'ref' or 'gerb'"
        img_data_path = PathFormatter.format(img_data_path)
        img = ImageData(img_data_path)
        print('Checking: ', img.name)
        if ref in ['ref', 'Ref']:
            ref_img = img.ref
        else:
            ref_img = img.gerb
        if show_mask:
            img.mask.show(named_window='Mask', is_binary=True)
        img.cur.show(named_window='Cur')
        ref_img.show(named_window=ref.capitalize(), wait_key=True, destroy_window=False)

    @staticmethod
    def groupby_distribution(src, by=None, sort_values=True, show_ratio=True):
        if by is not None:
            src = src.groupby(by=by).count()
            if sort_values:
                src.sort_values(by=['Cur'], ascending=False, inplace=True)
            drop_cols = list(src.columns)
            drop_cols.remove('Cur')
            src.drop(labels=drop_cols, axis=1, inplace=True)
            if show_ratio:
                src['percentage'] = src['Cur'] / src['Cur'].sum() * 100
                src['cumsum'] = src['percentage'].cumsum()
        return src

    @staticmethod
    def select_defect_and_zone(src, defect_code=None, zone_code=None, drop_duplicates=False):
        if defect_code is not None:
            src = select_in_col(src, 'defect_code', defect_code)
        if zone_code is not None:
            src = select_in_col(src, 'zone_code', zone_code)
        if drop_duplicates:
            src = src.drop_duplicates('ori', keep='first')
        return src

    @staticmethod
    def get_vrs_label(vrs_data_root, dtype='dict'):
        vrs_data_root = PathFormatter.format(vrs_data_root)
        vrs_data_patch = DataPatch(vrs_data_root)
        vrs_label = vrs_data_patch.get_vrs_labels(dtype=dtype)
        return vrs_label

    @staticmethod
    def extract_selected_to(selected, dst, force=False):
        assert isinstance(selected, pd.DataFrame) and 'Cur' in selected
        for cur in selected['Cur']:
            img = ImageData(cur)
            img.copy_to(dst, force=force)

    def extract_ori_file(self, selected, dst, force=False):
        assert isinstance(selected, pd.DataFrame) and 'ori' in selected
        cur_roots = []
        for root, _, _ in os.walk(self.data_root):
            if 'Cur' in root:
                cur_roots.append(root)
        if len(cur_roots) > 0:
            for ori_name in selected.drop_duplicates(['ori'], keep='first')['ori']:
                for cur_root in cur_roots:
                    cur = os.path.join(cur_root, ori_name)
                    if os.path.exists(cur):
                        img = ImageData(cur)
                        img.copy_to(dst, force=force)
                        break

        print('> Finished')

    def match_vrs_label(self, df, img_name_col='ori'):
        df = df.copy()
        vrs_data_root = self.data_root
        vrs_label = RedetectAnalyzer.get_vrs_label(vrs_data_root, dtype='dict')
        df['vrs_label'] = df[img_name_col].map(vrs_label)
        return df


