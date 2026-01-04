import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import reduce, partial

from abc import ABC
from typing import Optional, Callable, List, Dict


from datatools.utils import PathFormatter
from datatools.dataset.container import DataContainer
from datatools.image.mappers import ClassMapper
from datatools.image.data import ImageData
from datatools.image.layer import CompLayers, DefectObject


class DatasetAnalyzer(ABC):
    ANALYSIS_MODE = {}

    def __init__(self, data_container: 'DataContainer', **kwargs):
        self.dataset = data_container
        self.dataset_info = None
        assert self.ANALYSIS_MODE, 'ANALYSIS_MODE should be defined in subclass!'

    def export_dataset_info(self, save_path: str):
        if self.dataset_info is None:
            self.get_dataset_info()
        save_path = PathFormatter.format(save_path)
        save_path = save_path if save_path.endswith('.csv') else save_path + '.csv'
        self.dataset_info.to_csv(save_path)

    def get_dataset_info(self,
                         num_workers: int = 8,
                         parse_func: Optional[Callable] = None) -> pd.DataFrame:
        parse_func = self._parse_datacontainer if parse_func is None else parse_func
        result = []
        pbar = tqdm(total=self.dataset.total_num)
        with multiprocessing.Pool(processes=num_workers) as pool:
            for cluster in self.dataset:
                num = len(self.dataset[cluster])
                cluster_result: List = pool.map(parse_func, self.dataset[cluster])
                cluster_result: Dict = reduce(self.reduction, cluster_result)
                pbar.update(num)
                cluster_result['cluster'] = [cluster] * num
                result.append(cluster_result)
        dataset_info = pd.DataFrame(reduce(self.reduction, result))
        dataset_info['cluster'].astype(str)
        self.dataset_info = dataset_info.set_index('file_path')
        return self.dataset_info

    @staticmethod
    def reduction(x, y):
        raise NotImplementedError('This method should be implemented in subclass!')

    def _parse_datacontainer(self, img_data: 'ImageData'):
        raise NotImplementedError('This method should be implemented in subclass!')


class CompsegDatasetAnalyzer(DatasetAnalyzer):
    ANALYSIS_MODE = {
        'region_area': 'get_area_for_each_region',
    }

    def __init__(self,
                 data_container: 'DataContainer',
                 color_mapper: 'ClassMapper',
                 mode: str,
                 **kwargs,):
        assert mode in self.ANALYSIS_MODE, f'mode should be one of {self.ANALYSIS_MODE.keys()}, got {mode}'
        super(CompsegDatasetAnalyzer, self).__init__(data_container)
        self.color_mapper = color_mapper
        self.mode = mode
        self._get_layer_info_fn = getattr(self, self.ANALYSIS_MODE[self.mode])

    @staticmethod
    def reduction(x, y):
        ret = dict()
        for key in x.keys():
            ret[key] = x[key] + y[key]
        return ret

    @staticmethod
    def get_area_for_each_region(layer):
        return [region.area for region in layer]

    def _parse_datacontainer(self, img_data: 'ImageData'):
        img_data.disable_single_image()
        ret = dict()
        comp_layers = CompLayers.from_path(src=img_data.mask, color_mapper=self.color_mapper)
        for layer_name, layer in comp_layers:
            layer_info = self._get_layer_info_fn(layer)
            ret[layer_name] = [layer_info]
        ret['file_path'] = [img_data.cur]
        return ret


class ICSegNorefDatasetAnalyzer(DatasetAnalyzer):
    ANALYSIS_MODE = {
        'region_area': 'get_area_for_each_region',
    }

    def __init__(self,
                 data_container: 'DataContainer',
                 color_mapper: 'ClassMapper',
                 mode: str,
                 **kwargs,):
        assert mode in self.ANALYSIS_MODE, f'mode should be one of {self.ANALYSIS_MODE.keys()}, got {mode}'
        super(ICSegNorefDatasetAnalyzer, self).__init__(data_container)
        self.color_mapper = color_mapper
        self.mode = mode
        self._get_layer_info_fn = getattr(self, self.ANALYSIS_MODE[self.mode])

    @staticmethod
    def get_area_for_each_region(layer):
        return [region.area for region in layer]

    @staticmethod
    def reduction(x, y):
        ret = dict()
        for key in x.keys():
            ret[key] = x[key] + y[key]
        return ret

    def _parse_datacontainer(self, img_data: 'ImageData'):
        img_data.disable_single_image()
        ret = {defect_name: [0] for defect_name in self.color_mapper.classes if defect_name != '000'}
        if img_data.mask is not None:
            defect_obj = DefectObject.from_path(src=img_data.mask, color_mapper=self.color_mapper)
            for defect in defect_obj:
                defect_info = self._get_layer_info_fn(defect)
                ret[defect.defect_code] = [defect_info]
        ret['file_path'] = [img_data.cur]
        return ret


if __name__ == '__main__':
    from datatools import AVICompMapper, ICSemanticMapper
    import time
    dc = DataContainer.from_scan_dir(r"\data\dataset2\TrainData\AVI\ic-color\seg_noref\KSNY\NG_20231013#17808",
                                     ignore_ref=False, ignore_gerb=False)
    # analyzer = CompsegDatasetAnalyzer(data_container=dc, color_mapper=AVICompMapper, mode='region_area')
    color_mapper = {
        '000': [(0, 0, 0)],
        '0131': [(0, 250, 154)],
    }
    # color_mapper = None
    analyzer = ICSegNorefDatasetAnalyzer(data_container=dc,
                                         color_mapper=ICSemanticMapper.update_mapper(custom_palette=color_mapper),
                                         mode='region_area')
    s = time.time()
    info = analyzer.get_dataset_info()
    print((time.time() - s) / dc.total_num)
    analyzer.export_dataset_info(r'\data\dataset2\TrainData\AVI\ic-color\seg_noref\seg_noref_test.csv')



