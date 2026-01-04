from .analyzer import *
from .dataset import *
from .fileio import *
from .image import *
from .manager import *
from .utils import *

import os
import configparser


def get_package_version(work_dir):
    config = configparser.ConfigParser()

    config_path = os.path.join(work_dir, "version.cfg")
    config.read(config_path)
    version_main = config.get("version", "update")
    return version_main


dir_path = os.path.dirname(os.path.realpath(__file__))
__version__ = get_package_version(work_dir=dir_path)

__all__ = ['RedetectAnalyzer', 'ClsEvaluation', 'ClsInference',
           'DataPatch', 'DataListGenerator', 'DataContainer', 'DataCluster', 'DataListParser',
           'SingleImage', 'ImageData',
           'ImageConvertor', 'MaskLayer', 'DefectObject', 'CompLayers',
           'PathFormatter', 'SuffixFormatter',
           'ProjectManager', 'ArchiveManager',
           'is_none', 'is_not_none', 'exists_or_make', 'select_in_col', 'group_count', 'check_img',
           "RunningMetrics",
           'extract_layer', 'extract_target_layers', 'ClassMapper',
           'AOICompMapper', 'AVICompMapper', 'AVICompDataMapper', 'ICSemanticMapper', 'DefectSegMapper'
           ]
