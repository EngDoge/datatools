from typing import Optional, List, NoReturn, Union

import pandas as pd

from datatools.image.data import ImageData
from datatools.utils.format_tools import PathFormatter

CUR_COL = 'Cur'
CLUSTER_COL = 'Cluster'
GERBER_COL = 'Gerb'
MASK_COL = 'Mask'
NUM_COL = 'Num'
SHAPE_COL = 'Shape'


def select_in_col(src: pd.DataFrame, col: str, condition: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(condition, str):
        condition = [condition]
    else:
        assert isinstance(condition, list) or isinstance(condition, pd.Series), \
            'condition given must be a list, str or pandas.Series!'
    return src[src[col].isin(condition)]


def group_count(src: pd.DataFrame, cols: Union[str, List[str]]) -> pd.DataFrame:
    assert CUR_COL in src.columns, f"Column '{CUR_COL}' is required!"
    return src.rename(columns={CUR_COL: NUM_COL}).groupby(cols).count()[NUM_COL]


def check_img(data_path: str,
              show_mask: bool = False,
              ref: Optional[str] = 'gerb',
              is_binary: bool = True,
              destroy_window: bool = True) -> NoReturn:
    data_path = PathFormatter.format(data_path)
    img = ImageData(data_path)
    if isinstance(ref, str):
        assert ref in ['ref', 'gerb', 'Ref', 'Gerb'], "ref must be 'ref' or 'gerb'"
        ref_img = img.ref if ref in ['ref', 'Ref'] else img.gerb
        ref = ref.capitalize()
        try:
            ref_img.show(named_window=ref)
        except AttributeError:
            print(f'{ref} of image is not found: {data_path}')
    if show_mask:
        img.mask.show(named_window='Mask', is_binary=is_binary)
    img.cur.show(named_window='Cur', wait_key=True, destroy_window=destroy_window)


def exclude_by_condition(src, col: str, exceptions: List[str]):
    exclusion = ','.join(['"' + exclude + '"' for exclude in exceptions])
    condition = f'{col} not in ({exclusion})'
    return src.query(condition).copy()


def groupby_counts(src, cols):
    if isinstance(cols, str):
        cols = [cols]
    return src[cols+[CUR_COL]].rename(columns={CUR_COL: NUM_COL}).groupby(cols).count()


