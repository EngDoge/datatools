import os
import pandas as pd

from datatools.utils.format_tools import PathFormatter


class ColorAnalyzer:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    @classmethod
    def from_file(cls, file_path):
        file_path = PathFormatter.format(file_path)
        data = pd.read_csv(file_path, dtype=str, index_col=0)
        return cls(data=data)

    def get_color_distribution_by_cls(self, defect_code):
        pass


