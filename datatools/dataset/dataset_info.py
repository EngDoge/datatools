import json
import numpy as np
import os.path as osp


class DatasetInfo:

    @staticmethod
    def parse(dataset_info):
        """Parse dataset info and return a dict of info."""
        with open(osp.join(dataset_info, 'dataset_info.json'), 'r') as f:
            data = json.load(f)
        top_labels = list(data['top_label_weights'].keys())
        top_cw = np.array(list(data['top_label_weights'].values()))
        return top_labels, top_cw
