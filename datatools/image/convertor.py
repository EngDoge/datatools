import os
import cv2

import numpy as np
from tqdm import tqdm
from typing import Union, Optional


from datatools.image.mappers import ClassMapper
from datatools.dataset.container import DataContainer


class ImageConvertor:
    idx_map = np.zeros(16777216, dtype=np.uint32) - 1  # 256 * 256 * 256
    weights = np.array([65536, 256, 1], dtype=np.int32)  # 256 * 256, 256, 1

    def __init__(self,
                 data: Union[DataContainer, str],
                 color_map: Union[ClassMapper, dict],
                 ignore_ref: bool = True,
                 ignore_gerb: bool = True):

        if isinstance(data, str):
            data = DataContainer.from_scan_dir(src=data, ignore_ref=ignore_ref, ignore_gerb=ignore_gerb)

        assert isinstance(data, DataContainer), f'data must be a DataContainer, {type(data)} is given'

        if isinstance(color_map, ClassMapper):
            color_map = color_map.bgr_to_idx
        elif isinstance(color_map, dict):
            print('Caution: color_map should be in BGR color space!')

        self.data = DataContainer.merge_cluster(data, 'image')

        self.color_mapper = color_map

    @staticmethod
    def color2idx(img: np.ndarray, color_map: dict, require_color_in_mapper: bool = True):
        # img and color_map should be in the same color space
        img_shape = img.shape
        assert 3 <= len(img_shape) <= 4, f'img shape should be (H, W, C) or (N, H, W, C), got {img_shape}'
        img = img[..., :3]
        assert img_shape[-1] == 3, f'img given must be RGB or BGR, given channel number of {img_shape}'

        for color, idx in color_map.items():
            ImageConvertor.idx_map[np.dot(color, ImageConvertor.weights)] = idx
        img_id = ImageConvertor.idx_map[np.dot(img, ImageConvertor.weights)]
        if np.any(img_id == 4294967295) and require_color_in_mapper:
            raise ValueError(f'img has some colors not in color_map!')
        return np.uint8(img_id.copy())

    def __getitem__(self, item):
        img = self.data['image'][item]
        img.enable_single_image()
        img.mask.open_with_color()
        try:
            mask_id = self.color2idx(img.mask.image, self.color_mapper)
        except ValueError:
            raise ValueError(f'{img.path} has some colors not in color_map!')
        save_path = img.get_renamed_path('png', 'id')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask_id.astype(np.uint8))
        return 0

    def __len__(self):
        return self.data.total_num

    def convert(self, batch_size=6, num_workers=6):
        import torch
        from torch.utils.data import Dataset


        class Temp(Dataset, type(self)):
            def __getitem__(self, item):
                return super(Temp, self).__getitem__(item)


        temp_dataset = Temp(data=self.data, color_map=self.color_mapper)
        dataloader = torch.utils.data.DataLoader(temp_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers)
        pbar = tqdm([i for i in range(self.data.total_num // batch_size)])
        for _ in dataloader:
            pbar.update(1)


if __name__ == '__main__':
    from datatools import DataContainer, ICSemanticMapper
    dc = DataContainer.from_scan_dir(r'/data/dataset2/Workshop/wangyueyi/KSNY/'
                                     r'train_add_data/color/noref_semantic/20231007')
    convertor = ImageConvertor(dc, ICSemanticMapper)
    convertor.convert()

    # from datatools import SingleImage
    # import time
    # class_colors = [[0, 0, 0], [255, 182, 193], [220, 20, 60], [219, 112, 147], [255, 105, 180],
    #                 [199, 21, 133], [218, 112, 214], [216, 191, 216], [221, 160, 221], [255, 0, 255],
    #                 [128, 0, 128], [75, 0, 130], [138, 43, 226], [123, 104, 238], [230, 230, 250],
    #                 [0, 0, 255], [25, 25, 112], [65, 105, 225], [176, 196, 222], [119, 136, 153],
    #                 [155, 42, 42], [200, 0, 0], [30, 144, 255], [70, 130, 180], [0, 191, 255],
    #                 [95, 158, 160], [175, 238, 238], [0, 255, 255], [0, 206, 209], [0, 128, 128],
    #                 [127, 255, 170], [0, 250, 154], [0, 255, 127], [60, 179, 113], [143, 188, 143],
    #                 [0, 100, 0], [124, 252, 0], [173, 255, 47], [85, 107, 47], [245, 245, 220],
    #                 [255, 255, 0], [189, 183, 107], [255, 250, 205], [240, 230, 140], [255, 215, 0],
    #                 [255, 228, 181], [255, 165, 0], [222, 184, 135], [205, 133, 63], [255, 218, 185],
    #                 [139, 69, 19], [255, 160, 122], [255, 69, 0], [255, 228, 225], [250, 128, 114]]
    # class_colors = [[0, 0, 0], [220, 20, 60], [219, 112, 147], [255, 105, 180],
    #                 [199, 21, 133], [218, 112, 214], [216, 191, 216], [221, 160, 221], [255, 0, 255],
    #                 [128, 0, 128], [75, 0, 130], [138, 43, 226], [123, 104, 238], [230, 230, 250],
    #                 [0, 0, 255], [25, 25, 112], [65, 105, 225], [176, 196, 222], [119, 136, 153],
    #                 [155, 42, 42], [200, 0, 0], [30, 144, 255], [70, 130, 180], [0, 191, 255],
    #                 [95, 158, 160], [175, 238, 238], [0, 255, 255], [0, 206, 209], [0, 128, 128],
    #                 [127, 255, 170], [0, 250, 154], [0, 255, 127], [60, 179, 113], [143, 188, 143],
    #                 [0, 100, 0], [124, 252, 0], [173, 255, 47], [85, 107, 47], [245, 245, 220],
    #                 [255, 255, 0], [189, 183, 107], [255, 250, 205], [240, 230, 140], [255, 215, 0],
    #                 [255, 228, 181], [255, 165, 0], [222, 184, 135], [205, 133, 63], [255, 218, 185],
    #                 [139, 69, 19], [255, 160, 122], [255, 69, 0], [255, 228, 225], [250, 128, 114]]
    # color_map_ = {tuple(color[::-1]): idx for idx, color in enumerate(class_colors)}
    # mask_path = "/data/dataset2/Workshop/wangyueyi/KSNY/train_add_data/color/noref_semantic/20231007/test/Mask" \
    #             "/N39T10210011U4505_(02453,03134)_008_T_(010,001)_CUMC01C-CO_044_39_PAD_mask.png"
    #
    # img = SingleImage(mask_path)
    # # img.show('mask')
    # data = img.image
    # tic = time.time()
    # iterate = 1000
    # for i in range(iterate):
    # # img.apply(ImageConvertor.color2idx, args=(color_map_,), show=False, wait_key=True, is_binary=True, name='id')
    #     x = ImageConvertorTest.color2idx(data, color_map_)
    # print((time.time() - tic) / iterate)
