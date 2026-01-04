import cv2
import random
import numbers
import numpy as np
from enum import Enum
from PIL import Image
from typing import Union, Tuple


class Transform(Enum):
    BILINEAR = 0
    NEAREST = 1
    BICUBIC = 2
    LANCZOS = 3
    AREA = 4


Transform.BILINEAR.PIL = Image.BILINEAR
Transform.NEAREST.PIL = Image.NEAREST
Transform.BICUBIC.PIL = Image.BICUBIC
Transform.LANCZOS.PIL = Image.LANCZOS


Transform.BILINEAR.CV2 = cv2.INTER_LINEAR
Transform.NEAREST.CV2 = cv2.INTER_NEAREST
Transform.BICUBIC.CV2 = cv2.INTER_CUBIC
Transform.LANCZOS.CV2 = cv2.INTER_LANCZOS4
Transform.AREA.CV2 = cv2.INTER_AREA




class RandomMoveCenterCrop:
    """
    Only Crop Image smaller than the target size to its target size.
    """
    def __init__(self,
                 size: Union[Tuple, numbers.Number] = 160,
                 random_move: Union[Tuple, numbers.Number] = 2):

        if isinstance(size, numbers.Number):
            size = (int(size), int(size))

        if isinstance(random_move, numbers.Number):
            random_move = (int(random_move), int(random_move))

        self.size = size
        self.random_move = random_move

    def _get_min_coord(self, width: int, height: int, target_w: int, target_h: int):
        if self.random_move is None:
            delta_w = 0
            delta_h = 0
        else:
            delta_w = random.randint(-self.random_move[0], self.random_move[0])
            delta_h = random.randint(-self.random_move[1], self.random_move[1])

        w_min = max(0, int(round((width - target_w) / 2.)) + delta_w)
        h_min = max(0, int(round((height - target_h) / 2.)) + delta_h)

        return w_min, h_min

    def _crop_img_numpy(self, img, target_h, target_w):
        img_shape = img.shape
        num_dim = len(img_shape)
        assert num_dim in [2, 3], f"Dimension of image must be 2 or 3, shape of {img_shape} is given!"
        height, width = img_shape[0], img_shape[1]
        if width > target_w or height > target_h:
            w_min, h_min = self._get_min_coord(width, height, target_w, target_h)
            if num_dim == 3:
                img = img[h_min: (h_min + target_h), w_min: (w_min + target_w), :]
            else:
                img = img[h_min: (h_min + target_h), w_min: (w_min + target_w)]
            img_shape = img.shape
            height, width = img_shape[0], img_shape[1]
        return img, height, width

    def _crop_img_pillow(self, img, target_h, target_w):
        width, height = img.size
        if width > target_w or height > target_h:
            w_min, h_min = self._get_min_coord(width, height, target_w, target_h)
            img = img.crop((w_min, h_min, w_min + target_w, h_min + target_h))
            width, height = img.size
        return img, height, width

    def transform(self, img):
        target_h, target_w = self.size
        if isinstance(img, np.ndarray):
            crop_fn = self._crop_img_numpy
        elif isinstance(img, Image.Image):
            crop_fn = self._crop_img_pillow
        else:
            raise NotImplementedError(f"Only np.ndarray and PIL.Image, {type(img)} is given!")
        return crop_fn(img, target_h, target_w)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'random_move={self.random_move})'
        return repr_str


class Resize:
    """
    Only Resize Image smaller than the target size to its target size.
    """
    def __init__(self,
                 size: Union[Tuple, numbers.Number] = 160,
                 interpolate: str = 'BILINEAR',
                 zoom_out: bool = False):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))

        self.size = size
        self.zoom_out = zoom_out
        self.flag = getattr(Transform, interpolate.upper())

    def _resize_img_numpy(self, img, target_h, target_w):
        img_shape = img.shape
        assert len(img_shape) in [2, 3], f"Dimension of image must be 2 or 3, shape of {img_shape} is given!"
        height, width = img_shape[0], img_shape[1]
        if width < target_w or height < target_h or self.zoom_out:
            img = cv2.resize(img, self.size, interpolation=self.flag.CV2)
            img_shape = img.shape
            height, width = img_shape[0], img_shape[1]
        return img, height, width

    def _resize_img_pillow(self, img, target_h, target_w):
        width, height = img.size
        if width < target_w or height < target_h or self.zoom_out:
            img = img.resize(size=self.size, resample=self.flag.PIL)
            width, height = img.size
        return img, height, width

    def transform(self, img):
        target_h, target_w = self.size
        if isinstance(img, np.ndarray):
            resize_fn = self._resize_img_numpy
        elif isinstance(img, Image.Image):
            resize_fn = self._resize_img_pillow
        else:
            raise NotImplementedError(f"Only np.ndarray and PIL.Image, {type(img)} is given!")
        return resize_fn(img, target_h, target_w)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolate={self.flag.name})'
        return repr_str


class Pad:
    def __init__(self,
                 size: Union[Tuple, numbers.Number] = 160,):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))

        self.size = size




if __name__ == '__main__':
    from datatools import SingleImage
    xxx = SingleImage(r"\data\data_cold\Workshop\general_InnerLayer\defect_cls\merged\002\Gerb\388-20201216-73483-43-669193730901_C253823C_C253823-C-L5_101_2_004_gerb.png",
                      backend='cv2')
    trans = RandomMoveCenterCrop(size=(64, 320))
    xxx.show('ori')
    xxx.apply(trans.transform, show=True, wait_key=True)

