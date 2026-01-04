import cv2
import numpy as np
import os.path as osp

# import warnings
from skimage.measure import label, regionprops
from typing import List, Dict, Tuple, Union, Iterator, Optional

from datatools.image.data import SingleImage
from datatools.image.mappers import ClassMapper
from datatools.image.convertor import ImageConvertor
from datatools.image.utils import extract_target_layers


def on_change(func, record=False):
    def wrapper(*args, **kwargs):
        args[0].clear()
        ret = func(*args, **kwargs)
        args[0].pop_empty_region()
        if record:
            raise NotImplementedError("Record is not implemented yet")
        return ret

    return wrapper


def cascade(func):
    def cascade_parent(*args, **kwargs):
        while args[0].has_parent:
            parent = args[0].parent
            non_cascade_call = getattr(parent, "non_cascade_call")
            non_cascade_call(f'_{func.__name__}', *args[1:], **kwargs)
            args = (parent, *args[1:])

    def cascade_child(*args, **kwargs):
        for child in args[0]:
            if child.has_children:
                cascade_child(child, *args[1:], **kwargs)
            non_cascade_call = getattr(child, "non_cascade_call")
            non_cascade_call(f'_{func.__name__}', *args[1:], **kwargs)

    def wrapper(*args, **kwargs):
        if args[0].has_children:
            cascade_child(*args, **kwargs)
        func(*args, **kwargs)
        cascade_parent(*args, **kwargs)
        # func(*args, **kwargs)
        # cascade_parent(*args, **kwargs)
        # if args[0].has_children:
        #     cascade_child(*args, **kwargs)

    return wrapper


class MaskLayer:
    __slots__ = ['data', 'parent', 'name', 'id', 'bbox', '_regions', '_area', '_shape']
    LAZY_ATTRIBUTES = ['_area']
    """
    Binary layer with 2D data
    """

    def __init__(self,
                 data: np.ndarray,
                 name='MaskLayer',
                 parent=None,
                 bbox=None,
                 idx=255, ):
        """

        Parameters:
            data: [np.ndarray] 2D binary data, shape=(H, W), dtype=bool
            name: [str] name of the layer
            parent: [MaskLayer] parent layer
            bbox: [tuple] bounding box of the layer, (h_min, w_min, h_max, w_max)
            idx: [int] index of the layer, range from 0 to 255
        """
        self.id = idx
        self.data = data
        self.bbox = bbox
        self.name = name
        self.parent = parent

        self._area = None
        self._shape = None
        self._regions = None

    @classmethod
    def from_SingleImage(cls,
                         single_image: 'SingleImage',
                         color_mapper: Optional['ClassMapper'] = None,
                         color_space: str = 'BGR',
                         name: Optional[str] = None,
                         *args,
                         **kwargs):
        """
        Convert SingleImage to MaskLayer

        the single_image.image should be binary image with shape (H, W)

        Parameters:
            single_image: [SingleImage] SingleImage instance
            name: [str] name of the layer
            color_mapper: [ClassMapper] color mapper
            color_space: [str] color space of the image, default to be BGR
        """

        data = np.zeros_like(single_image.image, dtype=bool)
        data[single_image.image.astype(bool)] = True
        return cls(data=data,
                   name=osp.splitext(single_image.name)[0] if name is None else name)

    @classmethod
    def from_path(cls,
                  src: str,
                  color_mapper: Optional['ClassMapper'] = None,
                  name: Optional[str] = None,
                  *args, **kwargs):
        """
        Convert image path to MaskLayer

        the image should be binary image with shape (H, W), such as "Id" images

        Parameters:
            src: [str] path to the image
            name: [str] name of the layer
            color_mapper: No use
        """
        return cls.from_SingleImage(SingleImage(src=src), name=name)

    @classmethod
    def from_bbox_list(cls,
                       bbox_list: List[List[int]],
                       shape: Tuple[int, int],
                       bbox_limit: int = 12,
                       reverse_hw: bool = False) -> 'MaskLayer':
        """
        Generate a mask layer from a list of bbox

        Parameters:
            bbox_list: list of [h_min, w_min, h_max, w_max] or [w_min, h_min, w_max, h_max]
            shape: [H, W]
            bbox_limit: int
            reverse_hw: bbox sequence is [h_min, w_min, h_max, w_max] (False) or [w_min, h_min, w_max, h_max] (True)
        """
        mask = np.zeros(shape=shape, dtype=bool)
        for bbox in bbox_list:
            if reverse_hw:
                w_min, h_min, w_max, h_max = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                h_min, w_min, h_max, w_max = bbox[0], bbox[1], bbox[2], bbox[3]
            bbox_h, bbox_w = h_max - h_min, w_max - w_min
            center_w, center_h = w_min + (bbox_w // 2), h_min + (bbox_h // 2)

            box_w = max(bbox_limit, bbox_w)  # limit [32, crop_size]
            box_w = min(box_w, shape[1] - 1)
            box_h = max(bbox_limit, bbox_h)  # limit [32, crop_size]
            box_h = min(box_h, shape[0] - 1)
            w_min, h_min = center_w - (box_w // 2), center_h - (box_h // 2)
            w_max, h_max = center_w + (box_w // 2), center_h + (box_h // 2)
            w_min, h_min = max(0, w_min), max(0, h_min)
            w_max, h_max = min(w_max, shape[1] - 1), min(h_max, shape[0] - 1)
            mask[h_min:h_max, w_min:w_max] = 1
        return cls(data=mask, name="bbox_mask")

    @property
    def regions(self) -> List['MaskLayer']:
        """
        Get all regions in the layer
        Returns a list of MaskLayer if the layer has children, otherwise return an empty list.
        """
        if self._regions is None:
            regions = regionprops(label(self.data.astype(np.uint8)))
            # num_regions = len(regions)
            if (num_regions := len(regions)) >= 1:
                # self._regions = [self._get_region_layer(region=region,
                #                                         name=f'{self.name}[{i}]')
                #                  for i, region in enumerate(regions)]
                if num_regions == 1:
                    self._regions = [self]
                    self.bbox = regions[0].bbox

                elif num_regions > 1:
                    self._regions = [self._get_region_layer(region=region,
                                                            name=f'{self.name}[{i}]')
                                     for i, region in enumerate(regions)]
            else:
                self._regions = []

            # self._regions = [] \
            #     if len(regions) == 0 \
            #     else [self._get_region_layer(region=region, name=f'{self.name}[{i}]')
            #           for i, region in enumerate(regions)]

        return self._regions

    @property
    def num_regions(self) -> int:
        """
        Get number of regions in the layer
        """
        return len(self.regions)

    @property
    def has_children(self) -> bool:
        """
        Check if the layer has children
        """
        return bool(self.regions) and self not in self.regions

    @property
    def has_parent(self) -> bool:
        """
        Check if the layer has parent
        """
        return self.parent is not None

    @property
    def area(self) -> int:
        """
        Get area of the layer
        Lazy attribute will update when the layer is changed
        """
        if self._area is None:
            self._area = np.sum(self.data)
        return self._area

    @property
    def area_ratio(self) -> float:
        return self.area / (self.img_h * self.img_w)

    @property
    def mask(self) -> np.ndarray:
        """
        Get mask of the layer, with its origional index, default to be 255
        """
        mask = np.zeros_like(self.data, dtype=np.uint8)
        mask[self.data] = self.id
        return mask

    @property
    def binary_mask(self) -> np.ndarray:
        """
        Get the binary mask of the layer
        """
        mask = np.zeros_like(self.data, dtype=np.uint8)
        mask[self.data] = 255
        return mask

    @property
    def is_empty(self) -> bool:
        return not np.any(self.data)

    @property
    def is_not_empty(self) -> bool:
        return np.any(self.data)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get shape of the layer, shape in (H, W)
        """
        if self._shape is None:
            self._shape = self.data.shape
        return self._shape

    @property
    def img_h(self) -> int:
        """
        Get height of the image
        """
        return self.shape[0]

    @property
    def img_w(self) -> int:
        """
        Get width of the image
        """
        return self.shape[1]

    @property
    def h_min(self):
        try:
            return self.bbox[0]
        except Exception:
            return None

    @property
    def w_min(self):
        try:
            return self.bbox[1]
        except Exception:
            return None

    @property
    def h_max(self):
        try:
            return self.bbox[2]
        except Exception:
            return None

    @property
    def w_max(self):
        try:
            return self.bbox[3]
        except Exception:
            return None

    @property
    def h(self) -> int:
        """
        Get height of the mask area
        """
        if self.bbox is None:
            return self.data.shape[0]
        return self.bbox[2] - self.bbox[0]

    @property
    def w(self) -> int:
        """
        Get width of the mask area
        """
        if self.bbox is None:
            return self.data.shape[1]
        return self.bbox[3] - self.bbox[1]

    @property
    def min_hw(self) -> int:
        """
        Get min height and width of the mask area
        """
        return min(self.h, self.w)

    @property
    def max_hw(self) -> int:
        """
        Get max height and width of the mask area
        """
        return max(self.h, self.w)

    def erode(self, kernel_size: int = 3, iterations: int = 1) -> 'MaskLayer':
        """
        Erode the layer

        Parameters:
        -----------
        kernal_size [int]: size of the kernal
        iterations [int]: number of iterations
        """
        return MaskLayer(data=cv2.erode(np.uint8(self.data),
                                        np.ones((kernel_size, kernel_size), np.uint8),
                                        iterations=iterations).astype(bool),
                         name=f"Erode({self.name}, kernel={kernel_size}, iter={iterations})",
                         idx=self.id)

    def dilate(self, kernel_size: int = 3, iterations: int = 1) -> 'MaskLayer':
        """
        Dilate the layer

        Parameters:
        -----------
        kernal_size [int]: size of the kernal
        iterations [int]: number of iterations
        """

        return MaskLayer(data=cv2.dilate(np.uint8(self.data),
                                         np.ones((kernel_size, kernel_size), np.uint8),
                                         iterations=iterations).astype(bool),
                         name=f"Dilate({self.name}, kernel={kernel_size}, iter={iterations})",
                         idx=self.id)

    def morph_open(self, kernel_size: int = 3, iterations: int = 1) -> 'MaskLayer':
        """
        Morpholgy open operation to the layer

        Parameters:
        -----------
        kernal_size [int]: size of the kernal
        iterations [int]: number of iterations
        """
        return MaskLayer(data=cv2.morphologyEx(np.uint8(self.data),
                                               cv2.MORPH_OPEN,
                                               np.ones((kernel_size, kernel_size), np.uint8),
                                               iterations=iterations).astype(bool),
                         name=f"MorphOpen({self.name}, kernel={kernel_size}, iter={iterations})",
                         idx=self.id)

    def morph_close(self, kernel_size: int = 3, iterations: int = 1) -> 'MaskLayer':
        """
        Morpholgy close operation to the layer

        Parameters:
        -----------
        kernal_size [int]: size of the kernal
        iterations [int]: number of iterations
        """
        return MaskLayer(data=cv2.morphologyEx(np.uint8(self.data),
                                               cv2.MORPH_CLOSE,
                                               np.ones((kernel_size, kernel_size), np.uint8),
                                               iterations=iterations).astype(bool),
                         name=f"MorphClose({self.name}, kernel={kernel_size}, iter={iterations})",
                         idx=self.id)

    # def get_contour(self):
    #     contours, _ = cv2.findContours(np.uint8(self.data)), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     contour_length = [len(contour_set) for contour_set in contours]
    #     contour_idx = contour_length.index(max(contour_length))
    #     return np.array(contours[0][contour_idx]).squeeze()

    @staticmethod
    def _get_contours(image, external_only=True, all_contour_points=True) -> List:
        mode = cv2.RETR_EXTERNAL if external_only else cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_NONE if all_contour_points else cv2.CHAIN_APPROX_SIMPLE
        contours, _ = cv2.findContours(image, mode=mode, method=method)
        return contours

    def get_contour_coords(self, external_only=True, all_contour_points=True) -> List:
        contours = self._get_contours(np.uint8(self.data),
                                      external_only=external_only,
                                      all_contour_points=all_contour_points)
        return [contour.squeeze() for contour in contours]

    def get_contour_layer(self, external_only=True, all_contour_points=True) -> 'MaskLayer':
        contours = self._get_contours(np.uint8(self.data),
                                      external_only=external_only,
                                      all_contour_points=all_contour_points)
        contour_img = np.zeros_like(self.data, dtype=np.uint8)
        cv2.drawContours(contour_img, contours, -1, 255, 1)
        return MaskLayer(data=contour_img.astype(bool),
                         name=f'Contour({self.name})',
                         idx=self.id)

    def get_minAreaRect_hw(self,
                           min_hw: int = 224,
                           max_hw: int = 224) -> Tuple[int, int]:
        contours = self._get_contours(np.uint8(self.data),
                                      external_only=True,
                                      all_contour_points=True)
        if len(contours) == 1:
            contour = contours[0]
            rect_dft = cv2.minAreaRect(contour)
            min_hw, max_hw = min(rect_dft[1][0], rect_dft[1][1]), max(rect_dft[1][0], rect_dft[1][1])

        return min_hw, max_hw


    def copy(self, rename: bool = True) -> 'MaskLayer':
        """
        Copy the layer
        """
        return MaskLayer(data=self.data.copy(),
                         bbox=self.bbox,
                         name=f'Copy({self.name})' if rename else self.name,
                         idx=self.id)

    def select(self, bbox: Tuple[int, int, int, int]):
        """
        Select a region from the layer
        """
        w_min, h_min, w_max, h_max = bbox
        region_layer = np.zeros_like(self.data, dtype=bool)
        region_layer[h_min:h_max, w_min:w_max] = self.data[h_min:h_max, w_min:w_max]
        return MaskLayer(data=region_layer,
                         parent=self,
                         name=f'Select({self.name}, {bbox})',
                         bbox=(h_min, w_min, h_max, w_max),
                         idx=self.id)

    def show(self,
             name: Optional[str] = None,
             wait_key=False,
             destroy_window=False,
             show_binary_mask=True,
             allowed_keys: Union[str, List[str], None] = ' ', ) -> None:
        """
        Show the layer in a window

        Parameters:
        -----------
        name [str]: name of the window
        wait_key [bool]: whether to wait for a key to be pressed
        destroy_window [bool]: whether to destroy the window after the key is pressed
        show_binary_mask [bool]: whether to show the binary mask of the layer or the original index mask
        """
        name = self.name if name is None else name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(winname=name,
                   mat=self.binary_mask if show_binary_mask else self.mask)

        if wait_key:
            if allowed_keys is None:
                cv2.waitKey(0)
            while True:
                key = chr(cv2.waitKey(0))
                if key in allowed_keys:
                    break
            if destroy_window:
                cv2.destroyAllWindows()

    def clear(self) -> None:
        """
        Clear lazy attributes
        """
        for attr in self.LAZY_ATTRIBUTES:
            setattr(self, attr, None)

    def non_cascade_call(self, fn_name, *args, **kwargs):
        """
        Call a function without cascade

        Parameters:
        -----------
        fn_name [str]: name of the function
        *args: arguments of the function
        **kwargs: keyword arguments of the function
        """
        func = getattr(self, fn_name)
        func(*args, **kwargs)

    @on_change
    def _remove_by_bbox(self, h_min, w_min, h_max, w_max):
        """
        Remove the layer by bounding box
        Inner function, do not call directly

        Parameters:
        -----------
        h_min [int]: the minor value of the height of the bounding box
        w_min [int]: the minor value of the width of the bounding box
        h_max [int]: the greater value of the height of the bounding box
        w_max [int]: the greater value of the width of the bounding box
        """
        self.data[h_min:h_max, w_min:w_max] = 0

    @on_change
    def _remove_by_layer(self, layer: 'MaskLayer'):
        """
        Remove the layer by another layer
        Inner function, do not call directly

        Parameters:
        -----------
        layer [MaskLayer]: the layer to be removed
        """
        self.data[layer.data] = 0

    @cascade
    def remove_by_bbox(self, h_min, w_min, h_max, w_max):
        """
        Remove the layer by bounding box. Cascade call will remove the layer in its children and parent

        Parameters:
        -----------
        h_min [int]: the minor value of the height of the bounding box
        w_min [int]: the minor value of the width of the bounding box
        h_max [int]: the greater value of the height of the bounding box
        w_max [int]: the greater value of the width of the bounding box
        """
        self._remove_by_bbox(h_min, w_min, h_max, w_max)

    @cascade
    def remove_by_layer(self, layer: 'MaskLayer'):
        """
        Remove the layer by another layer. Cascade call will remove the layer in its children and parent

        Parameters:
        -----------
        layer [MaskLayer]: the layer to be removed
        """
        self._remove_by_layer(layer.copy())

    @on_change
    def pop(self, item):
        """
        Pop a region from the layer

        Parameters:
        -----------
        item [int]: index of the region
        """
        self.remove_by_layer(self.regions[item])
        return self._regions.pop(item)

    def pop_empty_region(self):
        """
        Pop empty regions from the layer
        """
        # if self.has_children:
        #     for i, region in enumerate(self.regions):
        #         if region.is_empty:
        #             self._regions.pop(i)
        for i, region in enumerate(self.regions):
            if region.is_empty:
                self._regions.pop(i)

    def _get_region_layer(self, region, name: str) -> 'MaskLayer':
        """
        Get a region layer from the layer
        """
        h_min, w_min, h_max, w_max = region.bbox
        region_layer = np.zeros_like(self.data, dtype=bool)
        region_layer[h_min:h_max, w_min:w_max] = np.array(region.image, dtype=bool)
        return MaskLayer(data=region_layer, parent=self, name=name, bbox=region.bbox, idx=self.id)

    def union_with(self, other: 'MaskLayer') -> 'MaskLayer':
        return self | other

    def intersect_with(self, other: 'MaskLayer') -> 'MaskLayer':
        return self & other

    def difference_with(self, other: 'MaskLayer') -> 'MaskLayer':
        return self ^ other

    def subtract(self, other: 'MaskLayer') -> 'MaskLayer':
        return self - other

    def invert(self) -> 'MaskLayer':
        return ~self

    def __add__(self, other: 'MaskLayer') -> 'MaskLayer':
        return self | other

    def __or__(self, other: 'MaskLayer') -> 'MaskLayer':
        return MaskLayer(data=np.bitwise_or(self.data, other.data),
                         name=f'Union({self.name}, {other.name})')

    def __sub__(self, other: 'MaskLayer') -> 'MaskLayer':
        return MaskLayer(data=np.bitwise_and(self.data, np.bitwise_not(other.data)),
                         name=f'Subtract({self.name}, {other.name})')

    def __and__(self, other: 'MaskLayer') -> 'MaskLayer':
        return MaskLayer(data=np.bitwise_and(self.data, other.data),
                         name=f'Intersect({self.name}, {other.name})')

    def __xor__(self, other: 'MaskLayer') -> 'MaskLayer':
        return MaskLayer(data=np.bitwise_xor(self.data, other.data),
                         name=f'Difference({self.name}, {other.name})')

    def __invert__(self) -> 'MaskLayer':
        return MaskLayer(data=np.bitwise_not(self.data),
                         name=f'Invert({self.name})')

    def __getitem__(self, item) -> 'MaskLayer':
        return self.regions[item]

    def __iter__(self) -> Iterator['MaskLayer']:
        return iter(self.regions) if self.has_children else iter([self])

    def __len__(self) -> int:
        return self.num_regions

    def __repr__(self):
        return f"MaskLayer(name={self.name})"


class DefectObject(MaskLayer):
    __slots__ = ['_regions', '_is_semantic']

    def __init__(self,
                 data: np.ndarray,
                 defect_codes: Union[str, Dict[str, int], List[str]],
                 ignore_defect_codes: Optional[List[str]] = None,
                 is_semantic: bool = False,
                 name: str = 'DefectObject'):
        """
        Parameters:
        -----------
        image [np.ndarray]: 2D binary image, shape=(H, W), dtype=bool, the index on the image should be the defect idx
        defect_codes [str, dict]: defect codes, if is_semantic is True, defect_codes should be a dict, str otherwise
        ignore_defect_codes [List[str]]: defect codes to be ignored, do not record as a defect layer
        is_semantic [bool]: whether the image contains multiply defect idx or is only binary segmentation
        """
        assert len(data.shape) == 2, f'Image must be a 2D array, {data.shape} is given'
        super(DefectObject, self).__init__(data=data.astype(bool),
                                           name=name)
        self._is_semantic = is_semantic

        self._build_defect_regions(data=data,
                                   defect_codes=defect_codes,
                                   ignore_defect_codes=[] if ignore_defect_codes is None else ignore_defect_codes,
                                   is_semantic=is_semantic)

    @classmethod
    def from_path(cls,
                  src: str,
                  color_mapper: 'ClassMapper' = None,
                  name: str = 'DefectObject',
                  ignore_defect_codes: Optional[List[str]] = None,
                  defect_codes: Union[str, Dict, None] = None,
                  require_color_in_mapper: bool = True,
                  is_semantic: bool = False,
                  *args, **kwargs):
        assert color_mapper is not None, f'color_mapper must be given, {color_mapper} is given!'
        assert isinstance(src, str), f"Input path should be [str], {type(src)} is given!"
        return cls.from_SingleImage(single_image=SingleImage(src, imread_flag=cv2.IMREAD_COLOR),
                                    color_mapper=color_mapper,
                                    defect_codes=defect_codes,
                                    is_semantic=is_semantic,
                                    ignore_defect_codes=ignore_defect_codes,
                                    require_color_in_mapper=require_color_in_mapper,
                                    name=name)

    @classmethod
    def from_SingleImage(cls,
                         single_image: 'SingleImage',
                         color_mapper: Optional['ClassMapper'] = None,
                         ignore_defect_codes: Optional[List[str]] = None,
                         defect_codes: Union[str, Dict, None] = None,
                         require_color_in_mapper: bool = True,
                         color_space: str = 'BGR',
                         name: str = 'DefectObject',
                         is_semantic: bool = False,
                         *args, **kwargs):
        assert color_space in ['BGR', 'RGB'], f'color_space must be in ["BGR", "RGB"], {color_space} is given'
        assert color_mapper is not None or not is_semantic, f'color_mapper must be given for semantic defect!'

        ignore_defect_codes = ['OK', '000'] if ignore_defect_codes is None else ignore_defect_codes

        color2idx = getattr(color_mapper, f'{color_space.lower()}_to_idx')

        single_image.open_with_color()

        return cls(data=single_image.apply(ImageConvertor.color2idx,
                                           args=(color2idx, require_color_in_mapper)).image,
                   defect_codes=color_mapper.name_to_idx if defect_codes is None else defect_codes,
                   ignore_defect_codes=ignore_defect_codes,
                   is_semantic=is_semantic,
                   name=name)

    def _build_defect_regions(self, data, defect_codes, ignore_defect_codes, is_semantic):
        if is_semantic:
            assert isinstance(defect_codes, dict) or isinstance(defect_codes, list), \
                f'If is_semantic is True, defect_codes must be a dict or a list, {type(defect_codes)} is given'
            self._regions = [MaskLayer(data=img,
                                       parent=self,
                                       name=defect_code,
                                       idx=idx)
                             for idx, (defect_code, image) in
                             enumerate(extract_target_layers(mask=data, layers=defect_codes).items())
                             if defect_code not in ignore_defect_codes and np.any(img := image.astype(bool))]
        else:
            assert isinstance(defect_codes, str), f'If is_semantic is False, ' \
                                                  f'defect_codes must be a str, {type(defect_codes)} is given'
            self._regions = [MaskLayer(data=data.astype(bool), parent=self, name=defect_codes, idx=1)]

    @property
    def defect_code(self) -> List:
        return [defect.name for defect in self] if self._is_semantic else [self.defects[0].name]

    @property
    def regions(self):
        return self.defects

    @property
    def num_regions(self) -> int:
        """
        Get number of defect code in the layer
        """
        return self.num_defects

    @property
    def defects(self):
        return self._regions

    @property
    def num_defects(self) -> int:
        """
        Get number of regions in the layer
        """
        return len(self.defects)

    @property
    def defect_code_to_idx(self):
        return {defect_code: idx for idx, defect_code in enumerate(self.defect_code)}

    def __iter__(self) -> Iterator['MaskLayer']:
        return iter(self.defects)

    def __getitem__(self, item: Union[str, int, List]) -> Union['MaskLayer', List['MaskLayer']]:
        if isinstance(item, list):
            return [self[it_] for it_ in item]
        if isinstance(item, str):
            return self.defects[self.defect_code_to_idx[item]]
        return self.defects[item]

    def __repr__(self):
        if len(self.defect_code) > 1:
            defect_code = ', '.join(self.defect_code)
        else:
            defect_code = self.defect_code[0]
        _repr = f"{self.name}[{defect_code}]"
        return _repr


class CompLayers:
    __slots__ = ['_layers', '_name', '_merge_layers', '_layer_map', '_components']

    def __init__(self,
                 data: np.ndarray,
                 layer_map: Union[List, Dict],
                 merge_layers: Optional[Dict] = None,
                 name: str = 'CompLayers'):
        """
        Parameters:
        -----------
        data [np.ndarray]: 2D data, shape=(H, W), dtype=uint8, the index on the image should be the component idx
        layer_map [List, Dict]: layer map, list or dict of component idx
        merge_layers [Dict]: layers to be merged
        name [str]: name of the layer, "chkcomp" or "refcomp" recommended, "CompLayers" as default
        """
        self._name = name
        self._layer_map = layer_map if isinstance(layer_map, list) else list(layer_map.keys())
        self._merge_layers = merge_layers
        self._components = None

        self.__get_component_layers(data=data, layer_map=layer_map)
        if merge_layers is not None:
            self.__merge_layers(merge_layers=merge_layers)

    @classmethod
    def from_path(cls,
                  src: str,
                  color_mapper: ClassMapper,
                  layer_map: Optional[Union[List, Dict]] = None,
                  merge_layers: Optional[Dict] = None,
                  require_color: bool = True,
                  name: str = 'CompLayers'):
        return cls.from_SingleImage(single_image=SingleImage(src, imread_flag=cv2.IMREAD_COLOR),
                                    color_mapper=color_mapper,
                                    layer_map=layer_map,
                                    merge_layers=merge_layers,
                                    require_color=require_color,
                                    name=name)

    @classmethod
    def from_SingleImage(cls,
                         single_image: SingleImage,
                         color_mapper: ClassMapper,
                         layer_map: Optional[Union[List, Dict]] = None,
                         merge_layers: Optional[Dict] = None,
                         require_color: bool = True,
                         color_space: str = 'BGR',
                         name: str = 'CompLayers'):
        assert color_space in ['BGR', 'RGB'], f'color_space must be in ["BGR", "RGB"], {color_space} is given'
        color2idx = getattr(color_mapper, f'{color_space.lower()}_to_idx')
        single_image.open_with_color()
        return cls(data=single_image.apply(ImageConvertor.color2idx, args=(color2idx, require_color,)).image,
                   layer_map=color_mapper.classes if layer_map is None else layer_map,
                   merge_layers=merge_layers,
                   name=name)

    def __get_component_layers(self,
                               data: np.ndarray,
                               layer_map: Union[List, Dict]):
        layer_data = extract_target_layers(mask=data, layers=layer_map)
        self._layers = {layer: MaskLayer(data=layer_data[layer].astype(bool),
                                         name=f'{self.name}.{layer}')
                        for layer in layer_map}

    def __merge_layers(self, merge_layers: Dict):
        for merged, source in merge_layers.items():
            for i, layer in enumerate(source):
                if i == 0:
                    self._layers[merged] = self._layers[layer]
                else:
                    self._layers[merged] = self._layers[merged] | self._layers[layer]
            self._layers[merged].name = f'{self.name}.{merged}'
        self._layer_map.extend(merge_layers.keys())

    @property
    def layer_map(self):
        return self._layer_map

    @property
    def merge_layers(self):
        return self._merge_layers

    @property
    def layers(self):
        if self._components is None:
            self._components = [name for name, layer in self._layers.items() if layer.is_not_empty]
        return self._components

    @property
    def name(self):
        return self._name

    def dilate(self, layer: str, kernel_size: int = 3, iterations: int = 1):
        self._layers[layer] = self._layers[layer].dilate(kernel_size=kernel_size, iterations=iterations)

    def erode(self, layer: str, kernel_size: int = 3, iterations: int = 1):
        self._layers[layer] = self._layers[layer].erode(kernel_size=kernel_size, iterations=iterations)

    def __contains__(self, item):
        return self.layers.__contains__(item)

    def __getitem__(self, item) -> Union['MaskLayer', str, List]:
        return self._layers[item]

    def __getattr__(self, item) -> Union['MaskLayer', str, List]:
        """
        Get layer by name
        """
        if item in self.layer_map:
            return self._layers[item]
        raise AttributeError(f"{item} is not legal attribute")  # self.__dict__[item]

    def __iter__(self) -> Iterator[Tuple[str, 'MaskLayer']]:
        return iter(self._layers.items())

    def __repr__(self):
        return f'{self.name}({self.layers})'

    # def _get_defct_obj_list(self, defect_obj, semantic_mask, img_h, img_w):
    #     """
    #     Get all defect binary masks according to semantic mask
    #     """
    #     from collections import defaultdict
    #
    #     defect_num_dict = defaultdict(int)
    #     defect_obj_list = []
    #     ids = np.unique(semantic_mask).tolist()
    #     for dft_idx in ids:
    #         dft_idx_mask = np.zeros(shape=(img_w, img_h), dtype=np.uint8)
    #         dft_str = self.idx2name[dft_idx]
    #         if dft_str == "000":
    #             continue
    #         dft_idx_mask[semantic_mask == dft_idx] = 1
    #
    #         defect_mask_regions = regionprops(label(dft_idx_mask))
    #         for single_defect_region in defect_mask_regions:
    #             defect_num_dict[dft_str] += 1
    #             single_defect_obj = dict()
    #             y_min, x_min, y_max, x_max = single_defect_region.bbox
    #             single_defect_obj["defect_code"] = dft_str
    #             single_defect_obj["area"] = single_defect_region.area
    #             single_defect_mask = self.get_single_defect_image(img_h, img_w, single_defect_region)
    #             min_hw, max_hw = self.calcuate_min_mask(single_defect_mask)
    #             single_defect_obj["single_defect_mask"] = single_defect_mask
    #             single_defect_obj["min_hw"] = min_hw
    #             single_defect_obj["max_hw"] = max_hw
    #             single_defect_obj["bbox"] = [x_min, y_min, x_max, y_max]
    #             single_defect_obj["h"] = y_max - y_min
    #             single_defect_obj["w"] = x_max - x_min
    #             defect_obj_list.append(single_defect_obj)
    #     defect_obj.defect_obj_list = defect_obj_list
    #     defect_obj.defect_num = len(defect_obj_list)
    #     defect_obj.defect_type_num = len(ids) - 1
    #     defect_obj.defect_num_dict = defect_num_dict
    #
    #     return defect_obj


if __name__ == '__main__':
    # from datatools.image.data import SingleImage
    # mask = SingleImage(r"\data\dataset2\Workshop\wangyueyi\KSNY\data_history\color_20230921\AING\Mask"
    #                    r"\N37TA2710051U0101_(01968,01978)_003_B_(001,003)_ALMTA7C-CO_016_39_PAD_mask.png")
    #
    # defect_obj = DefectObject(image=mask.image, defect_codes='001', ignore_defect_codes=None)
    # print(1)
    from datatools import ICsemanticMapper, AOICompMapper
    import time

    # ic_ss = DefectObject.from_path(
    #     r"\data\dataset2\Workshop\wangyueyi\KSNY\train_add_data\color\noref_semantic\all_classify_20200722-语义-2-2712\014\Mask\110144ARF 1112_(00857,01553)_104_T_(016,010)_mask.png",
    #     color_mapper=ICSegmenticMapper, name='ic_ss')

    comp_layers = CompLayers.from_path(
        r"\data\dataset2\Workshop\jianglai\sccsz2\train_data\compseg\aing_gbrcomseg_selected\006644250896_12339_b_039_gerb_mask.png",
        color_mapper=AOICompMapper)
    # comp_layers.Circuit.show(wait_key=True)
    s = time.time()
    regions_ = comp_layers.SubChar.regions
    print(time.time() - s)

