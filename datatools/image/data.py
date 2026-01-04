import os
import re
import cv2
import shutil
import numpy as np
import os.path as osp

import json
import yaml
from PIL import Image
from hashlib import md5
from functools import wraps
from typing import Optional, Union, List, Tuple, Callable, Dict
from datatools.utils import PathFormatter, SuffixFormatter, convert2map, exists_or_make
from datatools.image.mappers import ClassMapper


# Decorator to ensure that the image is loaded before calling a function that operates on it
def open_file(fn):
    @wraps(fn)
    # TODO: modify the code to fileio adapter
    def wrapped_fn(*args, **kwargs):
        if getattr(args[0], '_img_data') is None:
            args[0].imread()
        return fn(*args, **kwargs)

    return wrapped_fn


def clear_cache(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        args[0].release()
        return fn(*args, **kwargs)
    return wrapped_fn


class SingleImage:
    __slots__ = ['_root', '_name', '_img_data', '_imread_fn', '_backend', '_imread_flag', '_parent', '_mark']

    BACKEND_ALIAS = convert2map({
        'cv2': ['cv2', 'opencv', 'opencv-python'],
        'pillow': ['pillow', 'PIL', 'pil']
    })
    ALLOWED_BACKEND = set(BACKEND_ALIAS.values())

    def __init__(self,
                 src: str = './',
                 new: bool = False,
                 name: Optional[str] = None,
                 imread: Optional[Callable] = None,
                 backend: str = 'cv2',
                 imread_flag: int = cv2.IMREAD_UNCHANGED,
                 parent: Optional['ImageData'] = None):
        """ Image Object Compatible with 'cv2' and 'pillow'.

        Properties:
            image (np.array or Image.Image): Image data.

            name (str): Name of the image.

            root (str): Root path of the image.

            path (str): Path of the image.

            shape (Tuple): Shape of the image.

        Functions:
            show: Show the image.

            apply: Apply a function to the image.

            save: Save the image.

            copy_to: Copy the image to another directory.

        Args:
            src (str): Source path of the image.
            new (bool): Whether this is a new image.
            name (str, optional): Name of the image. If None, a default name is used.
            imread (Callable, optional): Image read function.
            backend (str): Backend for image processing (cv2 or pillow).
        """

        assert SingleImage.BACKEND_ALIAS[backend] in SingleImage.ALLOWED_BACKEND, \
            f'Backend must be either cv2 or pillow! {backend} is not allowed!'
        assert isinstance(src, str), f"src must be str, while {type(src)} is given."
        src = PathFormatter.format(src)
        temp_root, temp_name = osp.split(src)
        if not new:
            assert osp.exists(src) and osp.isfile(src), f"Image Does Not Exist! {src}"
        else:
            if name is None:
                temp_name = temp_name if SuffixFormatter.is_supported_format(temp_name) else 'temp'

        self._root = temp_root
        self._name = temp_name if name is None else name
        self._img_data = None
        self._imread_fn = imread
        self._backend = backend
        self._imread_flag = imread_flag
        self._parent = parent
        self._mark = None

    def _get_img_read_fn(self) -> Callable:
        if self._backend in ['cv2', 'opencv', 'opencv-python']:
            return cv2.imread
        elif self._backend in ['pillow', 'PIL', 'pil']:
            return Image.open
        return cv2.imread

    @clear_cache
    def use_backend(self, backend: str):
        assert backend in SingleImage.ALLOWED_BACKEND, \
            f'Backend must be either cv2 or pillow! {backend} is not allowed!'
        self._backend = backend

    @clear_cache
    def open_with_unchanged(self):
        self._imread_flag = cv2.IMREAD_UNCHANGED

    @clear_cache
    def open_with_grayscale(self):
        self._imread_flag = cv2.IMREAD_GRAYSCALE

    @clear_cache
    def open_with_color(self):
        self._imread_flag = cv2.IMREAD_COLOR

    def imread(self) -> None:
        # imread_fn = self._get_img_read_fn()
        if SuffixFormatter.is_encrypted_format(self.name):
            pass
        elif self._backend in ['cv2', 'opencv', 'opencv-python']:
            self._img_data = cv2.imread(self.path, flags=self._imread_flag)
        elif self._backend in ['pillow', 'PIL', 'pil']:
            self._img_data = Image.open(self.path)
            if self._imread_flag == cv2.IMREAD_GRAYSCALE:
                self._img_data = self._img_data.convert('L')
            elif self._imread_flag == cv2.IMREAD_COLOR:
                self._img_data = self._img_data.convert('RGB')

    @property
    def path(self) -> str:
        return osp.join(self._root, self._name)

    @property
    def shape(self) -> Tuple:
        if self._backend == 'cv2':
            return self.image.shape
        elif self._backend == 'pillow':
            return np.array(self.image).shape
        return self.image.shape

    @property
    def name(self) -> str:
        return self._name

    @property
    def root(self) -> str:
        return self._root

    @property
    def mark(self):
        return self._mark

    @property
    def md5(self) -> str:
        hash_calc = md5()
        with open(self.path, 'rb') as cur:
            hash_calc.update(cur.read())
        return hash_calc.hexdigest()

    @property
    def has_parent(self) -> bool:
        return self._parent is not None

    # @property
    # def img_data(self) -> Union[np.ndarray, Image.Image, None]:
    #     return self._img_data

    @property
    @open_file
    def image(self) -> Union[np.ndarray, Image.Image]:
        return self._img_data

    @property
    @open_file
    def properties(self) -> None:
        return None

    def set_backend(self, backend):
        assert backend in ['cv2', 'pillow'], f'Backend must be either cv2 or pillow! {backend} is not allowed!'
        self._backend = backend
        self._img_data = None

    def release(self) -> None:
        self._img_data = None

    def mark_image(self, mark: str) -> None:
        self._mark = ord(mark)
        if self.has_parent:
            self._parent.mark_image(mark)

    def from_numpy(self,
                   image: np.ndarray,
                   use_raw: bool = False) -> 'SingleImage':
        self._img_data = image if use_raw else np.uint8(image)
        return self

    def from_pil_image(self, image: Image.Image) -> 'SingleImage':
        self._img_data = image
        return self

    def save(self, dst: Optional[str] = None, force: bool = True, extension: str = '.png') -> None:
        assert self.image is not None, 'No Image Data'
        dst = self._root if dst is None else PathFormatter.format(dst)
        if SuffixFormatter.is_supported_format(self.name):
            file_name = self.name
        else:
            extension = extension if extension.startswith('.') else '.' + extension
            extension = extension if SuffixFormatter.is_supported_format(extension) else '.png'
            file_name = self.name + extension

        if not osp.exists(dst):
            os.makedirs(dst)
        file_path = osp.join(dst, file_name)
        is_existed = osp.exists(file_path)
        if (is_existed and force) or (not is_existed):
            cv2.imwrite(file_path, self._img_data)
        else:
            raise FileExistsError(f'File exists: {file_path}')

    @open_file
    def apply(self,
              fn,
              args: Tuple = (),
              use_raw: bool = False,
              in_place: bool = False,
              show: bool = False,
              name: Optional[str] = 'temp',
              wait_key: bool = False,
              is_binary: bool = False,
              destroy_window: bool = False,
              backend: Optional[str] = None,
              **kwargs) -> 'SingleImage':
        kwargs = dict() if kwargs is None else kwargs
        try:
            ret = fn(self.image, *args, **kwargs)
        except ValueError:
            raise ValueError(f'Error: {self.path}, function: {fn.__name__}')
        if isinstance(ret, tuple):
            img_data, *_ = ret
        else:
            img_data = ret
        temp = SingleImage(name=name,
                           new=True,
                           backend=self._backend if backend is None else backend)
        if isinstance(img_data, Image.Image):
            temp = temp.from_pil_image(img_data)
        elif isinstance(img_data, np.ndarray):
            temp = temp.from_numpy(image=img_data, use_raw=use_raw)
        else:
            raise TypeError(f'Unknown Type: {type(img_data)}')

        if show:
            temp.show(named_window=name, wait_key=wait_key, is_binary=is_binary, destroy_window=destroy_window)
        if in_place:
            self._img_data = img_data
            return self
        return temp

    @open_file
    def show(self,
             named_window: Optional[str] = None,
             wait_key: bool = False,
             is_binary: bool = False,
             destroy_window: bool = False,
             destroy_all_windows: bool = False,
             allowed_keys: Union[str, List[str], None] = ' ',
             mark_input: bool = False,
             allowed_marks: Union[str, List[str], None] = None,):

        if self._backend in ['cv2', 'opencv', 'opencv-python']:
            named_window = self._name if named_window is None else named_window
            im_show = self.image * 255 if is_binary else self.image
            cv2.namedWindow(named_window, cv2.WINDOW_NORMAL)
            cv2.imshow(winname=named_window,
                       mat=im_show)
            if wait_key:
                if isinstance(allowed_keys, str):
                    allowed_keys = [allowed_keys]

                if isinstance(allowed_marks, str):
                    allowed_marks = [allowed_marks]

                if allowed_marks is not None:
                    mark_input = True

                if allowed_keys is None:
                    cv2.waitKey(0)
                else:
                    while True:
                        key = chr(cv2.waitKey(0))
                        if key in allowed_keys:
                            break
                        if mark_input:
                            if allowed_marks is None or key in allowed_marks:
                                if self.mark is not None:
                                    im_show_with_mark = im_show.copy()
                                    color = (255, 255, 255) if len(self.shape) > 2 else (255, )
                                    cv2.putText(im_show_with_mark, key, (self.shape[0] // 2, self.shape[1] // 2,),
                                                cv2.FONT_HERSHEY_COMPLEX, 2.0, color, 5)
                                    cv2.imshow(winname=named_window,
                                               mat=im_show_with_mark)
                                self.mark_image(key)
                if destroy_all_windows:
                    cv2.destroyAllWindows()
                elif destroy_window:
                    cv2.destroyWindow(winname=named_window)
        elif self._backend in ['pillow', 'PIL', 'pil']:
            self.image.show()
        else:
            print(f'Unknown Backend: {self._backend}')

    @open_file
    def copy_to(self,
                dst: str,
                force: bool = False) -> None:
        dst = PathFormatter(dst).path
        if not force and not osp.exists(dst):
            raise FileNotFoundError(f'No such directory: \'{dst}\'')
        self._root = dst
        self.save()

    @staticmethod
    def get_suffix(file: str) -> Optional[str]:
        _, ext = osp.splitext(file)
        suffix_pattern = re.compile('_(?P<suffix>[a-zA-Z]+)' + ext)
        res = re.search(suffix_pattern, file)
        return res['suffix'] if res is not None else None

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'
        repr_str += f'("{self.path}", '
        repr_str += f'backend="{self._backend}")'
        return repr_str

    # @staticmethod
    # def extract_layer(comp_mask: np.ndarray, mask_h: int = 160, mask_w: int = 160):
    #     comp_bg = np.zeros((mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_bg, layer_mask=comp_mask, layer_id=0, num_threads=1)
    #
    #     comp_ink = np.zeros((mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_ink, layer_mask=comp_mask, layer_id=1, num_threads=1)
    #
    #     comp_pad = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_pad, layer_mask=comp_mask, layer_id=2, num_threads=1)
    #
    #     comp_ring = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_ring, layer_mask=comp_mask, layer_id=3, num_threads=1)
    #
    #     comp_via = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_via, layer_mask=comp_mask, layer_id=4, num_threads=1)
    #
    #     comp_silk = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_silk, layer_mask=comp_mask, layer_id=5, num_threads=1)
    #
    #     comp_laser_char = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_laser_char, layer_mask=comp_mask, layer_id=6, num_threads=1)
    #
    #     comp_vcut = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_vcut, layer_mask=comp_mask, layer_id=7, num_threads=1)
    #
    #     comp_silk_char = np.zeros((mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_silk_char, layer_mask=comp_mask, layer_id=8, num_threads=1)
    #
    #     comp_circuit = np.zeros((mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_circuit, layer_mask=comp_mask, layer_id=9, num_threads=1)
    #
    #     comp_gold_char = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_gold_char, layer_mask=comp_mask, layer_id=10, num_threads=1)  # gold char
    #
    #     comp_finger = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_finger, layer_mask=comp_mask, layer_id=11, num_threads=1)  # finger
    #
    #     comp_opt = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_opt, layer_mask=comp_mask, layer_id=12, num_threads=1)  # opt
    #
    #     comp_fp = np.zeros((mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_fp, layer_mask=comp_mask, layer_id=13, num_threads=1)
    #
    #     comp_special = np.zeros((mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_special, layer_mask=comp_mask, layer_id=14, num_threads=1)
    #
    #     comp_tiebar = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_tiebar, layer_mask=comp_mask, layer_id=15, num_threads=1)
    #
    #     comp_qrcode = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
    #     Stunner.extract_layer(layer=comp_qrcode, layer_mask=comp_mask, layer_id=16, num_threads=1)
    #
    #     return comp_bg, comp_ink, comp_pad, comp_ring, comp_via, comp_silk, comp_laser_char, comp_vcut,
    #     comp_silk_char, comp_circuit, comp_gold_char, comp_finger, comp_opt, comp_fp, comp_special,
    #     comp_tiebar, comp_qrcode
    #
    # @staticmethod
    # def comp_slicer(comp_mask, img_h: int = 160, img_w: int = 160):
    #     comp_bg, comp_ink, comp_pad, comp_ring, comp_via, comp_silk, comp_laser_char, comp_vcut, comp_silk_char, \
    #     comp_circuit, comp_gold_char, comp_finger, comp_opt, comp_fp, comp_special, comp_tiebar, comp_qrcode = \
    #         SingleImage.extract_layer(comp_mask=comp_mask, mask_h=img_h, mask_w=img_w)
    #
    #     comp_bg += comp_via
    #     comp_bg = np.clip(comp_bg, a_min=0, a_max=1).astype(dtype=np.uint8)
    #
    #     comp_pad += comp_ring
    #     comp_pad += comp_gold_char
    #     comp_pad += comp_finger
    #     comp_pad += comp_opt
    #     comp_pad += comp_tiebar
    #     comp_pad = np.clip(comp_pad, a_min=0, a_max=1).astype(dtype=np.uint8)
    #
    #     comp_layers = {"comp_bg": comp_bg,
    #                    "comp_ink": comp_ink,
    #                    "comp_pad": comp_pad,
    #                    "comp_ring": comp_ring,
    #                    "comp_via": comp_via,
    #                    "comp_silk": comp_silk,
    #                    "comp_laser_char": comp_laser_char,
    #                    "comp_vcut": comp_vcut,
    #                    "comp_silk_char": comp_silk_char,
    #                    "comp_circuit": comp_circuit,
    #                    "comp_gold_char": comp_gold_char,
    #                    "comp_finger": comp_finger,
    #                    "comp_opt": comp_opt,
    #                    "comp_fp": comp_fp,
    #                    "comp_special": comp_special,
    #                    "comp_tiebar": comp_tiebar,
    #                    "comp_qrcode": comp_qrcode}
    #     return comp_layers

    # @staticmethod
    # def gerb_slicer(comp_gbr, img_h: int = 160, img_w: int = 160):
    #     comp_bg_gbr, comp_ink_gbr, comp_pad_gbr, comp_ring_gbr, comp_via_gbr, comp_silk_gbr, comp_laser_char_gbr, \
    #     comp_vcut_gbr, comp_silk_char_gbr, comp_circuit_gbr, comp_gold_char_gbr, comp_finger_gbr, comp_opt_gbr, \
    #     comp_fp_gbr, comp_special_gbr, comp_tiebar_gbr, comp_qrcode_gbr = \
    #         SingleImage.extract_layer(comp_mask=comp_gbr, mask_h=img_h, mask_w=img_w)
    #     comp_pad_gbr += comp_ring_gbr
    #     comp_pad_gbr += comp_gold_char_gbr
    #     comp_pad_gbr += comp_finger_gbr
    #     comp_pad_gbr += comp_opt_gbr
    #     comp_pad_gbr += comp_tiebar_gbr
    #     comp_pad_gbr = np.clip(comp_pad_gbr, a_min=0, a_max=1).astype(dtype=np.uint8)
    #     comp_gbr_layers = {"comp_pad_gbr": comp_pad_gbr,
    #                        "comp_tiebar_gbr": comp_tiebar_gbr,
    #                        "comp_ink_gbr": comp_ink_gbr,
    #                        "comp_via_gbr": comp_via_gbr}
    #     return comp_gbr_layers


class ImageData:
    __slots__ = ['_cur', '_ref', '_gerb', '_ann', '_mask',
                 '_cam', '_comp', '_refcomp', '_gerbcomp', '_speccomp',
                 '_id',
                 # '_infrared', '_refinfrared',
                 '__label', '__separated', '__allowed', '__remove_no_use_attr', '__mark',
                 '__backend', '__use_single_image', '__require_mask', '__strict_inspection', '__hard_sample']

    ALLOWED = [attr[1:] for attr in __slots__ if attr.startswith('_') and not attr.startswith('__')]
    NO_MASK_CLUSTER = '000'
    DUP_RENAME_PATTERN = re.compile(f'.+_Copy-(?P<dup_num>[0-9]+).*')

    def __init__(self,
                 file_path: str,
                 separated: Optional[bool] = None,
                 use_single_image: bool = True,
                 backend: str = 'cv2',
                 require_mask: bool = False,
                 strict_inspection: bool = False,
                 hard_sample: bool = False,
                 mark: Optional[str] = None):
        """ Load image and its auxiliary data.


        ImageData takes the image path of [chk/cur] image as input, and loads the image and its auxiliary data.
        The auxiliary data should be named with associated suffix and either be stored in the same folder
        as the [chk/cur] image or separated in associated folder.


        Separated DataCluster Example
                |----Cur ---- xxx.png
                \\

                |----Ref ---- xxx_std.png
                \\

                |----Mask ---- xxx_mask.png
                \\

                |----Gerb ---- xxx_gerb.png
                \\

        \\

        Non-Separated DataCluster Example
                |---- XXX.png
                \\

                |---- XXX_std.png
                \\

                |---- XXX_mask.png
                \\

                |---- XXX_gerb.png
                \\


        Args:
            file_path (str): The file path of [chk/cur] image, or regard [ref/gerb] as [chk/cur] if needed.
            separated (bool, optional): Indicator of the image and its auxiliary data that are stored separately.
            use_single_image (bool): Flag of using SingleImage or String (faster) as the type of attribute.
            backend (str): Target backend for loading image if using SingleImage. Should be 'cv2' or 'PIL'.
            require_mask (bool): Flag of requiring attribute 'mask' of the image.
            strict_inspection (bool): Flag of using md5 for checking duplicate images.
            hard_sample (bool): Flag of hard sample.
        """

        assert backend in ['cv2', 'pillow'], f'Backend must be either cv2 or pillow! {backend} is not allowed!'
        
        file_path = PathFormatter.format(file_path)
        
        cur_folder = 'Cur'.join([os.sep, os.sep])
        if separated is None:
            separated = cur_folder in file_path
            
        assert osp.exists(file_path) and osp.isfile(file_path), f"Image Does Not Exist! {file_path}"
        assert separated == (cur_folder in file_path), f"Image Not Separated to 'Cur' folder: {file_path}" \
            if separated else f"Image Already Separated to 'Cur' folder: {file_path}"

        self.__mark = mark
        self.__label = None
        self._cur = file_path
        self.__backend = backend
        self.__separated = separated
        self.__hard_sample = hard_sample
        self.__remove_no_use_attr = False
        self.__require_mask = require_mask
        self.__use_single_image = use_single_image
        self.__strict_inspection = strict_inspection

        self.__set_default_value()

    def __set_default_value(self) -> None:
        for attr in ImageData.ALLOWED:
            if attr == 'cur':
                continue
            tgt_attr = '_' + attr.lower()
            setattr(self, tgt_attr, None)

    def release(self):
        self.__set_default_value()

    def mark_image(self, mark: str):
        self.__mark = mark

    def _redirect(self, cur_path: str) -> None:
        self._cur = cur_path
        self.release()

    @property
    def cur(self) -> Union[SingleImage, str]:
        return SingleImage(self._cur, backend=self.__backend, parent=self) if self.__use_single_image else self._cur

    @property
    def center(self) -> Tuple[int]:
        # [H, W]
        file = self.get_renamed_path('txt', 'center')
        with open(file, 'r') as file:
            center = tuple(int(coord) for coord in file.read().split('|'))
        return center
    
    @property
    def info(self):
        ann = self.ann
        if ann:
            suffix = ann.split(".")[-1]
            if suffix in ['json']:
                with open(ann, 'r') as f:
                    info = json.load(f)
                return info
            else:
                print(f"Unsuppoorted format: {suffix}")
        return None

    @property
    def name(self) -> str:
        return osp.basename(self._cur)

    @property
    def mark(self):
        return self.__mark

    @property
    def require_mask(self) -> bool:
        return self.__require_mask

    @property
    def md5(self) -> str:
        hash_calc = md5()
        with open(self._cur, 'rb') as cur:
            hash_calc.update(cur.read())
        return hash_calc.hexdigest()

    @property
    def attributes(self) -> List:
        return [attr for attr in ImageData.ALLOWED if getattr(self, attr) is not None]

    @property
    def use_single_image(self) -> bool:
        return self.__use_single_image

    @property
    def label(self) -> str:
        if self.__label is None:
            path_list = self._cur.split(os.sep)
            cls_idx = -3 if self.__separated else -2
            self.__label = path_list[cls_idx]
        return self.__label

    @property
    def is_hard_sample(self) -> bool:
        return self.__hard_sample

    def set_backend(self, backend) -> None:
        assert backend in ['cv2', 'pillow'], f'Backend must be either cv2 or pillow! {backend} is not allowed!'
        self.__backend = backend
        self.release()

    def get_single_image(self, target_file) -> Optional[SingleImage]:
        tgt_attr = target_file.lower()
        if tgt_attr in ImageData.ALLOWED:
            attr = getattr(self, tgt_attr)
            if self.__use_single_image:
                return attr
            if attr is not None:
                return SingleImage(src=attr, backend=self.__backend)
        return None

    def get_semantic_cluster(self,
                             color_mapper: 'ClassMapper',
                             priority_class: Optional[List[str]] = None,
                             require_mask: bool = False,) -> Optional[str]:
        from datatools.image.layer import DefectObject
        self.disable_single_image()
        if self.mask is None:
            if require_mask:
                raise ValueError(f'No mask for {self._cur}')
            return ImageData.NO_MASK_CLUSTER
        defect_ss = DefectObject.from_path(src=self.mask, color_mapper=color_mapper, is_semantic=True)
        if len(defect_ss.defect_code) == 0:
            return ImageData.NO_MASK_CLUSTER
        defect_area = {label: defect_ss[label].area for label in defect_ss.defect_code}
        if priority_class is not None:
            for priority in priority_class:
                if priority in defect_area:
                    return priority
        return sorted(defect_area.items(), key=lambda x: x[1], reverse=True)[0][0]

    def get_compseg_cluster(self,
                            color_mapper: 'ClassMapper',
                            priority_class: Optional[List[str]] = None):
        from datatools.image.layer import CompLayers
        self.disable_single_image()
        if self.mask is None:
            raise ValueError(f'No mask for {self._cur}')
        comp_layers = CompLayers.from_path(src=self.mask, color_mapper=color_mapper)
        if priority_class is not None:
            for layer in priority_class:
                if layer in comp_layers.layers:
                    return layer
        area = {layer: getattr(comp_layers, layer).area for layer in comp_layers.layers if layer != '000'}
        return sorted(area.items(), key=lambda x: x[1], reverse=True)[0][0]

    def enable_single_image(self) -> None:
        if not self.__use_single_image:
            self.release()
            self.__use_single_image = True

    def disable_single_image(self) -> None:
        if self.__use_single_image:
            self.release()
            self.__use_single_image = False

    def enable_strict_inspection(self) -> None:
        self.__strict_inspection = True

    def disable_strict_inspection(self) -> None:
        self.__strict_inspection = False

    def rename(self, new_name: str, exceptions: Optional[List] = None) -> None:
        assert '.' not in new_name and os.sep not in new_name, 'Invalid new name!'
        exceptions = exceptions if exceptions is not None else []
        use_single_image = self.__use_single_image
        self.disable_single_image()
        cwd = os.getcwd()
        renamed_cur = None
        for attr in ImageData.ALLOWED:
            if attr in exceptions:
                continue
            file = getattr(self, attr)
            if file is not None:
                file_root, file_name = osp.split(file)
                _, ext = osp.splitext(file_name)
                suffix = SuffixFormatter.get_suffix(file_name)
                rename = f'{new_name}_{suffix}{ext}' \
                    if suffix is not None and attr not in ['cur', 'Cur'] else f'{new_name}{ext}'
                os.chdir(file_root)
                os.rename(file_name, rename)
                if attr in ['cur', 'Cur']:
                    renamed_cur = osp.join(file_root, rename)
        os.chdir(cwd)
        self._redirect(cur_path=renamed_cur)
        if use_single_image:
            self.enable_single_image()
        else:
            self.disable_single_image()

    def force_copy_to(self,
                      dst: str,
                      overwrite: bool = False,
                      move: bool = False,
                      inplace: bool = False,
                      separate: Optional[bool] = None,
                      target_attrs: Union[List, str, None] = None,
                      exceptions: Union[List, str, None] = None) -> None:

        self.copy_to(dst=dst,
                     force=True,
                     force_copy=True,
                     overwrite=overwrite,
                     move=move,
                     inplace=inplace,
                     separate=separate,
                     target_attrs=target_attrs,
                     exceptions=exceptions)

    def copy_to(self,
                dst: str,
                force: bool = False,
                force_copy: bool = False,
                overwrite: bool = False,
                move: bool = False,
                inplace: bool = False,
                separate: Optional[bool] = None,
                target_attrs: Union[List, str, None] = None,
                exceptions: Union[List, str, None] = None) -> None:
        dst = PathFormatter.format(dst)
        target_attrs = ImageData.ALLOWED if target_attrs is None else target_attrs
        if isinstance(target_attrs, str):
            target_attrs = [target_attrs]
        if exceptions is not None:
            exceptions = [attr.lower() for attr in exceptions] if isinstance(exceptions, list) else [exceptions.lower()]
            target_attrs = list(set(target_attrs) - set(exceptions))

        separate = self.__separated if separate is None else separate
        for attr in target_attrs:
            file = getattr(self, attr)
            if file is not None:
                attr_dst = osp.join(dst, attr.capitalize()) if separate else dst
                file = file.path if isinstance(file, SingleImage) else file
                exists_or_make(attr_dst, make_dir=force)
                save_path = osp.join(attr_dst, osp.basename(file))
                if osp.exists(save_path):
                    if force_copy:
                        if attr in ['cur', 'Cur']:
                            if not overwrite:
                                dup_img = ImageData(save_path, use_single_image=False)
                                dup_img.rename(dup_img.get_dup_rename_under_dir(attr_dst))
                                print(f'File exists: {save_path}, rename to {dup_img.name}')
                            else:
                                print(f'File overwriten: {save_path}')
                    else:
                        raise FileExistsError(f"File exists: {save_path}, "
                                              f"use 'force_copy = True' to overwrite or make copy!")
                execution = shutil.move if move else shutil.copy
                execution(src=file, dst=attr_dst)
        if move or inplace:
            moved_cur = osp.join(dst, 'Cur', self.name) if separate else osp.join(dst, self.name)
            self._redirect(cur_path=moved_cur)

    def get_dup_rename_under_dir(self, src: str):
        src = PathFormatter.format(src)
        file_name, ext = osp.splitext(self.name)
        dup_num = 0
        rename = f'{file_name}_Copy-{int(dup_num) + 1}'
        while osp.exists(osp.join(src, f'{rename}{ext}')):
            dup_num = int(dup_num) + 1
            rename = f'{file_name}_Copy-{dup_num}'
        return rename

    def get_renamed_path(self,
                         ext: str,
                         suffix: str) -> str:
        folder = suffix.capitalize().join([os.sep, os.sep])
        cur_folder = 'Cur'.join([os.sep, os.sep])
        replace_suffix = '.'.join(['_' + suffix, ext]) if suffix not in ['Ann', 'ann'] else '.' + ext
        cur = self._cur
        _, cur_ext = osp.splitext(cur)
        renamed = cur.replace(cur_folder, folder).replace(cur_ext, replace_suffix)
        if suffix in ['ref', 'Ref']:
            found = False
            for ext in ['jpg', 'png', 'bmp']:
                for ref_suffix in ['std', 'ref']:
                    if osp.exists(renamed):
                        found = True
                        break
                    replace_suffix = '.'.join(['_' + ref_suffix, ext])
                    renamed = cur.replace(cur_folder, folder).replace(cur_ext, replace_suffix)
                if found:
                    break
        return renamed

    def __get_attr(self, attr: str) -> Union['SingleImage', str, None]:
        attr = attr.lower()
        ext = 'png' if attr not in ['ann'] else 'json'
        latent_attr = '_' + attr
        memory = getattr(self, latent_attr)
        if memory is None:
            memory = self.get_renamed_path(ext=ext, suffix=attr)
            if not osp.exists(memory):
                if self.__remove_no_use_attr:
                    ImageData.ALLOWED.remove(attr)
                return None
            if self.__use_single_image and attr not in ['ann']:
                memory = SingleImage(memory, backend=self.__backend, parent=self)
            setattr(self, latent_attr, memory)
        return memory

    def switch_attr(self, ori_attr: str, new_attr: str, force: bool = False):
        assert ori_attr in self.ALLOWED and ori_attr not in ['ann', 'Ann'], \
            f'Ori attribute {ori_attr} is not allowed to switch!'
        assert new_attr in self.ALLOWED and ori_attr not in ['ann', 'Ann'], \
            f'New attribute {new_attr} is not allowed to switch!'
        ori_attr = getattr(self, ori_attr)
        ori_attr = ori_attr.path if isinstance(ori_attr, SingleImage) else ori_attr
        new_attr = self.get_renamed_path(ext='png', suffix=new_attr)
        if os.path.exists(new_attr):
            if force:
                os.remove(new_attr)
            else:
                raise FileExistsError(f'New Attribute File exists: {new_attr}')
        os.makedirs(osp.dirname(new_attr), exist_ok=True)
        shutil.move(ori_attr, new_attr)

    def __getattr__(self, item: str) -> Union['SingleImage', str, None]:
        item = item.lower()
        if item in ImageData.ALLOWED:
            return self.__get_attr(item)
        raise AttributeError(f'\'{item}\' is not allowed!')

    def __eq__(self, other) -> bool:
        if not isinstance(other, ImageData):
            raise TypeError(f'Cannot compare ImageData with {type(other)}!')
        return hash(self) == hash(other)

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'
        repr_str += f'("{self._cur}", '
        repr_str += f'separated={self.__separated})'
        return repr_str

    def __hash__(self):
        return hash(self.md5) if self.__strict_inspection else hash(self.name)
