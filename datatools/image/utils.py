from pystunner.stunner import Stunner
from typing import Union, List, Dict
import numpy as np


def extract_layer(mask: np.ndarray,
                  layer_id: int,
                  num_threads: int = 1,
                  height: int = 160,
                  width: int = 160) -> np.ndarray:
    layer = np.zeros((height, width), dtype=np.uint8)
    Stunner.extract_layer(layer=layer, layer_mask=mask, layer_id=layer_id, num_threads=num_threads)
    return layer


def extract_target_layers(mask: np.ndarray,
                          layers: Union[Dict, List],
                          num_threads: int = 1) -> Dict:
    if isinstance(layers, List):
        layers = {layer: i for i, layer in enumerate(layers)}
    mask_shape = mask.shape
    return {layer: extract_layer(mask=mask,
                                 layer_id=idx,
                                 num_threads=num_threads,
                                 height=mask_shape[0],
                                 width=mask_shape[1]) for layer, idx in layers.items()}


# def color2id(mask: np.ndarray,
#              color_map: Union[List, Dict],
#              channel='BGR'):
#
#     h, w, _ = mask.shape
#     mask_id = np.zeros((h, w), dtype=np.uint8)
#     if isinstance(color_map, list):
#         color_map = {idx: color for idx, color in enumerate(color_map)}
#     for target_idx, color in color_map.items():
#         if channel in ['BGR']:
#             color.reverse()
#         indices = mask == color
#         indices = np.all(indices, axis=2)
#         mask_id[indices] = target_idx
#
#     return mask_id


if __name__ == '__main__':
    from datatools.image import SingleImage
    from mappers import ColorMapper

    temp_mask = SingleImage(r"\data\dataset2\Workshop\sunjianyao\hzsh\train_data\compseg\20230419_semislot_bg_addup\semislot_bg\008664262293_41_t_020_0_std_mask.png",
                            backend='pillow')
    print(type(temp_mask.image))
    # temp_mask.show(wait_key=True)
    # id_mask = temp_mask.apply(color2id, args=(ColorMapper.PRODUCT_INSPECT_COMP.id2color,))
    # layer = id_mask.apply(extract_layer,
    #                       show=True,
    #                       wait_key=True,
    #                       is_binary=True,
    #                       args=(ColorMapper.PRODUCT_INSPECT_COMP.name2id['SolderMask'],))
    # print(time.time() - s)
