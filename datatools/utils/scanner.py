import os
import os.path as osp
from pathlib import Path
from typing import Optional, Union, Callable


def scandir(dir_path: str,
            with_suffix: Union[str, tuple, None] = None,
            exclude_suffix: Union[str, tuple, None] = None,
            with_extension: Union[str, tuple, None] = None,
            exclude_extension: Union[str, tuple, None] = None,
            recursive: bool = False,
            case_sensitive: bool = True,
            process: Optional[Callable] = None,
            **kwargs):

    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (with_suffix is not None) and not isinstance(with_suffix, (str, tuple)):
        raise TypeError('"with_suffix" must be a string or tuple of strings')

    if with_suffix is not None and not case_sensitive:
        with_suffix = with_suffix.lower() \
            if isinstance(with_suffix, str) \
            else tuple(item.lower() for item in with_suffix)

    if exclude_suffix is not None and not isinstance(exclude_suffix, (str, tuple)):
        raise TypeError('"exclude_suffix" must be a string or tuple of strings')

    if exclude_suffix is not None and not case_sensitive:
        exclude_suffix = exclude_suffix.lower() \
            if isinstance(exclude_suffix, str) \
            else tuple(item.lower() for item in exclude_suffix)

    if (with_extension is not None) and not isinstance(with_extension, (str, tuple)):
        raise TypeError('"with_extension" must be a string or tuple of strings')

    if with_extension is not None and not case_sensitive:
        with_suffix = with_extension.lower() \
            if isinstance(with_extension, str) \
            else tuple(item.lower() for item in with_extension)

    if exclude_extension is not None and not isinstance(exclude_extension, (str, tuple)):
        raise TypeError('"exclude_extension" must be a string or tuple of strings')

    if exclude_extension is not None and not case_sensitive:
        exclude_extension = exclude_extension.lower() \
            if isinstance(exclude_extension, str) \
            else tuple(item.lower() for item in exclude_extension)


    root = dir_path

    def _scandir(dir_path, with_suffix, exclude_suffix,
                 with_extension, exclude_extension,
                 recursive, case_sensitive, process, **kwargs):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                file = os.path.split(rel_path)[-1] if case_sensitive else os.path.split(rel_path)[-1].lower()
                file_name, ext = osp.splitext(file)
                if ((with_suffix is None or file_name.endswith(with_suffix))
                        and (exclude_suffix is None or not file_name.endswith(exclude_suffix))
                        and (with_extension is None or ext.endswith(with_extension))
                        and (exclude_extension is None or not ext.endswith(exclude_extension))):
                    if process is not None:
                        yield process(file_path=entry.path, rel_path=rel_path, **kwargs)
                    else:
                        yield entry.path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(dir_path=entry.path,
                                    with_suffix=with_suffix,
                                    exclude_suffix=exclude_suffix,
                                    with_extension=with_extension,
                                    exclude_extension=exclude_extension,
                                    recursive=recursive,
                                    case_sensitive=case_sensitive,
                                    process=process,
                                    **kwargs)

    return _scandir(dir_path=dir_path,
                    with_suffix=with_suffix,
                    exclude_suffix=exclude_suffix,
                    with_extension=with_extension,
                    exclude_extension=exclude_extension,
                    recursive=recursive,
                    case_sensitive=case_sensitive,
                    process=process,
                    **kwargs)


if __name__ == '__main__':
    pass
    # pc = proc()
    # import time
    # from datatools.dataset import *
    # from datatools.image import *
    #
    #
    # def tp(file_path, rel_path, **kwargs):
    #     if len(rel_path.split('/')) > 3:
    #         print(file_path)
    #     return file_path
    #
    #
    # s = time.time()
    # root_dir = r'\data\data_cold\Workshop\general_PI\osphj\other_ink\seg_withref\NG\wxjd2a_20230516_cleaned'.replace('\\', '/')
    # res = DataContainer()
    # for ret in scandir(root_dir,
    #                    recursive=True,
    #                    exclude_suffix=('_mask', '_gerb', '_ref', '_std'),
    #                    with_extension=('.jpg', '.png'),
    #                    process=tp):
    #     img = ImageData(ret, separated='Cur' in ret)
    #     res[img.label].append(img)
    # m = time.time()
    # dp = DataPatch(root_dir, num_workers=8, duplicates={'all': 1})
    # e = time.time()
    #
    # print(m - s)
    # print(e - m)
    # print(res.total_num)
    # print(dp.data.total_num)
    # dst = r'\data\dataset2\Workshop\sunjianyao\test\check'.replace('\\', '/')
    # dup = dp.data.strict_duplication_check()
    # for m5dv, imgs in dup.items():
    #     for idx, img in enumerate(imgs):
    #         img.copy_to(os.path.join(dst, os.path.basename(root_dir), m5dv), force=True)
    #         # img.rename(f'{m5dv}_{idx}')
    # print(dup.total_num)

