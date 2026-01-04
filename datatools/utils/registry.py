from typing import Optional, Dict, Type, Union, List, Callable
import inspect


def build_from_cfg(cfg: dict,
                   registry: 'Registry',
                   default_args: Optional[Dict] = None,
                   recursive: bool = True):

    if not isinstance(cfg, dict):
        raise TypeError(
            f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}')

    if default_args is not None and not isinstance(default_args, dict):
        raise TypeError(
            f'default_args should be a dict, ConfigDict or Config, but got {type(cfg)}')

    if 'type' not in cfg:
        raise KeyError(
            '`cfg` or `default_args` must contain the key "type", '
            f'but got {cfg}\n')

    if not isinstance(registry, Registry):
        raise TypeError('registry must be a Registry object, '
                        f'but got {type(registry)}')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')

    if obj_type in ['Dict', 'dict']:
        return args

    if recursive:
        for k, v in args.items():
            if isinstance(v, dict):
                args[k] = build_from_cfg(v, registry)

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if isinstance(obj_type, str):
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.name} registry. '
                    f'Please check whether the value of `{obj_type}` is '
                    'correct or it was registered as expected.')
        elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        obj = obj_cls(**args)

        return obj


class Registry:
    def __init__(self,
                 name: str,
                 build_function: Callable = build_from_cfg):
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self.build_func = build_function

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self,
                         module: Type,
                         module_name: Optional[Union[str, List[str]]] = None,
                         force: bool = False) -> None:
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self._module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                               f'at {existed_module.__module__}')
            self._module_dict[name] = module

    def register_module(self,
                        name: Optional[str] = None,
                        force: bool = False,
                        module: Optional[Type] = None):

        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    def get(self, key: str) -> Optional[Type]:
        obj_cls = self._module_dict.get(key, None)
        return obj_cls

    def build(self, cfg, recursive=True, *args, **kwargs):
        return self.build_func(cfg, *args, **kwargs, registry=self, recursive=recursive)


if __name__ == '__main__':
    TEST = Registry('test')

    @TEST.register_module(name='ABC')
    class A:
        def __init__(self, a):
            self.a = a


    @TEST.register_module(name='B')
    class B:
        def __init__(self, x):
            self.x = x


    test_cfg = dict(type='ABC',
                    a=dict(
                        type='B',
                        x=1
                    ))
    result = build_from_cfg(test_cfg, TEST)
    print(result.a)

