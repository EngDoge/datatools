from typing import Any, NoReturn
from collections import defaultdict


class EasyDict(defaultdict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        self[name] = value

    def __delattr__(self, name: str) -> NoReturn:
        del self[name]


if __name__ == '__main__':
    a = EasyDict(int)
