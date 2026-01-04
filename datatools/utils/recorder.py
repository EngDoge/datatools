from collections import defaultdict
from datatools.utils.dict import EasyDict



class ActionRecorder:
    def __init__(self, obj_type):
        self.actions = defaultdict(ActionRecorder.default_fn)
        self.obj_type = obj_type

    @staticmethod
    def default_fn():
        return EasyDict(list)

    def items(self):
        for key, value in self.actions.items():
            yield key, value.args, value.kwargs

    def __getitem__(self, item):
        return self.actions[item]

    def duplicate_action(self, obj):
        assert isinstance(obj, self.obj_type), f'Current ActionRecorder only works for {self.obj_type}'
        for action, args, kwargs in self.items():
            for iter_args, iter_kwargs in zip(args, kwargs):
                todo = getattr(obj, action)
                todo(*iter_args, **iter_kwargs)

    def __str__(self):
        ret = '\n'.join([f'{self.obj_type.__name__}.{fn}(*{args}, **{kwargs})'
                         for fn in self.actions
                         for args, kwargs in zip(self.actions[fn].args, self.actions[fn].kwargs)])
        return ret




if __name__ == '__main__':
    from functools import wraps

    class TClass:
        def __init__(self):
            self.recorder = ActionRecorder(TClass)
            self.record_action = True
            self.ttt = 1

        def record(fn):
            @wraps(fn)
            def wrapped_fn(self, *args, **kwargs):
                if self.record_action:
                    self.recorder.actions[fn.__name__].args.append(args)
                    self.recorder.actions[fn.__name__].kwargs.append(kwargs)
                return fn(self, *args, **kwargs)

            return wrapped_fn

        @record
        def test_fn(self, *args, **kwargs):
            print(self.ttt)
            print(f'> args: {args}')
            print('setting ttt')
            print(f'> kwargs: {kwargs}')
            print('___________')

        def duplicate(self):
            self.record_action = False
            self.recorder.duplicate_action(self)
            self.record_action = True

        @record
        def __setattr__(self, key, value):
            setattr(self, key, value)

    a = TClass()
    a.test_fn(123, abc=1234)
    a.test_fn(abc=False)
    a.bbb = 3

    a.duplicate()
    print(a.ttt)
    print(a.recorder)
    # a.test_fn(*(), **{'abc': False})
    # print(a.__name__)

    # a = ActionRecorder(TClass)
    # a['123'].args.append([1, 2])
    # a['123'].kwargs.append({'abc': 2})
    # a['123'].args.append([2])
    # a['123'].kwargs.append({'abc': 4})
    # for k, _args, _kwargs in a.items():
    #     print(k, _args, _kwargs)
