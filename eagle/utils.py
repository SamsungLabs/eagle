# Copyright 2020 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import sys
import collections


class staticproperty(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super().__init__(fget, fset, fdel, doc)
        if doc is None and fget is not None:
            try:
                self.__doc__ = fget.__func__.__doc__
            except AttributeError:
                self.__doc__ = fget.__doc__

    def __get__(self, inst, cls=None):
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget()

    def __set__(self, inst, val):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        return self.fset(val)

    def __delete__(self, inst):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        return self.fdel()


class LazyModule():
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)


def add_module_properties(module_name, properties):
    module = sys.modules[module_name]
    replace = False
    if isinstance(module, LazyModule):
        lazy_type = type(module)
    else:
        lazy_type = type('LazyModule({})'.format(module_name), (LazyModule,), {})
        replace = True

    for name, prop in properties.items():
        setattr(lazy_type, name, prop)

    if replace:
        sys.modules[module_name] = lazy_type(module)


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n)==str for n in f)


def freeze(seq, frozen_type=tuple):
    ''' Recursively replaces all mutable sequences with immutable type provided in ``frozen_type``.
    '''
    if isinstance(seq, collections.Mapping):
        t = type(seq)
        if isinstance(seq, collections.MutableMapping):
            t = frozen_type
        return t(map(lambda pair: (freeze(pair[0]), freeze(pair[1])), seq.items()))
    elif isinstance(seq, collections.Sequence) and not isinstance(seq, str) and not isinstance(seq, range):
        t = type(seq)
        if isinstance(seq, collections.MutableSequence):
            t = frozen_type
        if isnamedtupleinstance(seq):
            return t(*map(lambda a: freeze(a), seq))
        else:
            return t(map(lambda a: freeze(a), seq))
    else:
        return seq
