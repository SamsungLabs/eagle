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

import itertools
import collections


class ModelRequirements():
    dtypes_and_format = collections.namedtuple('dtypes_and_format', ['data_type', 'compute_type', 'data_format'])
    def __init__(self, frameworks=None, data_types_and_formats=None, max_dims=None, min_dims=None):
        ''' For all arguments ``None`` means "don't care".

            Arguments:
                frameworks : list of supported frameworks
                data_types_and_formats : list of tuples (data type, compute type, data format)
                    defining what combinations of the three parameters are supported
                max_dims : maximum number of dimensions
                min_dims : minimum number of dimensions
        '''
        self.frameworks = set(frameworks) if frameworks is not None else frameworks
        self.data_types_and_formats = set(ModelRequirements.dtypes_and_format(*v) for v in data_types_and_formats) if data_types_and_formats is not None else data_types_and_formats
        self.max_dims = max_dims
        self.min_dims = min_dims

    @staticmethod
    def check_nones(t1, t2):
        if t1 is None:
            if t2 is None:
                return True, None
            return True, t2
        if t2 is None:
            return True, t1

        return False, None

    @staticmethod
    def is_container(t):
        if isinstance(t, str) or isinstance(t, bytes):
            return False
        try:
            _ = iter(t)
        except AttributeError:
            return False

        return True

    @staticmethod
    def intersect_single(t1, t2):
        quick, value = ModelRequirements.check_nones(t1, t2)
        if quick:
            return value

        ret = t1.intersection(t2)
        #if not ret:
        #    raise ValueError('Incompatible {} and {}'.format(t1, t2))
        return ret

    @staticmethod
    def intersect_multiple(t1, t2):
        quick, value = ModelRequirements.check_nones(t1, t2)
        if quick:
            return value

        if type(t1) is not type(t2):
            raise TypeError('Different types intersected: {} and {}'.format(type(t1).__name__, type(t2).__name__))

        if not ModelRequirements.is_container(t1):
            if t1 != t2:
                raise ValueError('Incompatible {} and {}'.format(t1, t2))
            return t1

        good = []
        if type(t1) is set:
            values_iter = itertools.product(t1, t2)
        else:
            if len(t1) != len(t2):
                raise ValueError('Equal size expected: {} and {}'.format(t1, t2))

            values_iter = zip(t1, t2)

        for value1, value2 in values_iter:
            try:
                value = ModelRequirements.intersect_multiple(value1, value2)
            except ValueError:
                continue

            good.append(value)

        if type(t1) is not set and len(good) != len(t1):
            raise ValueError('Incompatible {} and {}'.format(t1, t2))

        if type(t1) is ModelRequirements.dtypes_and_format:
            result = type(t1)(*good)
        else:
            result = type(t1)(good)

        return result

    @staticmethod
    def min(t1, t2):
        quick, value = ModelRequirements.check_nones(t1, t2)
        if quick:
            return value

        return min(t1, t2)

    @staticmethod
    def max(t1, t2):
        quick, value = ModelRequirements.check_nones(t1, t2)
        if quick:
            return value

        return max(t1, t2)

    def intersection(self, other):
        frameworks = ModelRequirements.intersect_single(self.frameworks, other.frameworks)
        data_types_and_formats = ModelRequirements.intersect_multiple(self.data_types_and_formats, other.data_types_and_formats)
        max_dims = ModelRequirements.min(self.max_dims, other.max_dims)
        min_dims = ModelRequirements.max(self.min_dims, other.min_dims)
        return ModelRequirements(frameworks=frameworks, data_types_and_formats=data_types_and_formats, max_dims=max_dims, min_dims=min_dims)

    def __bool__(self):
        if self.frameworks is not None and not self.frameworks:
            return False
        if self.data_types_and_formats is not None and not self.data_types_and_formats:
            return False
        if self.max_dims is not None and self.max_dims <= 0:
            return False
        if self.max_dims is not None and self.min_dims is not None and self.min_dims > self.max_dims:
            return False
        return True

    @property
    def framework(self):
        if self.frameworks is None:
            return None
        if not self.frameworks:
            raise ValueError('No compatible framework found')
        return next(iter(self.frameworks))

    @property
    def data_type(self):
        if self.data_types_and_formats is None:
            return None
        if not self.data_types_and_formats:
            raise ValueError('No compatible data type found')
        return next(iter(self.data_types_and_formats)).data_type

    @property
    def compute_type(self):
        if self.data_types_and_formats is None:
            return None
        if not self.data_types_and_formats:
            raise ValueError('No compatible compute type found')
        return next(iter(self.data_types_and_formats)).compute_type

    @property
    def data_format(self):
        if self.data_types_and_formats is None:
            return None
        if not self.data_types_and_formats:
            raise ValueError('No compatible data format found')
        return next(iter(self.data_types_and_formats)).data_format


class ModelInfo():
    def __init__(self,
                framework,
                inputs,
                outputs,
                data_type,
                compute_type,
                data_format):
        '''
            Arguments:
                framework : framework name
                inputs : list of input dimensions
                outputs : list of output dimensions
                data_type : data type for weights
                compute_type : data type for compute (and activations)
                data_format : storage format
        '''
        self.framework = framework
        self.inputs = inputs
        self.outputs = outputs
        self.data_type = data_type
        self.compute_type = compute_type
        self.data_format = data_format

    def get_matching_requirements(self):
        return ModelRequirements(
            frameworks=[self.framework],
            data_types_and_formats=[
                (self.data_type, self.compute_type, self.data_format)
            ],
            max_dims=max(len(tensor) for tensors_list in [self.inputs, self.outputs] for tensor in tensors_list),
            min_dims=max(len(tensor) for tensors_list in [self.inputs, self.outputs] for tensor in tensors_list)
        )

    def check(self, requirements):
        my_req = self.get_matching_requirements()
        if not my_req.intersection(requirements):
            raise ValueError('Model incompatible with requirements: {} and {}'.format(self, requirements))


if __name__ == '__main__':
    r1 = ModelRequirements(data_types_and_formats=[
        ('float32', 'float32', 'channels_last'),
        ('float32', 'float32', 'channels_first'),
        ('uint8', 'uint8', 'channels_last'),
        ('uint8', 'uint8', 'channels_first'),
        ('uint8', 'float32', 'channels_last'),
        ('uint8', 'float32', 'channels_first')
    ])

    r2 = ModelRequirements(data_types_and_formats=[
        ('uint8', None, None)
    ])

    r3 = ModelRequirements(data_types_and_formats=[
        (None, 'uint8', 'channels_last')
    ])

    r4 = ModelRequirements(data_types_and_formats=[
        (None, None, 'channels_first'), # whatever if channels_first
        ('uint8', 'uint8', 'channels_last'), # special case
        (None, 'float32', None) # whatever if float32 computations
    ])

    import pprint
    pprint.pprint(r1.intersection(r2).data_types_and_formats)
    pprint.pprint(r1.intersection(r3).data_types_and_formats)
    pprint.pprint(r2.intersection(r3).data_types_and_formats)
    pprint.pprint(r2.intersection(r4).data_types_and_formats)
    pprint.pprint(r3.intersection(r4).data_types_and_formats)
