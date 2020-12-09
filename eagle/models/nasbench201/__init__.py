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

from . import utils
from .. import model_info

import functools


def get_supported_models():
    return [
        model_info.ModelRequirements(
            frameworks=['tf'],
            data_types_and_formats=[
                ('float32', 'float32', None),
                ('float16', 'float16', None)
            ],
            min_dims=3
        ),
        model_info.ModelRequirements(
            frameworks=['tflite'],
            data_types_and_formats=[
                ('float32', 'float32', None),
                ('uint8', 'float32', None),
                ('uint8', 'uint8', None)
            ],
            min_dims=3
        ),
        model_info.ModelRequirements(
            frameworks=['trt'],
            data_types_and_formats=[
                ('float16', 'float16', None),
                ('uint8', 'uint8', None)
            ],
            min_dims=3
        )
    ]


def _update_args_for_build_model(model_requirements, model_args, istflite):
    if 'data_type' not in model_args:
        if not istflite:
            model_args['data_type'] = model_requirements.data_type or 'float32'
        else:
            model_args['data_type'] = 'float32'

    if 'data_format' not in model_args:
        model_args['data_format'] = model_requirements.data_format or 'channels_last'
    
    if (model_requirements.max_dims or 4) == 3:
        if model_args.get('data_format') == 'channels_last':
            model_args.setdefault('input_size', (32,32,3))
        elif model_args.get('data_format') == 'channels_first':
            model_args.setdefault('input_size', (3,32,32))
        else:
            raise ValueError('Unknown data format: {}'.format(model_args.get('data_format')))

    if (model_requirements.max_dims) == 4:
        if model_args.get('data_format') == 'channels_last':
            model_args.setdefault('input_size', (1,32,32,3))
        elif model_args.get('data_format') == 'channels_first':
            model_args.setdefault('input_size', (1,3,32,32))
        else:
            raise ValueError('Unknown data format: {}'.format(model_args.get('data_format')))

    return model_args


def get_tf_func_and_args(model_requirements, model_args):
    model_args = _update_args_for_build_model(model_requirements, model_args, False)
    model_args.setdefault('graph_dir', None)
    model_args.setdefault('output_file', None)
    model_args.setdefault('checkpoint', None)
    from .. import tf_utils
    tf_utils.disable_warnings()
    from . import tf_model
    return tf_model.build_and_convert_to_graph_def, model_args


def get_tflite_func_and_args(model_requirements, model_args):
    assert model_requirements.data_type in ['uint8', 'float32']
    assert model_requirements.compute_type in ['uint8', 'float32']
    quantize_weights = model_requirements.data_type == 'uint8'
    quantize_activations = model_requirements.compute_type == 'uint8'

    model_args = _update_args_for_build_model(model_requirements, model_args, True)
    model_args.setdefault('graph_dir', None)
    model_args.setdefault('output_file', None)
    model_args.setdefault('checkpoint', None)
    model_args.setdefault('quantize_weights', quantize_weights)
    model_args.setdefault('quantize_activations', quantize_activations)

    from .. import tf_utils
    tf_utils.disable_warnings()
    from . import tf_model
    return tf_model.build_and_convert_to_tflite, model_args


def get_trt_func_and_args(model_requirements, model_args):
    assert model_requirements.data_type in ['uint8', 'float16']
    assert model_requirements.compute_type in ['uint8', 'float16']
    quantize_weights = True
    quantize_activations = True

    model_args = _update_args_for_build_model(model_requirements, model_args, True)
    model_args.setdefault('graph_dir', None)
    model_args.setdefault('output_file', None)
    model_args.setdefault('checkpoint', None)
    model_args.setdefault('quantize_weights', quantize_weights)
    model_args.setdefault('quantize_activations', quantize_activations)

    from .. import tf_utils
    tf_utils.disable_warnings()
    from . import tf_model
    return tf_model.build_and_convert_to_trt, model_args


def get_models_ctor(model_requirements, model_args):
    frm = model_requirements.framework or 'tf'
    if frm == 'tf':
        func, args = get_tf_func_and_args(model_requirements, model_args)
    elif frm == 'tflite':
        func, args = get_tflite_func_and_args(model_requirements, model_args)
    elif frm == 'trt':
        func, args = get_trt_func_and_args(model_requirements, model_args)
    else:
        raise ValueError('Unsupported framework: {}'.format(frm))

    return functools.partial(func, **args)


get_models_iter = utils.get_models_iter
get_total_models = utils.get_total_models

get_matrix_and_ops = utils.get_matrix_and_ops
get_adjacency_and_features = utils.get_adjacency_and_features
