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

import pathlib

import tensorflow as tf
if tf.__version__.startswith('2.'):
    tf = tf.compat.v1

from . import utils
from .. import tf_utils


def nor_conv_1x1(net, channels, data_format, data_type):
    net = tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=(1,1),
        strides=(1,1),
        padding='same',
        data_format='channels_last',
        activation=tf.nn.relu,
        dtype=data_type,
        use_bias=False
    )(net)
    # we skip batch normalization and merge relu with conv to optimize latency
    return net

def nor_conv_3x3(net, channels, data_format, data_type):
    net = tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        dtype=data_type,
        use_bias=False
    )(net)
    # we skip batch normalization and merge relu with conv to optimize latency
    return net

def avg_pool_3x3(net, unused_channels, data_format, data_type):
    net = tf.keras.layers.AvgPool2D(
        pool_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        dtype=data_type
    )(net)
    return net

def reduction_avg_pool_2x2(net, unused_channels, data_format, data_type):
    net = tf.keras.layers.AvgPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='valid',
        data_format=data_format,
        dtype=data_type
    )(net)
    return net

def reduction_conv_1x1(net, channels, data_format, data_type):
    net = tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=(1,1),
        strides=(1,1),
        padding='same',
        use_bias=False,
        data_format=data_format,
        activation=None,
        dtype=data_type
    )(net)
    return net

def reduction_conv_3x3_stride_2(net, channels, data_format, data_type):
    net = tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=(3,3),
        strides=(2,2),
        padding='same',
        data_format=data_format,
        activation=None,
        dtype=data_type,
        use_bias=False
    )(net)
    return net

def reduction_conv_3x3_stride_1(net, channels, data_format, data_type):
    net = tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        activation=None,
        dtype=data_type,
        use_bias=False
    )(net)
    return net

def eltwise_add(net, channels, data_format, data_type):
    net = tf.keras.layers.Add()([net,net])
    return net

_opindex_to_ctor = {
    '2': lambda *a: nor_conv_1x1(*a),
    '3': lambda *a: nor_conv_3x3(*a),
    '4': lambda *a: avg_pool_3x3(*a),
    '5': lambda *a: reduction_avg_pool_2x2(*a),
    '6': lambda *a: reduction_conv_1x1(*a),
    '7': lambda *a: reduction_conv_3x3_stride_2(*a),
    '8': lambda *a: reduction_conv_3x3_stride_1(*a),
    '9': lambda *a: eltwise_add(*a)
}


def build_ops(input_size=(1,32,32,3), stacks_count=3, cells_count=5, num_classes=10, data_format='channels_last', data_type='float32'):
    input_placeholder = tf.placeholder(data_type, input_size, name='input')
    net = input_placeholder

    if len(net.shape) < 4:
        if len(net.shape) < 3 and data_format == 'channels_last':
            input = tf.expand_dims(input, -1)

        while len(net.shape) < 4:
            input = tf.expand_dims(input, 0)
    
    nets = []

    with tf.variable_scope('nasbench201_dnn'):
        with tf.variable_scope('stem'):
            input = net
            net = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(3,3),
                strides=(1,1),
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu,
                dtype=data_type,
                use_bias=False
            )(input)

        nets.append((net, input, 'stem'))

        channels = 16
        for stack in range(stacks_count):
            with tf.variable_scope(f'stack{stack}'):
                # Only need to profile one cell per stack
                with tf.variable_scope('cell'):
                    input = net
                    for op in ['2','3','4']:
                        net = _opindex_to_ctor[op](input, channels, data_format, data_type)
                        nets.append((net, input, f'stack{stack}_{op}'))
                    input = net
                    net = _opindex_to_ctor['9'](input, channels, data_format, data_type)
                    nets.append((net, input, f'stack{stack}_9'))

                if stack + 1 != stacks_count:
                    with tf.variable_scope('reduction'):
                        temp = net
                        for op in ['5','6']:
                            input = net
                            net = _opindex_to_ctor[op](input, channels * 2, data_format, data_type)
                            nets.append((net, input, f'stack{stack}_reduction_{op}'))
                        net = temp
                        for op in ['7','8']:
                            input = net
                            net = _opindex_to_ctor[op](input, channels * 2, data_format, data_type)
                            nets.append((net, input, f'stack{stack}_reduction_{op}'))
                        input = net
                        net = _opindex_to_ctor['9'](input, channels, data_format, data_type)
                        nets.append((net, input, f'stack{stack}_reduction_9'))

            channels = channels * 2

        with tf.variable_scope('last'):
            input = net
            net = tf.keras.layers.AvgPool2D(
                pool_size=net.shape[1:3],
                strides=(1,1),
                padding='valid',
                data_format=data_format,
                dtype=data_type
            )(input)
            nets.append((net, input, 'last_0'))

            input = net
            net = tf.reshape(input, [net.shape[0], -1])
            nets.append((net, input, 'last_1'))

            input = net
            net = tf.keras.layers.Dense(
                units=num_classes,
                use_bias=True,
                dtype=data_type
            )(input)
            nets.append((net, input, 'last_2'))

    return nets


def build_cell(net, matrix, ops, channels, data_format, data_type):
    if len(matrix) != len(ops):
        raise ValueError('Dimensions mismatch')
    for row in matrix:
        if len(row) != len(ops):
            raise ValueError('Adjacency matrix not square')
    if ops[0] != 'input':
        raise ValueError('First operation should be "input"')
    if ops[-1] != 'output':
        raise ValueError('Last operation should be "output"')

    if len(ops) == 2:
        return net

    tensors = [net]
    for node in range(1, len(ops)):
        op = ops[node]
        inputs = [i for i in range(len(matrix)) if matrix[i][node]]
        input_tensors = [tensors[i] for i in inputs]
        assert input_tensors, f'{node}, {matrix}, {ops}'

        # Add op inputs together
        op_input = input_tensors[0]
        for input_tensor in input_tensors[1:]:
            op_input += input_tensor

        if op != 'output':
            output = _opindex_to_ctor[op](op_input, channels, data_format, data_type)
        else:
            output = op_input

        tensors.append(output)

    return tensors[-1]


def build_reduction(net, channels, data_format, data_type):
    ''' Args:
            net : input tensor
            channels : output channels (i.e. after reduction)
            data_format : 'channels_last' or 'channels_first', see TF documentation
    '''
    with tf.variable_scope('residual'):
        residual = tf.keras.layers.AvgPool2D(
            pool_size=(2,2),
            strides=(2,2),
            padding='valid',
            data_format=data_format,
            dtype=data_type
        )(net)

        residual = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=(1,1),
            strides=(1,1),
            padding='same',
            use_bias=False,
            data_format=data_format,
            activation=None,
            dtype=data_type
        )(residual)

        net = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=(3,3),
            strides=(2,2),
            padding='same',
            data_format=data_format,
            activation=None,
            dtype=data_type,
            use_bias=False
        )(net)

        net = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            data_format=data_format,
            activation=None,
            dtype=data_type,
            use_bias=False
        )(net)

        return net + residual


def build_model(arch_vector, input_size=(1,32,32,3), stacks_count=3, cells_count=5, num_classes=10, data_format='channels_last', data_type='float32'):
    ''' Args:
            arch_vector : should be a point from the nasbench201 search space, as used by the Nasbench201 backend in autocaml
                (i.e. 6 values from range [0-4])
    '''
    matrix, ops = utils.get_matrix_and_ops(arch_vector)
    assert bool(matrix) == bool(ops)
    if not matrix:
        return None, None

    try:
        data_type = getattr(tf, data_type)
    except AttributeError:
        raise ValueError('Unsupported data type: {}'.format(data_type))

    input_placeholder = tf.placeholder(data_type, input_size, name='input')
    net = input_placeholder

    if len(net.shape) < 4:
        if len(net.shape) < 3 and data_format == 'channels_last':
            net = tf.expand_dims(net, -1)

        while len(net.shape) < 4:
            net = tf.expand_dims(net, 0)


    with tf.variable_scope('nasbench201_dnn'):
        # Initial stem convolution
        with tf.variable_scope('stem'):
            net = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(3,3),
                strides=(1,1),
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu,
                dtype=data_type,
                use_bias=False
            )(net)

        channels = 16
        for stack in range(stacks_count):
            with tf.variable_scope(f'stack{stack}'):
                for cell in range(cells_count):
                    with tf.variable_scope(f'cell{cell}'):
                        net = build_cell(net, matrix, ops, channels, data_format, data_type)

                if stack + 1 != stacks_count:
                    net = build_reduction(net, channels * 2, data_format, data_type)

            channels = channels * 2

        net = tf.keras.layers.AvgPool2D(
            pool_size=net.shape[1:3],
            strides=(1,1),
            padding='valid',
            data_format=data_format,
            dtype=data_type
        )(net)

        net = tf.reshape(net, [net.shape[0], -1])

        net = tf.keras.layers.Dense(
            units=num_classes,
            use_bias=True,
            dtype=data_type
        )(net)

    return net, input_placeholder


def build_graph(*args, **kwargs):
    ''' The same as :py:func:`build_model` but captures the returned network within a new ``tf.Graph`` object which is returned
        together with the output tensor (required for performing some operations on the graph)
    '''
    graph = tf.Graph()
    with graph.as_default():
        logits, _ = build_model(*args, **kwargs)

    if logits is None:
        return None, None

    return graph, logits


def build_and_convert_to_graph_def(arch_vector, profile_layers=None, graph_dir=None, output_file=None, checkpoint=None, **kwargs):
    ''' Creates a TF graph according to architecture vector 'arch_vector'.
        The created graph is converted to TF graph_def format and returned, if the 'arch_vector' describes a valid model,
        otherwise None is returned.
        Optionally, if 'graph_dir' is a non-empty string, the original TF Graph (i.e. before conversion)
        will be saved to the directory pointed by the argument (using summary mechanism).
        The converted model can also be optionally saved in the file specified by 'output_file'.

        The network created is not trained and is meant to only be used to measure inference time on-device.

        Args:
            arch_vector : list of integers describing nasbench201 model, to obtain an arch vector from an arch string see `get_arch_vector_from_arch_str`
            
            graph_dir : optional path to a directory in which a tf summary of the graph will be saved (to view in tensorboard), see `save_graph`
        
        The following arguments are passed directly to `convert_to_graph_def`:
            output_file : file which will contain tf model
            checkpoint : specifies a checkpoint to load before converting

        The remaining arguments are passed directly to `build_model`.

        Returns:
            Graph definition of the TF model if 'arch_vector' describes a valid model,
            otherwise None.
    '''

    graph = tf.Graph()
    with graph.as_default():
        if profile_layers:
            nets = build_ops(**kwargs)
            graph_defs = []
            for logits, input, tag in nets:
                graph_def = build_graph_def(logits, input, graph, graph_dir, output_file, checkpoint, input.get_shape())
                graph_def.append(tag)
                graph_defs.append(graph_def)
            return graph_defs
        else:
            logits, input_placeholder = build_model(arch_vector, **kwargs)
            return build_graph_def(logits, input_placeholder, graph, graph_dir, output_file, checkpoint, kwargs['input_size'])


def build_graph_def(logits, input_placeholder, graph, graph_dir, output_file, checkpoint, input_size):
    if logits is not None:
        if graph_dir:
            gpath = pathlib.Path(graph_dir)
            if output_file:
                gpath = gpath / pathlib.Path(output_file).name
            tf_utils.save_graph(graph, gpath)
        graph_def = tf_utils.convert_to_graph_def(graph, [logits], output_file=output_file, checkpoint=checkpoint)
        return [graph_def, input_placeholder.name, logits.name, input_size]
    else:
        return None


def build_and_convert_to_tflite(arch_vector, graph_dir=None, output_file=None, checkpoint=None, quantize_weights=True, quantize_activations=True, **kwargs):
    ''' Creates a tf graph (optimized for inference, not training) according to architecture vector 'arch_vector'.
        The created graph is converted to TFLite format and returned, if the 'arch_vector' describes a valid model,
        otherwise None is returned.
        Optionally, if 'graph_dir' is a non-empty string, the original TF Graph (i.e. before conversion)
        will be saved to the directory pointed by the argument (using summary mechanism).
        The converted model can also be optionally saved in the file specified by 'output_file'.

        The network created is not trained and is meant to only be used to measure inference time on-device.

        Args:
            arch_vector : list of integers describing nasbench201 model, to obtain an arch vector from an arch string see `get_arch_vector_from_arch_str`
            
            graph_dir : optional path to a directory in which a tf summary of the graph will be saved (to view in tensorboard), see `save_graph`
        
        The following arguments are passed directly to `convert_to_tflite`:
            output_file : file which will contain tflite model
            checkpoint : specifies a checkpoint to load before converting
            quantize_weights : whether to use the default post-training quantization
            quantize_activations : whether to perform computations on quantized data (requires quantization of activations)

        The remaining arguments are passed directly to `build_model`.

        Returns:
            Binary representation of the converted TFLite model  if 'arch_vector' describes a valid model,
            otherwise None.
    '''
    graph = tf.Graph()
    with graph.as_default():
        logits, input_placeholder = build_model(arch_vector, **kwargs)

    if logits is not None:
        if graph_dir:
            gpath = pathlib.Path(graph_dir)
            if output_file:
                gpath = gpath / pathlib.Path(output_file).name
            tf_utils.save_graph(graph, gpath)
        return tf_utils.convert_to_tflite(graph, [input_placeholder], [logits], output_file=output_file, checkpoint=checkpoint, quantize_weights=quantize_weights, quantize_activations=quantize_activations)
    else:
        return None


def build_and_convert_to_trt(arch_vector, graph_dir=None, output_file=None, checkpoint=None, quantize_weights=True, quantize_activations=True, **kwargs):
    graph = tf.Graph()
    with graph.as_default():
        logits, input_placeholder = build_model(arch_vector, **kwargs)

    if logits is not None:
        trt_graph = tf_utils.convert_to_trt(graph, [input_placeholder], [logits], output_file=output_file, checkpoint=checkpoint, quantize_weights=quantize_weights, quantize_activations=quantize_activations)
        return [trt_graph, input_placeholder.name, logits.name, kwargs['input_size']]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch_str', type=str, default=None, help='Architecture string of a model to build and convert to TFLite. Use this option or --arch_vec')
    parser.add_argument('--arch_vec', type=str, default=None, help='Architecture vector (comma-separated list of integers) of a model to build and convert to TFLite. Use this option or --arch_str')
    parser.add_argument('--noquantize', action='store_true', help='Disable default post-training quantization')
    parser.add_argument('--save_graph', type=str, default=None, help='Directory name where the TF graph will be saved. By default (or if the path is empty) the graph is not saved.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint name to load before conversion. By default (or if the path is empty) no checkpoint is loaded and the model is deployed using untrained weights.')
    parser.add_argument('--input_size', type=str, default=None, help='Input size to use. By default: 1,32,32,3 is used.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for the last dense layer. By default 10.')
    parser.add_argument('--stacks', type=int, default=3, help='Number of stacks in the macro model. By default 3.')
    parser.add_argument('--cells', type=int, default=5, help='Number of cells per stack in the macro model. By default 5.')
    parser.add_argument('--channels_first', action='store_true', help='Use channels_first data format instead of channels_last')
    parser.add_argument('output', type=str, help='Output filename where the converted model will be saved, "-" represents standard output.')

    print_model = False
    args = parser.parse_args()
    if args.output == '-':
        args.output = None
        print_model = True

    if args.arch_str and args.arch_vec:
        raise ValueError('Use either --arch_str or --arch_vec but not both')

    arch_vec = None
    if args.arch_str:
        arch_vec = utils.get_arch_vector_from_arch_str(args.arch_str)
    elif args.arch_vec:
        arch_vec = [int(i) for i in args.arch_vec.split(',')]
    else:
        raise ValueError('Either --arch_str or --arch_vec should be specified')

    if len(arch_vec) != 6:
        raise ValueError('Invalid arch_vec')
    for value in arch_vec:
        if value < 0 or value >= 5:
            raise ValueError('Invalid arch_vec')

    if args.input_size:
        input_size = (int(i) for i in args.input_size.split(','))
        if len(input_size) != 4:
            raise ValueError('Invalid input_size')
    else:
        if not args.channels_first:
            input_size = (1,32,32,3) 
        else:
            input_size = (1,3,32,32)

    model = build_and_convert(arch_vec, 
        graph_dir=args.save_graph,
        output_file=args.output,
        checkpoint=args.checkpoint,
        quantize=not args.noquantize,
        input_size=input_size,
        stacks_count=args.stacks,
        cells_count=args.cells,
        num_classes=args.num_classes,
        data_format='channels_last' if not args.channels_first else 'channels_first')

    if print_model:
        print(model)
