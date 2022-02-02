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


def disable_warnings():
    import os
    import warnings
    warnings.filterwarnings('ignore',category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        from tensorflow.python.util import module_wrapper as deprecation
    except ImportError:
        try:
            from tensorflow.python.util import deprecation_wrapper as deprecation
        except ImportError:
            from tensorflow.python.util import deprecation
    try:
        deprecation._PRINT_DEPRECATION_WARNINGS = False
    except:
        pass

    import tensorflow as tf
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except:
        pass

    try:
        tf.get_logger().setLevel('ERROR')
    except:
        pass


def save_graph(graph, output_dir):
    ''' Save a summary of the given TF Graph in the specified directory (to be viewed in tensorboard).

        Args:
            graph : tf.Graph to be saved
            output_dir : a directory to which the summary should be written
    '''
    import tensorflow as tf
    with tf.summary.FileWriter(logdir=output_dir, graph=graph) as _:
        pass

def convert_to_graph_def(graph, output, output_file=None, checkpoint=None):
    ''' Converts the given TF Graph as a graph_def and optionally save it to a protobuf file.

        Args:
            graph : tf.Graph to convert and save
            
            outputs : a list of tensors which are outputs of the network
            output_file : (optional) a filename to which the converted model will be saved
                This argument can be omitted if the user wants to skip saving to a file, in that
                case the returned value can be used instead of saving and then reading from the file
            checkpoint : (optional) a path to a checkpoint which will be loaded before converting
                (i.e. the weights from this checkpoint will be used in the converted model)
        Returns:
            the graph definition of the converted TF model
    '''
    output = [tensor.name[:-2] for tensor in output]
    import tensorflow as tf

    sessconfig = tf.ConfigProto(device_count = {'GPU': 0})
    sessconfig.gpu_options.allow_growth = True
    sessconfig.gpu_options.force_gpu_compatible = True

    with tf.Session(config=sessconfig, graph=graph) as sess:
        with graph.as_default():
            sess.run(tf.global_variables_initializer())
            if checkpoint is not None:
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint)
            
        graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output)
        output_graph_def = tf.graph_util.remove_training_nodes(graph_def, protected_nodes=output)

        if output_file:
            import os
            output_file = pathlib.Path(output_file)
            os.makedirs(output_file.parent, exist_ok=True)
            tf.train.write_graph(output_graph_def, str(output_file.parent), str(output_file.name), as_text=False)

        return output_graph_def

def convert_to_tflite(graph, inputs, outputs, output_file=None, checkpoint=None, quantize_weights=True, quantize_activations=True):
    ''' Converts the given TF Graph as a TFLite model and optionally save it to a file.

        Args:
            graph : tf.Graph to convert and save
            
            inputs : a list of tensors which are inputs to the network
            outputs : a list of tensors which are outputs of the network
            output_file : (optional) a filename to which the converted model will be saved
                This argument can be omitted if the user wants to skip saving to a file, in that
                case the returned value can be used instead of saving and then reading from the file
            checkpoint : (optional) a path to a checkpoint which will be loaded before converting
                (i.e. the weights from this checkpoint will be used in the converted model)
            quantize_weights : whether to enable the default post-training quantization on weights
            quantize_activations : whether to enable the default post-training quantization on activations
        Returns:
            the binary form of the converted TFLite model (i.e. the content of the .tflite file)
    '''
    import tensorflow as tf
    if tf.__version__.startswith('2.'):
        tf = tf.compat.v1

    sessconfig = tf.ConfigProto(device_count = {'GPU': 0})
    sessconfig.gpu_options.allow_growth = True
    sessconfig.gpu_options.force_gpu_compatible = True

    with tf.Session(config=sessconfig, graph=graph) as sess:
        with graph.as_default():
            sess.run(tf.global_variables_initializer())
            if checkpoint is not None:
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint)

        converter = tf.lite.TFLiteConverter.from_session(sess, inputs, outputs)
        converter.post_training_quantize = quantize_weights
        if quantize_activations:
            converter.inference_type = tf.uint8
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            converter.quantized_input_stats = { 'input': (0,1) }
            converter.default_ranges_stats = (-1,1)

        tflite_model = converter.convert()

        if output_file:
            import os
            output_file = pathlib.Path(output_file)
            os.makedirs(output_file.parent, exist_ok=True)
            with open(output_file, 'wb') as of:
                of.write(tflite_model)

    return tflite_model


def convert_to_trt(graph, inputs, outputs, output_file=None, checkpoint=None, quantize_weights=True, quantize_activations=True):
    ''' Converts the given TF Graph as a TF model optimized by TensorRT and optionally save it to a file.

        Args:
            graph : tf.Graph to convert and save
            
            inputs : a list of tensors which are inputs to the network
            outputs : a list of tensors which are outputs of the network
            output_file : (optional) a filename to which the converted model will be saved
                This argument can be omitted if the user wants to skip saving to a file, in that
                case the returned value can be used instead of saving and then reading from the file
            checkpoint : (optional) a path to a checkpoint which will be loaded before converting
                (i.e. the weights from this checkpoint will be used in the converted model)
            quantize_weights : whether to enable the default post-training quantization on weights
            quantize_activations : whether to enable the default post-training quantization on activations
        Returns:
            the binary form of the converted TFLite model (i.e. the content of the .tflite file)
    '''
    import tensorflow as tf
    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    with tf.Session(graph=graph) as sess:
        with graph.as_default():
            sess.run(tf.global_variables_initializer())
            if checkpoint is not None:
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint)
            
        output = [tensor.name[:-2] for tensor in outputs]
        graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output)
        frozen_graph = tf.graph_util.remove_training_nodes(graph_def, protected_nodes=output)

        if quantize_weights or quantize_activations:
            precision_mode = "FP16"
        else:
            precision_mode = "FP32"

        converter = trt.TrtGraphConverter(
            input_graph_def=frozen_graph,
            max_batch_size=1,
            max_workspace_size_bytes=1<<25,
            minimum_segment_size=10,
            precision_mode=precision_mode,
            use_calibration=True,
            is_dynamic_op=True,
            nodes_blacklist=output)
        frozen_graph = converter.convert()

        if output_file:
            import os
            output_file = pathlib.Path(output_file)
            os.makedirs(output_file.parent, exist_ok=True)
            tf.train.write_graph(frozen_graph, str(output_file.parent), str(output_file.name), as_text=False)
        
    return frozen_graph

def get_flops_and_params(graph, output):
    if graph is None:
        return 0, 0

    graph_def = convert_to_graph_def(graph, output)

    import tensorflow as tf
    run_meta = tf.RunMetadata()
    with tf.Graph().as_default() as graph_opt:
        tf.import_graph_def(graph_def, name='')
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(graph_opt, run_meta=run_meta, cmd='op', options=opts)

    with graph.as_default():
        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(graph, run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops, params.total_parameters
