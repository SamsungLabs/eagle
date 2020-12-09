#!/usr/bin/python3
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

import gc
import time
import statistics
import json
import re
import os

import numpy as np
from tensorflow.python.client import timeline

def benchmark(model, runs=1, interval=0.1, avg_between=10, device='cpu', profile_layers=False):

    import tensorflow as tf

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(model[0], name='')
    
    sessconfig = tf.ConfigProto()
    if device == 'cpu':
        sessconfig = tf.ConfigProto(device_count = {'GPU': 0})
    sessconfig.gpu_options.allow_growth = True                           
    sessconfig.gpu_options.force_gpu_compatible = True     

    with tf.Session(config=sessconfig, graph=graph) as sess:
        if profile_layers:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())    
        input = graph.get_tensor_by_name(model[1])
        output = graph.get_tensor_by_name(model[2])

        times = []
        for r in range(runs):

            random_input = tf.random.uniform(model[3]).eval(session=sess)

            gcold = gc.isenabled()
            gc.disable()
            try:
                subtimes = []
                for _ in range(avg_between):
                    if profile_layers:
                        sess.run(output, feed_dict={input: random_input}, options=run_options, run_metadata=run_metadata)
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        data = json.loads(ctf)
                        dur = 0
                        for info in data['traceEvents']:
                            if info['name'] != 'unknown' and 'dur' in info.keys():
                                dur += info['dur']
                        subtimes.append(dur / 1e6)
                    else:
                        s = time.time()
                        sess.run(output, feed_dict={input: random_input})
                        subtimes.append(time.time()-s)
            finally:
                if gcold:
                    gc.enable()

            times.append(subtimes)
            time.sleep(interval)

        return times

