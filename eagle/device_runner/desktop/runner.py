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

from ..runner import DeviceRunner

import pathlib
import tempfile
import subprocess


def get_supported_metrics():
    return ['latency']


def device_side_run(metric, model, **kwargs):
    ''' A device-side entry point for evaluating models.

        This function should delegate the passed model to different implementations
        specialized for measuring different metrics. By convention, these functions are
        implements in the ``run_model`` module in the device package.

        This function should not be called directly.
        See :py:class:`DesktopRunner` for more information about arguments etc.
    '''
    assert metric in get_supported_metrics()
    from . import run_model
    if metric == 'latency':
        return run_model.benchmark(model, **kwargs)


class DesktopRunner(DeviceRunner):
    ''' A DeviceRunner class for desktop device.
        This takes a TF model and send it to the target device
        for evaluation. The on-device endpoints for different metrics currently include:

            - for ``'latency'`` target metric :py:func:`eagle.device_runner.desktop.run_model.benchmark`
              will be eventually called.

        The device-side entry point is :py:func:`device_side_run`.
    '''
    def __init__(self, metric, separate_process=False, device_addr=None, **kwargs):
        super().__init__(metric, device_side_run, separate_process=separate_process, device_addr=device_addr, **kwargs)

    def run(self, tf_model, *args, **kwargs):
        ''' Send the given TF model to a device for
            benchmarking.

            Arguments:
                tf_model : a graph definition of the TF model to run
                *args : extra arguments which will be passed to the benchmarking function
                    on device
                **kwargs : extra arguments which will be passed to the benchmarking function
                    on device
        '''
        return super().run(tf_model, *args, **kwargs)


def get_runner(metric, model_config, device_args):
    ''' Returns a runner object for desktop device.
        This is the top-level API function which needs to be defined
        on the package level to make it conformant with the main ``main.py``
        script from ``eagle.device_runner``.
    '''
    if metric not in get_supported_metrics():
        raise ValueError('Unsupported metric: {}'.format(metric))

    assert model_config.framework == 'tf'
    assert model_config.data_type == 'float32'
    assert model_config.compute_type == 'float32'

    return DesktopRunner(
        metric=metric,
        **device_args
    )
