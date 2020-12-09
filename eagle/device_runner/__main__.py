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

''' This is the main module containing running logic to generate datasets of performance measurements
    for different families of models by running them on a specified device.
    The module uses a predefined API to communicate with the model
    implementation and the device implementation in order to drive its execution.

    The following list of functions is expected to be exposed by the model package:

        - ``get_supported_models()`` should return a list of :py:class:`eagle.models.model_info.ModelRequirements`
          objects specifying what kind of models it is able to generate.

        - ``get_models_ctor(model_requirements, extra_arguments)`` should return a function capable of constructing
          all models supported by the package. The constructed models should be compliant with the set of requirements
          specified by ``model_requirements`` - if this argument includes more than one possible option, any can be used.
          The model requirements are guaranteed to be compatible with at least one of the requirements objects returned
          in the list produced by ``get_supported_models()`` of the same package, they are calculated by considering
          capabilities of both the device and the model packages - the ``get_models_ctor`` function will not be called
          if a common set of model requirements cannot be computed.

          The ``extra_arguments`` argument will be a dict of extra arguments provided by the user which should be used
          when constructing models. The driver code does not check these and its the user's responsibility to make
          sure they can be consumed correctly. For example, current code in `nasbench201` package assumes that the
          extra arguments are arguments to the :py:func:`eagle.models.nasbench201.tf_model.build_model` function.

          The returned function will be called with different values obtained by iterating over the iterator returned
          by ``get_models_iter`` in a manner represented by the following code::

                ctor = module.get_models_ctor(requirements, extra_args)
                for model_idx in module.get_models_iter():
                    model = ctor(model)

          The constructor can return ``None`` if the model configuration is invalid, in which case the model will not be
          passed to the device and instead a string ``"invalid"`` will be saved in the dataset in place of measurements.

        - ``get_models_iter(start=None)`` should return an iterator (staring at position ``start`` if given -- the first
          value returned should be the following one, i.e. ``start`` should *NOT* be returned) iterating
          over (identifiers of) all models supported by the package. The type of the identifier is not restricted by the API
          so it is up to the package implementation what is used. The functional requirements for the returned value are
          the following:

              - it should be an iterator, i.e. it should be possible to use it in a for loop,
              - the values obtained by iterating over it should be valid arguments to the constructor function
                returned by ``get_models_ctor`` function (se its documentation for more information)
              - the values should be pickable
              - the number of values returned by iterating over an iterator staring at ``None`` should be equal
                to the number returned by ``get_total_models``. In other words::

                      len(module.get_models_iter) == module.get_total_models()

            The identifiers might include invalid models as long as they are they are valid inputs for the constructor function,
            as described above. The invalid models should also be counted towards ``get_total_models``.

        - ``get_total_models()`` should return a total number of different models the package is able to produce.
          It used purely to provide the user with information about progress of the main script.

    The following functions should be implemented by the device package:

        - ``get_supported_metrics()`` should return a list of metrics which can be measured on the target device
          (such as ``'latency'``, ``'power'`` etc.).

        - ``get_model_requirements()`` should return a :py:class:`eagle.models.model_info.ModelRequirements` object
          representing a set of requirements which a model has to meet in order for the device to be able to run it.

        - ``get_runner(metric, target_config, device_args)`` should return a runner object which needs to expose 
          a single function: ``run(model)``. This function will be called with different models and is supposed to
          do everything necessary to obtain measurements of the ``metric`` passed to the ``get_runner`` function.
          The output of the ``run`` function is what is directly stored in the measurements dataset in a model
          is valid.

          The ``target_config`` argument passed to the ``get_runner`` function is a requirements objects
          holding information about requirements the models passed to the ``run`` function will meet. These
          requirements are guaranteed to be compatible with those returned by ``get_model_requirements`` function
          of the same package (see also ``get_supported_models`` function of the model package).

          The ``device_args`` argument will be a dict of extra arguments provided by the user which should be used
          when running measurements. The driver code does not check these and its the user's responsibility to make
          sure they can be consumed correctly. For example, current code in `edge_tpu` package assumes that the
          extra arguments are arguments to the :py:class:`eagle.device_runner.runner.DeviceRunner` class - these
          include things like an IP address of the target device etc.

          The returned runner will be used in a ``with`` statement in the following manner::

                ctor = model_module.get_models_ctor(requirements, model_args)
                with device_module.get_runner(metric, requirements, device_args) as runner:
                    for model_idx in model_module.get_models_iter():
                        model = ctor(model_idx)
                        measurements = runner.run(model)

          therefore its ``__enter__`` and ``__exit__`` methods can be used to prepare and clean up environment
          for running models, respectively.

    The results are stored using pickle in a form of a Python list split across multiple ``results{index}.pickle`` files.
    Each pickle file contains a single list with unique measurements. The splitting is currently done based on number of elements
    in a list (rather than its size in memory) - this can change later to something more reasonable. The main motivation
    behind splitting the results is to prevent saving/keeping in memory a lot of data. The files are automatically organized
    in a folder structure using the following convention::

        <root experiments folder>/<device name>/<models family name>/[<framework>/]<metric>/results*.pickle

    where ``<root experiments folder>`` can be provided by user and defaults to ``results``, ``<device name>`` is the name of the
    device package, ``<models family name>`` is the name of the model package, ``<framework>`` is an optional part if the model-device
    pair supports multiple frameworks and ``<metric>`` is the measured metric.

    The script supports checkpointing as it flushed new results and saves its current position every time new measurements
    are finished. 
'''

import time
import pickle
import pathlib
import tempfile
import importlib

from ..models import model_info

def main(outdir, metric, model_name, device_name, fix=False, framework=None, model_args=None, device_args=None):
    model_args = model_args or {}
    device_args = device_args or {}

    model_module = importlib.import_module('.' + model_name, 'eagle.models')
    device_module = importlib.import_module('.' + device_name, 'eagle.device_runner')

    if metric not in device_module.get_supported_metrics():
        raise ValueError('Device {} does not support measuring metric {}'.format(device_name, metric))

    device_requirements = device_module.get_model_requirements()
    supported_models = model_module.get_supported_models()

    extra_requirements =  model_info.ModelRequirements(frameworks=[framework] if framework is not None else None)

    target_config = None
    possible_configs = []
    for supported_model in supported_models:
        matching_config = supported_model.intersection(device_requirements)
        if matching_config:
            possible_configs.append(matching_config)
            break

    possible_frameworks = set()
    many_frameworks = False
    for cfg in possible_configs:
        if cfg.frameworks is None:
            many_frameworks = True
        possible_frameworks.union(cfg.frameworks)
        after_extra = cfg.intersection(extra_requirements)
        if after_extra:
            if target_config is not None:
                raise ValueError('Found more than more possible configuration for device {} and model {}'.format(device_name, model_name))
            target_config = after_extra

    if len(possible_frameworks) > 1:
        many_frameworks = True

    if not target_config:
        raise ValueError('Could not find a compatible configuration for device {} and model {}'.format(device_name, model_name))
    if target_config.framework is None:
        raise ValueError('Could not determine target framework')

    if many_frameworks:
        outdir = pathlib.Path(outdir) / model_name / metric / device_name / target_config.framework
    else:
        outdir = pathlib.Path(outdir) / model_name / metric / device_name
    outdir.mkdir(parents=True, exist_ok=True)

    step_file = outdir / 'step.txt'
    results_folder = outdir / 'results'
    results_folder.mkdir(parents=True, exist_ok=True)

    if device_args['target_args'].get('profile_layers') == True:
        models_ctor = model_module.get_models_ctor(target_config, model_args)
        results_file = results_folder / 'results_layers.pickle'
        with open(results_file, 'wb') as f:
            data = []
            with device_module.get_runner(metric, target_config, device_args) as runner:
                models = models_ctor(None, profile_layers=True)
                for model in models:
                    run_result = runner.run(model)
                    print(model[2], model[4])
                    data.append((model[4], run_result))
            f.write(pickle.dumps(data))
        return

    step, point, rfile = 0, None, 0
    if step_file.exists():
        step, point, rfile = eval(step_file.read_text())

    if fix:
        target_steps = step
        target_point = point
        point = None
        step = 0
        existing = set()
        for f in range(rfile+1):
            results_file = results_folder / 'results{}.pickle'.format(f)
            if results_file.exists():
                print('Reading existing results from file: {}'.format(results_file))
                for pt, _ in pickle.loads(results_file.read_bytes()):
                    existing.add(tuple(pt))

        print('Found {} existing measurements'.format(len(existing)))
        if results_file.exists():
            rfile += 1

    total_points = model_module.get_total_models()
    models_iter = model_module.get_models_iter(start=point)
    models_ctor = model_module.get_models_ctor(target_config, model_args)

    if fix:
        if len(existing) < target_steps:
            print('Will try fixing {} missing measurements between steps 0 and {}'.format(target_steps - len(existing), target_steps))
        else:
            print('All points found, in the range from 0 to {}'.format(target_steps))
            return
    else:
        print('Starting from point:', point)
        print('  Step:', '{}/{}'.format(step, total_points))

    data = []
    results_file = results_folder / 'results{}.pickle'.format(rfile)
    if results_file.exists():
        data = pickle.loads(results_file.read_bytes())

    needs_open = True
    open_mode = 'ab'
    step_file = open(step_file, 'w')
    if not fix:
        step_file.write(str((step, point, rfile)))
    else:
        step_file.write(str((target_steps, target_point, rfile)))

    beginning = time.time()
    last_finished = True
    total_run = 0
    try:
        with device_module.get_runner(metric, target_config, device_args) as runner:
            for point in models_iter:
                if fix:
                    if tuple(point) in existing:
                        continue
                    print('Running a missing point: {} ...'.format(point))
                else:
                    print('{}/{}: {} ... '.format(step, total_points, point), end='', flush=True)

                last_finished = False

                start = time.time()
                model = models_ctor(point)
                if model is None:
                    data.append((point.copy(), 'invalid'))
                    print('invalid')
                else:
                    run_result = runner.run(model)
                    data.append((point.copy(), run_result))
                    if isinstance(run_result, str):
                        print(run_result)
                    else:
                        print(f'ok {time.time()-start}s')

                last_finished = True
                total_run += 1

                if needs_open:
                    results_file = open(results_folder / 'results{}.pickle'.format(rfile), open_mode)
                    needs_open = False

                results_file.seek(0)
                results_file.truncate(0)
                results_file.write(pickle.dumps(data))

                if len(data) >= 1000:
                    results_file.flush()
                    results_file.close()
                    rfile += 1
                    needs_open = True
                    open_mode = 'wb'
                    data.clear()
                    if fix:
                        step_file.seek(0)
                        step_file.truncate(0)
                        step_file.write(str((target_steps, target_point, rfile)))

                step += 1
                if not fix:
                    step_file.seek(0)
                    step_file.truncate(0)
                    step_file.write(str((step, point, rfile)))
                elif step >= target_steps:
                    break
    finally:
        if not needs_open:
            results_file.flush()
            results_file.close()
        step_file.flush()
        step_file.close()

        if not last_finished:
            print("Not finished")
        print(f'Took {time.time()-beginning} seconds to run {total_run} models')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model family to run, should a name of one of the packages under eagle.models')
    parser.add_argument('--device', type=str, required=True, help='Device on which the models will be run, should be a name of one of the packages under eagle.device_runner')
    parser.add_argument('--metric', type=str, default='latency', help='Metric to measure. Default: latency.')
    parser.add_argument('--cfg', type=str, default=None, help='Configuration file for device and model packages')
    parser.add_argument('--expdir', type=str, default='results', help='Folder in which the results of measurements will be saved. Default: results')
    parser.add_argument('--fix', '-f', action='store_true', help='Reads current measurements and checks for missing entries - those will be run and saved to a new file')
    args = parser.parse_args()

    extra_args = {}
    if args.cfg:
        import yaml
        with open(args.cfg, 'r') as f:
            extra_args = yaml.load(f, Loader=yaml.Loader)

        print('Model args:')
        print('============')
        print(yaml.dump(extra_args.get('model')))
        print('Device args:')
        print('============')
        print(yaml.dump(extra_args.get('device')))

    main(args.expdir, args.metric, args.model, args.device, fix=args.fix, model_args=extra_args.get('model'), device_args=extra_args.get('device'))
