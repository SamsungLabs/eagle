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

import pickle
import pathlib
import argparse
import importlib

import torch
import numpy as np

from .. import utils as eagle_utils # to distinguish from predictors_utils
from . import utils


def get_predictor(predictor_name, predictor_args=None, checkpoint=None, ignore_last=False, augment=0):
    predictor_args = predictor_args or {}
    if augment:
        predictor_args['augments'] = augment
    predictor_module = importlib.import_module('.' + predictor_name, 'eagle.predictors')
    predictor = predictor_module.get_predictor(predictor_args)
    if checkpoint:
        print(f"Loading weights from {checkpoint!r}")
        state_dict = torch.load(checkpoint)
        if ignore_last:
            print('   Without last layer...')
            for p in predictor.final_params():
                found = False
                for name, p2 in predictor.named_parameters():
                    if p2 is p:
                        del state_dict[name]
                        found = True
                        break
                if not found:
                    raise KeyError('Cannot find a parameter returned by "predictor.final_params()" in "predictor.named_parameters()"')
            predictor.load_state_dict(state_dict, strict=False)
        else:
            predictor.load_state_dict(torch.load(checkpoint))

    return predictor


def prepare_tensors(gs, targets, model_module, binary_classifier, normalize, augments=None, device=None):
    adjacency_batch, features_batch = [], []
    if augments:
        augments_batch = []
    else:
        augments_batch = None

    for g in gs:
        if binary_classifier:
            pair = g
            adjacency_pair, features_pair = [], []
            if augments:
                aug_pair = []
            for g in pair:
                if model_module is not None:
                    matrix, labels = model_module.get_matrix_and_ops(g, prune=True, keep_dims=True)
                    adjacency, features = model_module.get_adjacency_and_features(matrix, labels)
                else:
                    (adjacency, features), g = g

                adjacency_pair.append(adjacency)
                features_pair.append(features)
                if augments:
                    augs = [adict[g] for adict in augments]
                    aug_pair.append(augs)

            adjacency_batch.append(adjacency_pair)
            features_batch.append(features_pair)
            if augments:
                augments_batch.append(aug_pair)
        else:
            if model_module is not None:
                matrix, labels = model_module.get_matrix_and_ops(g, prune=True, keep_dims=True)
                adjacency, features = model_module.get_adjacency_and_features(matrix, labels)
            else:
                (adjacency, features), g = g

            adjacency_batch.append(adjacency)
            features_batch.append(features)
            if augments:
                augs = [adict[g] for adict in augments]
                augments_batch.append(augs)

    if targets is not None:
        if binary_classifier:
            if binary_classifier == 'oneway':
                targets = [[(pair[0] - pair[1] + 1) / 2] for pair in targets]
            elif binary_classifier == 'oneway-hard':
                targets = [[1 if pair[0] > pair[1] else 0] for pair in targets]
            else:
                if normalize:
                    max_target = max([max(pair) for pair in targets])
                    min_target = min([min(pair) for pair in targets])
                    targets = [[(t - min_target) / (max_target - min_target) for t in pair] for pair in targets]

                #targets = [[t - min(pair) for t in pair] for pair in targets]
                targets = [np.exp(pair) / sum(np.exp(pair)).tolist() for pair in targets]
        else:
            targets = [[t] for t in targets]

    def move(t):
        if device:
            return t.to(device=device)
        elif torch.cuda.is_available():
            return t.cuda()
        return t

    adjacency_batch = move(torch.DoubleTensor(adjacency_batch))
    features_batch = move(torch.DoubleTensor(features_batch))
    if targets is not None:
        targets = move(torch.DoubleTensor(targets))
    if augments:
        augments_batch = move(torch.DoubleTensor(augments_batch))

    return adjacency_batch, features_batch, targets, augments_batch


def simple_forward(model_module, predictor, point, augments=None):
    ''' Run batch 1 inference for a specified point and extract result
    '''
    adjacency, features, _, augments_t = prepare_tensors([point], None, model_module, predictor.binary_classifier, False, augments=augments)
    predictor.eval()
    with torch.no_grad():
        if augments is not None:
            ret = predictor(adjacency, features, augments_t)[0].cpu().item()
        else:
            ret = predictor(adjacency, features)[0].cpu().item()
    predictor.train()
    return ret


def _batched_helper(model_module, predictor, models, batch, augments, device, forward_fn):
    from tqdm import tqdm
    results = []
    steps = len(models) // batch
    if len(models) % batch != 0:
        steps += 1

    beg, end = 0, min(batch, len(models))
    for _ in tqdm(range(steps)):
        batch_of_models = models[beg:end]
        adjacency, features, _, augments_t = prepare_tensors(batch_of_models, None, model_module, False, False, augments=augments, device=device)

        if augments is not None:
            ret = forward_fn(adjacency, features, augments_t)
        else:
            ret = forward_fn(adjacency, features)

        ret = [t.detach().cpu() for t in ret]
        results.extend(ret)

        beg = end
        end = min(end + batch, len(models))

    assert beg == len(models), f'{beg}, {end}, {len(models)}, {steps}, {batch}, {len(results)}'
    return results



def batched_forward(model_module, predictor, models, batch, augments=None, device=None):
    return _batched_helper(model_module, predictor, models, batch, augments, device, predictor.__call__)


def precompute_embeddings(model_module, predictor, models, batch, augments=None, device=None):
    return _batched_helper(model_module, predictor, models, batch, augments, device, predictor.extract_features)


def precomputed_forward(predictor, input_idx, features):
    if predictor.binary_classifier:
        input1, input2 = input_idx
        inputs = [features[input1][None], features[input2][None]]
    else:
        inputs = [features[input_idx][None]]

    return predictor.regress(*inputs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Family of models')
    parser.add_argument('--device', required=True, help='Device name')
    parser.add_argument('--predictor', required=True, help='Predictor type')
    parser.add_argument('--checkpoint', required=True, help='Predictor weights to use')
    parser.add_argument('--metric', required=True, help='Metric name')
    parser.add_argument('--expdir', type=str, default='results', help='Folder in which the results of measurements will be saved. Default: results')
    parser.add_argument('--predictor_min', type=float, default=None, help='Min constant if predictor is generating normalized values')
    parser.add_argument('--predictor_max', type=float, default=None, help='Max constant if predictor is generating normalized values')
    parser.add_argument('--cfg', type=str, default=None, help='Configuration file for the predictor package')
    args = parser.parse_args()

    if args.predictor_min is not None or args.predictor_max is not None:
        if args.predictor_min is None or args.predictor_max is None:
            raise ValueError('--predictor_min requires --predictor_max and vice-verse')

    extra_args = {}
    if args.cfg:
        import yaml
        with open(args.cfg, 'r') as f:
            extra_args = yaml.load(f, Loader=yaml.Loader)

    results = {}
    models_module = importlib.import_module('.' + args.model, 'eagle.models')
    predictor = get_predictor(args.predictor, extra_args.get('predictor', {}), args.checkpoint, ignore_last=False)
    if torch.cuda.is_available():
        predictor.cuda()

    models_iter = models_module.get_models_iter()
    for model in models_iter:
        result = simple_forward(models_module, predictor, model)
        if args.predictor_min is not None:
            result = result * (args.predictor_max - args.predictor_min) + args.predictor_min
        results[eagle_utils.freeze(model)] = result
        print(model, result)

    if args.checkpoint:
        checkpoint_name = pathlib.Path(args.checkpoint)
        model_name = checkpoint_name.with_suffix('').name
    else:
        model_name = ""

    output = pathlib.Path(args.expdir) / args.model / args.metric / args.device / args.predictor
    output.mkdir(parents=True, exist_ok=True)
    if model_name:
        output = output / f'inference_{model_name}.pickle'
    else:
        output = output / f'inference.pickle'
    with open(output, 'wb') as out:
        pickle.dump(results, out)
