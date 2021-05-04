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

import os
import pickle
import pathlib
import argparse
import importlib
import functools
import contextlib
import statistics

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from . import utils
from . import infer
from . import dataset as dataset_mod


def _train(model_module, model, gs, latencies, optimizer, criterion, normalize=False, augments=None):
    adjacency, features, latency, aug = infer.prepare_tensors(gs, latencies, model_module, model.binary_classifier, normalize, augments=augments)

    model.train()
    optimizer.zero_grad()
    if augments is not None:
        predictions = model(adjacency, features, aug)
    else:
        predictions = model(adjacency, features)

    loss = criterion(predictions, latency)
    loss.backward()
    optimizer.step()

    return loss


def _test(model_module, model, g, latency, leeways, criterion, log_file=None, augments=None):
    if not model.binary_classifier:
        adjacency, features, latency, aug = infer.prepare_tensors([g], [latency], model_module, False, False, augments=augments)
    else:
        adjacency, features, latency, aug = infer.prepare_tensors(g, latency, model_module, model.binary_classifier, False, augments=augments)

    torch.set_grad_enabled(False)
    model.eval()
    if augments is not None:
        predictions = model(adjacency, features, aug)
    else:
        predictions = model(adjacency, features)

    if not model.binary_classifier:
        if log_file is not None:
            log_file.write(f'{latency.item()} {predictions.item()} {g}\n')

    loss = criterion(predictions, latency)
    torch.set_grad_enabled(True)

    if not model.binary_classifier:
        results = []
        for l in leeways:
            results.append(utils.valid(predictions, latency, leeway=l))

        return results, loss, (latency.item(), predictions.item())
    else:
        return None, loss, None


def train(training_set,
        validation_set,
        outdir,
        device_name,
        model_name,
        metric,
        predictor_name,
        predictor,
        tensorboard,
        epochs,
        learning_rate,
        weight_decay,
        lr_patience,
        es_patience,
        batch_size,
        shuffle,
        optim_name,
        lr_scheduler,
        exp_name=None,
        reset_last=False,
        warmup=0,
        save=True,
        augments=None):
    model_module = importlib.import_module('.' + model_name, 'eagle.models')

    outdir = pathlib.Path(outdir) / model_name / metric / device_name / predictor_name
    outdir.mkdir(parents=True, exist_ok=True)

    if tensorboard:
        import torch.utils.tensorboard as tb
        handler = tb.SummaryWriter(f'tensorboard/{exp_name}')

    if reset_last:
        predictor.reset_last()

    if optim_name == 'adamw':
        optimizer = optim.AdamW(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optim_name}')

    if lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience, threshold=0.01, verbose=True)
    elif lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    else:
        raise ValueError(f'Unknown lr scheduler: {lr_scheduler}')

    if not predictor.binary_classifier:
        criterion = torch.nn.L1Loss(reduction='sum')
    else:
        if predictor.binary_classifier == 'oneway' or predictor.binary_classifier == 'oneway-hard':
            criterion = torch.nn.BCELoss(reduction='sum')
        else:
            criterion = torch.nn.KLDivLoss(reduction='sum')

    es = utils.EarlyStopping(mode='min', patience=es_patience)

    if predictor.binary_classifier:
        training_set = utils.ProductList(training_set)
        def collate_fn(batch):
            return [[e[0] for e in pair] for pair in batch], [[e[1] for e in pair] for pair in batch]
    else:
        def collate_fn(batch):
            return [e[0] for e in batch], [e[1] for e in batch]

    training_data = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    data = validation_set

    train_corrects = [0, 0, 0, 0]
    test_corrects = [0, 0, 0, 0]
    best_accuracies = [0, 0, 0, 0]
    best_epochs = [0, 0, 0, 0]
    leeways = [0.01, 0.05, 0.1, 0.2] # +-% Accuracies
    lowest_loss = None

    if warmup:
        print(f'Warming up the last layer for {warmup} epochs')
        warmup_opt = optim.AdamW(predictor.final_params(), lr=learning_rate, weight_decay=0)
        for warmup_epoch in range(warmup):
            print(f"Warmup Epoch: {warmup_epoch}")
            for g, latency in training_data:
                loss = _train(model_module, predictor, g, latency, warmup_opt, criterion, augments=augments)

    for epoch_no in range(epochs):
        print(f"Epoch: {epoch_no}")

        for g, latency in training_data:
            loss = _train(model_module, predictor, g, latency, optimizer, criterion, augments=augments)

        train_loss = 0.

        if not predictor.binary_classifier:
            for g, latency in training_set:
                corrects, loss, _ = _test(model_module, predictor, g, latency, leeways, criterion, augments=augments)
                for i, c in enumerate(corrects):
                    train_corrects[i] += c
                train_loss += loss
        else:
            for g, latency in training_data:
                _, loss, _ = _test(model_module, predictor, g, latency, None, criterion, augments=augments)
                train_loss += loss

        avg_loss = train_loss / len(training_set)

        if not predictor.binary_classifier:
            train_accuracies = [train_correct / len(training_set) for train_correct in train_corrects]
            print(f'Top +-{leeways} Accuracy of train set for epoch {epoch_no}: {train_accuracies} ')
        print(f'Average loss of training set {epoch_no}: {avg_loss}')

        if not predictor.binary_classifier:
            val_loss = 0.
            for g, latency in validation_set:
                corrects, loss, _ = _test(model_module, predictor, g, latency, leeways, criterion, augments=augments)
                for i, c in enumerate(corrects):
                    test_corrects[i] += c
                val_loss += loss
            avg_loss = val_loss / len(validation_set)

            current_accuracies = [test_correct / len(validation_set) for test_correct in test_corrects]
            print(f'Average loss of validation set {epoch_no}: {avg_loss}')

            for i, best_accuracy in enumerate(best_accuracies):
                if current_accuracies[i] >= best_accuracy:
                    best_accuracies[i] = current_accuracies[i]
                    best_epochs[i] = epoch_no
        else:
            val_loss = train_loss

        if torch.cuda.is_available():
            val_loss = val_loss.cpu()

        if lowest_loss is None or val_loss < lowest_loss:
            lowest_loss = val_loss
            best_predictor_weight = predictor.state_dict()
            if save:
                torch.save(best_predictor_weight, outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt'))
            print(f'Lowest val_loss: {val_loss}... Predictor model saved.')

        if not predictor.binary_classifier:
            print(f'Top +-{leeways} Accuracy of validation set for epoch {epoch_no}: {current_accuracies}')
            print(f'[best: {best_accuracies} @ epoch {best_epochs}]')

        if lr_scheduler == 'plateau':
            if epoch_no > 20:
                scheduler.step(val_loss)
        else:
            scheduler.step()

        if epoch_no > 20:
            if es.step(val_loss):
                print('Early stopping criterion is met, stop training now.')
                break

        train_corrects = [0, 0, 0, 0]
        test_corrects = [0, 0, 0, 0]

        if tensorboard:
            handler.add_scalar('loss/training', avg_train_loss, epoch_no)
            handler.add_scalar('loss/validation', avg_val_loss, epoch_no)
            handler.add_scalar('accuracy_1/training', train_accuracies[0], epoch_no)
            handler.add_scalar('accuracy_1/validation', current_accuracies[0], epoch_no)
            handler.add_scalar('accuracy_5/training', train_accuracies[1], epoch_no)
            handler.add_scalar('accuracy_5/validation', current_accuracies[1], epoch_no)
            handler.add_scalar('accuracy_10/training', train_accuracies[2], epoch_no)
            handler.add_scalar('accuracy_10/validation', current_accuracies[2], epoch_no)
            handler.add_scalar('accuracy_20/training', train_accuracies[3], epoch_no)
            handler.add_scalar('accuracy_20/validation', current_accuracies[3], epoch_no)

    if tensorboard:
        handler.close()
    if save:
        torch.save(best_predictor_weight, outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt'))

    print("Training finished!")
    predictor.load_state_dict(best_predictor_weight)
    return predictor


def predict(testing_data,
        outdir,
        device_name,
        model_name,
        metric,
        predictor_name,
        predictor,
        log=False,
        exp_name=None,
        load=False,
        iteration=None,
        explored_models=None,
        valid_pts=None,
        use_fast=True,
        augments=None):
    model_module = importlib.import_module('.' + model_name, 'eagle.models')

    if load or log:
        outdir = pathlib.Path(outdir) / model_name / metric / device_name / predictor_name
        if log:
            outdir.mkdir(parents=True, exist_ok=True)

    if load and predictor_name != 'random':
        predictor.load_state_dict(torch.load(outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt')))
        print('Predictor imported.')

    criterion = torch.nn.L1Loss()

    test_corrects = [0, 0, 0, 0]
    leeways = [0.01, 0.05, 0.1, 0.2]

    log_file = None
    if log:
        log_filename = 'log.txt' if exp_name is None else f'log_{exp_name}.txt'
        if iteration is not None:
            log_filename = f'iter{iteration}_' + log_filename

        log_file = outdir / log_filename
        sep = False
        if log_file.exists():
            sep = True
        log_file = log_file.open('a')
        if sep:
            log_file.write('===\n')

    if predictor_name == 'random':
        print('Producing random ordering of the dataset...')
        predicted = []
        perm = np.random.permutation(len(testing_data))
        for idx, (point, gt_value) in enumerate(testing_data):
            predicted_value = perm[idx]
            predicted.append(predicted_value)
            if log:
                log_file.write(f'{gt_value} {predicted_value} {point}\n')

    elif not predictor.binary_classifier:
        predicted = []
        test_loss = 0
        for g, latency in testing_data:
            corrects, loss, values = _test(model_module, predictor, g, latency, leeways, criterion, log_file, augments)
            for i, c in enumerate(corrects):
                test_corrects[i] += c

            test_loss += loss
            predicted.append(values[1])

        current_accuracies = [test_correct / len(testing_data) for test_correct in test_corrects]
        avg_loss = test_loss / len(testing_data)

        print(f'Top +-{leeways} Accuracy of test set: {current_accuracies}')
        print(f'Average loss of test set: {avg_loss}')
    else:
        torch.set_grad_enabled(False)
        predictor.eval()

        if use_fast:
            print(f'Precomputing embeddings for {len(testing_data)} graphs')
            models = [r[0] for r in testing_data]
            precomputed = infer.precompute_embeddings(model_module, predictor, models, 1024, augments=augments)
            print('Done')

        total = 0
        correct = 0
        skipped = 0
        def predictor_compare(v1, v2):
            nonlocal total
            nonlocal correct
            nonlocal skipped
            total += 1
            if valid_pts is not None and v1[0] not in valid_pts:
                skipped += 1
                return -1
            if valid_pts is not None and v2[0] not in valid_pts:
                skipped += 1
                return 1
            latencies = [v1[1], v2[1]]
            if use_fast:
                result = infer.precomputed_forward(predictor, [v1[2], v2[2]], precomputed)
            else:
                gs = [v1[0], v2[0]]
                adjacency, features, _, aug = infer.prepare_tensors([gs], None, model_module, predictor.binary_classifier, False, augments=augments)
                if augments is not None:
                    result = predictor(adjacency, features, aug)
                else:
                    result = predictor(adjacency, features)
            if predictor.binary_classifier == 'oneway' or predictor.binary_classifier == 'oneway-hard':
                v1_better = result[0][0].cpu().item() - 0.5
                if latencies[0] > latencies[1]:
                    if v1_better > 0:
                        correct += 1
                elif v1_better < 0:
                    correct += 1

                return v1_better
            else:
                rv1, rv2 = result[0][0].cpu().item(), result[0][1].cpu().item()
                if latencies[0] > latencies[1]:
                    if rv1 > rv2:
                        correct += 1
                elif rv1 < rv2:
                    correct += 1

                # we want higher number to appear later (have higher "score"), so (v1 - v2) should get us the correct order
                return rv1 - rv2

        if use_fast:
            predictor.cpu()
            test_data_with_indices = [(*v, idx) for idx, v in enumerate(testing_data)]
            sorted_values = sorted(test_data_with_indices, key=functools.cmp_to_key(predictor_compare))
            sorted_values = { pt: (gt,idx) for idx,(pt,gt,_) in enumerate(sorted_values) }
            predictor.cuda()
        else:
            sorted_values = sorted(testing_data, key=functools.cmp_to_key(predictor_compare))
            sorted_values = { pt: (gt,idx) for idx,(pt,gt) in enumerate(sorted_values) }

        predicted = []
        for p, v in testing_data:
            r = sorted_values[p][1]
            predicted.append(r)
            if log:
                log_file.write(f'{v} {r} {p}\n')

        predictor.train()
        torch.set_grad_enabled(True)

    if log:
        log_file.write('---\n')
        explored_models = explored_models or []
        for p,v in explored_models:
            log_file.write(f'{p}\n')
        log_file.write('---\n')
        if predictor_name == 'random':
            pass
        elif not predictor.binary_classifier:
            log_file.write(f'{avg_loss}\n{current_accuracies}\n')
        else:
            log_file.write(f'{correct}/{total} predictions correct\n')
            log_file.write(f'{skipped}/{total} predictions skipped\n')
        log_file.close()

    return predicted


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model family to run, should be a name of one of the packages under eagle.models')
    parser.add_argument('--device', type=str, required=True, help='Device on which the models will be run, should be a name of one of the packages under eagle.device_runner')
    parser.add_argument('--metric', type=str, default='latency', help='Metric to measure. Default: latency.')
    parser.add_argument('--predictor', type=str, required=True, help='Predictor to train, should a name of one of the packages under eagle.predictors')
    parser.add_argument('--measurement', type=str, required=True, default=None, help='Measurement file for device')
    parser.add_argument('--cfg', type=str, default=None, help='Configuration file for device and model packages')
    parser.add_argument('--expdir', type=str, default='results', help='Folder in which the results of measurements will be saved. Default: results')
    parser.add_argument('--process', action='store_true', help='Process measurements - use this if the measurements are not already processed')
    parser.add_argument('--multiple_files', action='store_true', help='Combine results from multiple files - use this if the measurements are not already combined')
    parser.add_argument('--transfer', default=None, help='Perform transfer learning from a previously trained model - the argument should point to the checkpoint to load, the final layer is still randomly initialized, use --load if you just want to load a checkpoint')
    parser.add_argument('--load', default=None, help='Checkpoint to load')
    parser.add_argument('--warmup', default=0, type=int, help='Number of warmup epochs for the last layer')
    parser.add_argument('--foresight_warmup', type=str, help='Path to the dataset containing foresight metrics which will be used to warmup the predictor during iterative training')
    parser.add_argument('--foresight_simple', action='store_true', help='Do not train the predictor when doing foresight warmup, instead simply rank models with foresight scores directly')
    parser.add_argument('--foresight_augment', type=str, nargs='+', default=[], help='Path to foresight metrics, if set they will be passed to the predictor together with each model')
    parser.add_argument('--prediction_only', action='store_true', help='Run prediction with a pretrained predictor')
    parser.add_argument('--exp', default=None, help='Optional experiment name, used when saving the predictor to distinguish between different configurations')
    parser.add_argument('--uid', default=None, type=int, help='UID to distinguish between different concurrent runs')
    parser.add_argument('--log', action='store_true', help='Log prediction on test dataset together with ground truth')
    parser.add_argument('--tensorboard', action='store_true', help='Log training data for visualization in tensorboard')
    parser.add_argument('--torch_seed', type=int, default=None, help='Fixed seed to use with torch.random')
    parser.add_argument('--quiet', action='store_true', help='Suppress standard output')
    parser.add_argument('--iter', type=int, default=0, help='Number of iterations when using iterative search')
    parser.add_argument('--save', action='store_true', help='Save the best predictor')
    parser.add_argument('--eval', action='store_true', help='Eval model only, do not train (use with --load to eval pretrained model)')
    parser.add_argument('--lat_limit', type=float, default=None, help='Latency limit to prune the search space (requires --transfer to point to the latency predictor)')
    parser.add_argument('--sample_best', action='store_true')
    parser.add_argument('--sample_best2', action='store_true')
    parser.add_argument('--reset_last', action='store_true', help='Reset last layer (only applicable if checkpoint is loaded)')
    args = parser.parse_args()

    if args.uid is not None:
        if args.exp is None:
            args.exp = str(args.uid)
        else:
            args.exp += f'_{args.uid}'

    with contextlib.ExitStack() as es:
        if args.quiet:
            f = es.enter_context(open(os.devnull, 'w'))
            es.enter_context(contextlib.redirect_stdout(f))

        extra_args = {}
        if args.cfg:
            import yaml
            with open(args.cfg, 'r') as f:
                extra_args = yaml.load(f, Loader=yaml.Loader)

        if args.transfer:
            if args.load:
                raise ValueError('Both --load and --transfer are set, please use only one. Note: "--transfer X" is the same as "--load X --reset_last"')

            args.load = args.transfer
            args.reset_last = True

        if args.predictor == 'random':
            predictor = None
        else:
            predictor = infer.get_predictor(args.predictor, predictor_args=extra_args.get('predictor'), checkpoint=args.load, ignore_last=args.reset_last, augment=len(args.foresight_augment))

        # init model module if necessary
        model_module = importlib.import_module('.' + args.model, 'eagle.models')
        if hasattr(model_module, 'init_module'):
            model_module.init_module(pathlib.Path(args.expdir) / args.model, **extra_args.get('model', {}).get('init', {}))

        lat_predictor = None
        if args.lat_limit:
            if not args.transfer:
                raise ValueError('--lat_limit requires --transfer')

            lat_predictor_args = extra_args.get('predictor').copy()
            lat_predictor_args.pop('binary_classifier', None)
            lat_predictor = infer.get_predictor(args.predictor, predictor_args=lat_predictor_args, checkpoint=args.load, ignore_last=False)

        if args.predictor != 'random':
            if torch.cuda.is_available():
                predictor.cuda()
                if lat_predictor:
                    lat_predictor.cuda()
            else:
                raise RuntimeError('No GPU!')

        if args.model == 'darts':
            dataset_args = extra_args.get('dataset', {})
            dataset_file = dataset_args.pop('dataset_file', None)
            if dataset_file:
                dataset_file = pathlib.Path(args.expdir) / args.model / args.metric / args.device / dataset_file
            dataset = dataset_mod.DartsDataset(args.measurement,
                                    dataset_file=dataset_file,
                                    **extra_args.get('dataset', {}))
        else:
            dataset = dataset_mod.EagleDataset(args.measurement,
                                    args.process,
                                    args.multiple_files,
                                    **extra_args.get('dataset', {}),
                                    lat_limit=args.lat_limit,
                                    lat_predictor=lat_predictor,
                                    model_module=model_module)

            if args.foresight_warmup:
                if not args.iter:
                    raise ValueError('Foresight warmup requires iterative training!')
                if args.foresight_augment:
                    raise ValueError('Foresigh augment is incompatible with foresight warmup')

                foresight_dataset = dataset_mod.EagleDataset(args.foresight_warmup,
                    args.process,
                    args.multiple_files,
                    **extra_args.get('foresight', {}).get('dataset', {}),
                    lat_limit=args.lat_limit,
                    lat_predictor=lat_predictor,
                    model_module=model_module)

            if args.foresight_augment:
                print(f'Using {len(args.foresight_augment)} foresight metric(s) to augment graph embeddings')
                augments = []
                for aug in args.foresight_augment:
                    with open(aug, 'rb') as f:
                        d = pickle.load(f)
                        augments.append(d)
            else:
                augments = None

        explored_models = dataset.train_set
        if not args.eval:
            if args.iter:
                if args.foresight_warmup:
                    if not args.foresight_simple:
                        print(f'Warming up predictor using foresight dataset {args.foresight_warmup!r}')
                        foresight_train_args = extra_args.get('foresight', {}).get('training', {})
                        train(foresight_dataset.train_set,
                            foresight_dataset.valid_set,
                            args.expdir,
                            args.device,
                            args.model,
                            args.metric,
                            args.predictor,
                            predictor,
                            args.tensorboard,
                            **foresight_train_args,
                            exp_name=args.exp,
                            reset_last=args.reset_last,
                            warmup=args.warmup,
                            save=False)
                    else:
                        print(f'Sorting models using foresight metrics from: {args.foresight_warmup!r}')

                train_args = extra_args.get('training', {})

                target_batch = train_args.pop('batch_size')
                batch_per_iter = target_batch // args.iter
                current_batch = batch_per_iter

                target_epochs = train_args.pop('epochs')
                epochs_per_iter = target_epochs // args.iter
                current_epochs = epochs_per_iter

                points_per_iter = len(dataset.train_set) // args.iter
                candidates = list(dataset.dataset)

                if not args.foresight_warmup:
                    train_set = dataset_mod.select_random(candidates, points_per_iter)
                else:
                    train_set = []

                for i in range(args.iter):
                    print('Iteration', i)

                    if i or args.foresight_warmup:
                        # update training set
                        if i or not args.foresight_simple:
                            scores = predict(candidates, args.expdir, args.device, args.model, args.metric, args.predictor, predictor, log=False, exp_name=args.exp, load=False, augments=augments)
                        else:
                            scores = [p[1] for p in foresight_dataset.dataset]
                        if args.sample_best or args.sample_best2:
                            if not args.sample_best2:
                                median_score = statistics.median(scores)
                                candidates = [pt for pt, score in zip(candidates, scores) if score > median_score]
                            best_candidates = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
                            added = 0
                            for candidate, score in best_candidates:
                                if added == points_per_iter//2:
                                    break
                                if candidate in train_set:
                                    continue
                                train_set.append(candidate)
                                added += 1

                            if args.sample_best2:
                                random_th = best_candidates[len(scores) // (2**(i or 1))][1]
                                random_candidates = [pt for pt, score in zip(candidates, scores) if score > random_th]
                                selected_candidates = dataset_mod.select_random(random_candidates, points_per_iter//2, current=train_set)
                            else:
                                selected_candidates = dataset_mod.select_random(candidates, points_per_iter//2, current=train_set)
                            train_set.extend(selected_candidates)
                        else:
                            median_score = statistics.median(scores)
                            candidates = [pt for pt, score in zip(candidates, scores) if score > median_score]
                            sampled = dataset_mod.select_random(candidates, points_per_iter, current=train_set)
                            train_set.extend(sampled)

                    print('Number of candidate points:', len(candidates))
                    print('Number of training points:', len(train_set))
                    print('Batch size:', current_batch)
                    print('Number of epochs:', current_epochs)

                    train(train_set,
                        train_set,
                        args.expdir,
                        args.device,
                        args.model,
                        args.metric,
                        args.predictor,
                        predictor,
                        args.tensorboard,
                        **train_args,
                        batch_size=current_batch,
                        epochs=current_epochs,
                        exp_name=args.exp,
                        reset_last=args.reset_last and not i and not args.foresight_warmup,
                        warmup=args.warmup if (not i and not args.foresight_warmup) else 0,
                        save=args.save and i + 1 == args.iter,
                        augments=augments)

                    current_batch += batch_per_iter
                    current_epochs += epochs_per_iter
                    explored_models = train_set
            else:
                if not dataset.train_set:
                    raise ValueError('Training set is empty!')
                train(dataset.train_set,
                    dataset.valid_set,
                    args.expdir,
                    args.device,
                    args.model,
                    args.metric,
                    args.predictor,
                    predictor,
                    args.tensorboard,
                    **extra_args.get('training', {}),
                    exp_name=args.exp,
                    reset_last=args.reset_last,
                    warmup=args.warmup,
                    save=args.save,
                    augments=augments)

        predict(dataset.full_dataset,
            args.expdir,
            args.device,
            args.model,
            args.metric,
            args.predictor,
            predictor,
            args.log,
            exp_name=args.exp,
            load=False,
            explored_models=explored_models,
            valid_pts=dataset.valid_pts,
            augments=augments)
