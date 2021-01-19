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
import operator
import importlib
from itertools import chain
from collections import defaultdict, OrderedDict

import numpy as np

from . import utils
from .. import utils as base_utils


class EagleDataset():
    def __init__(self,
                file,
                process=False,
                multiple_files=False,
                model=None,
                training_points=None,
                validation_points=None,
                sampling_method=None,
                sampling_seed=None,
                total_points=None,
                lat_limit=None,
                lat_predictor=None,
                model_module=None):
        ''' Arguments:
                file : name of the folder containing raw measurements (multiple_files==True) or a single file with already preprocessed measurements (multiple_files==False)
                process : wether points read from a file/folder should be preprocessed (take a mean of the 25-75 percentile range) or not
                multiple_files : wether the dataset should be read directly from the file specified (argument set to False) or from a collection of files (True)
                training_points : how many points to use when training, if set to None defaults to 10% of all points in the dataset,
                    this can be either an integer, in which case it should specify the exact number of points to use (raise error if not enough points are in the
                    dataset), or can be a floating point number less than 1 in which case it will be interpreted as a fraction of the entire dataset
                validation_points : how many points to use for validation, the same semantics apply as for ``training_points``
                sampling_method : how to sample points from the dataset, for example, ``training_points`` many points are sampled from the dataset to form a training set
                    and this argument specifies how these points are being selected from the full dataset. Possible values include:

                        - ``"random"`` random selection,
                        - ``"bucket"`` bucket selection

                    defaults to ``"random"``.
                sampling_seed (int) : a seed value to use when sampling, might not be meaningful for all methods, if ``None`` the seed won't be fixed.
                total_points (int) : a total number of points to read from the dataset, if ``None`` all data will be used, otherwise only ``total_points`` first points
                    will be used

            Raises:
                ``ValueError`` if either:

                    - ``total_points`` is set but the specified file/directory does not contain that many points
                    - ``training_points + validation_points > len(dataset)``
                    - unknown ``sampling_method`` is used

        '''
        self.file = file
        self.process = process
        self.multiple_files = multiple_files
        self.model = model
        self.dataset = self._load_data(total_points)
        self.full_dataset = self.dataset

        if lat_limit:
            from . import infer
            print(f'Prefiltering models to include only those with latency < {lat_limit}')
            old_len = len(self.dataset)
            self.dataset = [(pt,value) for pt, value in self.dataset if infer.simple_forward(model_module, lat_predictor, pt) <= lat_limit]
            print(f'{len(self.dataset)} points out of {old_len} matched the criteria')
            self.valid_pts = set([p[0] for p in self.dataset])
        else:
            self.valid_pts = None

        self.total_points = total_points if total_points is not None else len(self.dataset)
        self.training_points = training_points if training_points is not None else int(len(self.dataset) * 0.1)
        self.validation_points = validation_points if validation_points is not None else int(len(self.dataset) * 0.1)
        if self.training_points + self.validation_points > self.total_points:
            raise ValueError('Bad train/validation split: not enough points in the dataset')
        self.sampling_method = sampling_method if sampling_method is not None else 'random'
        self.sampling_seed = sampling_seed

        self.train_set, self.valid_set, self.test_set = self._prepare_data()

    def _load_data(self, points_limit=None):
        dataset = []

        point_cnt = 0

        def _process(times):
            avgtimes = []
            for subtimes in times:
                q25 = np.percentile(subtimes, 25)
                q75 = np.percentile(subtimes, 75)
                subtimes_filtered = list(filter(lambda x : (x >= q25) and (x <= q75), subtimes))
                avgtimes.append(np.mean(subtimes_filtered))
            q25 = np.percentile(avgtimes, 25)
            q75 = np.percentile(avgtimes, 75)
            avgtimes_filtered = list(filter(lambda x : (x >= q25) and (x <= q75), avgtimes))
            return np.mean(avgtimes_filtered)

        if self.multiple_files:
            total_files = len([name for name in os.listdir('./'+self.file) if 'results' in name])
            file_cnt = 0
            while(file_cnt < total_files):
                with open('./' + self.file + '/results' + str(file_cnt) + '.pickle', 'rb') as f:
                    data = pickle.load(f)

                for g, times in data:
                    processed = times
                    if self.process:
                        if 'invalid' not in times and not isinstance(times, str):
                            processed = _process(times)
                        else:
                            continue

                    dataset.append((g, processed))
                    point_cnt += 1
                    if points_limit is not None and point_cnt == points_limit:
                        return dataset

                file_cnt += 1
        else:
            with open(self.file, 'rb') as f:
                data = pickle.load(f)

            for g, times in data.items():
                processed = times
                if self.process:
                    processed = _process(times)

                dataset.append((g, processed))
                point_cnt += 1
                if points_limit is not None and point_cnt == points_limit:
                    return dataset

        if points_limit is not None and point_cnt < points_limit:
            raise ValueError('Not enough points in the dataset: expected at least {} but got {}'.format(points_limit, point_cnt))

        return dataset

    def _prepare_data(self):
        assert len(self.dataset) == self.total_points
        indices = list(range(self.total_points))
        if self.sampling_seed is not None:
            np.random.seed(self.sampling_seed)

        if self.sampling_method == 'bucket_equal':
            training_indices, remaining_indices = EagleDataset.bucket_sampling(self.dataset, self.model, indices, self.training_points, 'equal')
            validation_indices = remaining_indices[:self.validation_points]
            testing_indices = remaining_indices[self.validation_points:]
        elif self.sampling_method == 'bucket_occ':
            training_indices, remaining_indices = EagleDataset.bucket_sampling(self.dataset, self.model, indices, self.training_points, 'occ')
            validation_indices = remaining_indices[:self.validation_points]
            testing_indices = remaining_indices[self.validation_points:]
        elif self.sampling_method == 'random':
            np.random.shuffle(indices)
            training_indices = indices[:self.training_points]
            validation_indices = indices[self.training_points:self.training_points+self.validation_points]
            testing_indices = indices[self.training_points+self.validation_points:]
        else:
            raise ValueError(f'Unknown sampling method {self.sampling_method!r}')

        training_data = [self.dataset[i] for i in training_indices]
        validation_data = [self.dataset[i] for i in validation_indices]
        testing_data = [self.dataset[i] for i in testing_indices]

        return training_data, validation_data, testing_data

    @staticmethod
    def fill_the_buckets(ss_dict, measure, bucket_criteria, p_in_bucket,
                         dataset, realize):
        '''
            measure: \"params/edges/flops/{device}\"

            bucket_criteria:[] the criteria for each bucket in a list.
            For instance, [c1, c2, c3, c4, ...], bucket 1 contains measure <= c1 and > c2.

            p_in_bucket: {} number of points required in each bucket.

            returns a dict where key is the bucket_id
                        and value is the list of points in the bucket
        '''
        s = utils.SearchSpaceIterator(ss_dict[dataset]['search_space'], shuffle=True)

        # handle zero case
        filled_bucket = defaultdict(list)
        to_del = []
        if p_in_bucket:
            for i in p_in_bucket.keys():
                if p_in_bucket[i] == 0:
                    to_del.append(i)
                    filled_bucket[i] = []
            for i in to_del:
                del p_in_bucket[i]

        total_points = 0
        for p in s:
            if measure == 'edges':
                m = utils.count_ops_edges(realize(p))['edges']
            elif tuple(p) in ss_dict[dataset][measure]:
                m = ss_dict[dataset][measure][tuple(p)]
            else:
                raise NotImplementedError()

            for i, (v, v2) in enumerate(zip(bucket_criteria, bucket_criteria[1:])):
                #v2 = bucket_criteria[i + 1]

                cf = operator.lt
                if i + 1 == len(bucket_criteria)-1:
                    # edge case: include max in search
                    cf = operator.le

                if m >= v and cf(m, v2):
                    if p_in_bucket:
                        if i in p_in_bucket:
                            p_in_bucket[i] -= 1
                            if p_in_bucket[i] == 0:
                                del p_in_bucket[i]
                            filled_bucket[i].append(p)

                            total_points += 1
                            if not p_in_bucket.keys():
                                print(f'Bucket filled! Returning dict of bucket with {total_points} points!')
                                return filled_bucket
                    else:
                        filled_bucket[i].append(p)
                        total_points += 1
        if p_in_bucket:
            for k, v in p_in_bucket.items():
                print(f'Cant find sufficient points! bucket {k} has {v} points left to fill')
                print(f'Returning dict of bucket with {total_points} points instead..')

        return filled_bucket

    @staticmethod
    def _bucket_sampling(ss_dict, N, measure, min_point, max_point, mode, buckets, realize, dataset, no_var_points=100):
        '''
         N: total number of training example
         model: 'equal'/'occ' for equal, no. of occurrence sampling approaches
         buckets: total number of buckets
         measure: \"params/edges/flops/{device}\"
         min_point & max_point: to be evaluated for min and max based on measure
         no_var_points:int number of points each bucket to calculate the variance of each bucket

         Returns:
             sampled points, min_measure of the min_point, max_measure of the max_point
        '''

        #### 1. Get min and max of measure
        if measure == 'edges':
            min_m = utils.count_ops_edges(realize(min_point))['edges']
            max_m = utils.count_ops_edges(realize(max_point))['edges']
        else:
            min_m = ss_dict[dataset][measure][tuple(min_point)]
            max_m = ss_dict[dataset][measure][tuple(max_point)]
        print(f'Min measure: {min_m}, Max measure: {max_m}')

        #### 2. Determine the bucket criterias
        bucket_criteria = list(range(min_m, max_m, max_m // buckets))

        if len(bucket_criteria) == buckets + 1:
            # not divisible by no of buckets
            bucket_criteria[-1] = max_m
        else:
            bucket_criteria.append(max_m)

        if measure == 'edges':
            # Literally Edge Case: we want to include the last edge in the criteria as a lower bound
            # Specifically, we want to include edges == max_m
            bucket_criteria.append(max_m + (max_m//buckets))

        print(f'Bucket Criterias: {bucket_criteria}')
        assert len(bucket_criteria) == buckets+1 # n+1 criterias for n buckets

        #### 3. Deciding no. of samples per bucket
        assert mode in ['equal', 'occ']
        p_in_bucket = OrderedDict()

        if mode == 'equal':
            for i in range(buckets):
                p_in_bucket[i] = N // buckets

            # we will just put leftover points in random buckets
            for i in np.random.choice(np.arange(buckets), size=N%buckets, replace=False):
                p_in_bucket[i] += 1

            b = EagleDataset.fill_the_buckets(ss_dict, measure, bucket_criteria, p_in_bucket, dataset=dataset, realize=realize)

            return list(chain.from_iterable([v for v in b.values()])), min_m, max_m

        elif mode == 'occ':
            b = EagleDataset.fill_the_buckets(ss_dict, measure, bucket_criteria, None, dataset=dataset, realize=realize)
            bucket_count = []
            # extract measurements based on number of points in each bucket
            for i in range(buckets):
                bucket_count.append(len(b[i]))
            print(f'Bucket count: {bucket_count}')

            # scale and round down
            bucket_count = [int(v / sum(bucket_count) * N) for v in bucket_count]
            print(f'After scaling to N: {bucket_count}')

            for i in range(buckets):
                p_in_bucket[i] = bucket_count[i]

            # leftovers from rounding down would be randomly allocated
            # this means that empty buckets can be selected too!
            leftovers = N - sum(bucket_count)
            for i in np.random.choice(np.arange(buckets), size=leftovers, replace=False):
                p_in_bucket[i] += 1

            b = EagleDataset.fill_the_buckets(ss_dict, measure, bucket_criteria, p_in_bucket, dataset=dataset, realize=realize)

            return list(chain.from_iterable([v for v in b.values()])), min_m, max_m

    @staticmethod
    def bucket_sampling(data, model_name, indices, training_points, mode):

        g = [list(x[0]) for x in data]

        model_module = importlib.import_module('.' + model_name, 'eagle.models')

        ss_dict = {}
        ss_dict[model_name] = {}
        ss_dict[model_name]['search_space'] = [5]*6
        s = utils.SearchSpaceIterator(ss_dict[model_name]['search_space'], shuffle=True)

        measure = 'edges'
        # Todo: Some settings are still hardcoded for nasbench201
        points, min_m, max_m = EagleDataset._bucket_sampling(ss_dict, training_points, measure, [0,0,0,1,0,0], [3,3,3,3,3,3], mode=mode, buckets=10, realize=model_module.get_matrix_and_ops, dataset=model_name)

        #training_indices = [s.point_to_int(p) for p in points if p in g]
        training_indices = [g.index(p) for p in points if p in g]

        np.random.shuffle(training_indices)
        remaining_indices = list(set(indices) - set(training_indices))
        np.random.shuffle(remaining_indices)

        return training_indices, remaining_indices


def select_random(data, k, current=None):
    indices = list(range(len(data)))
    ret = []
    while len(ret) < k:
        np.random.shuffle(indices)
        to_select = k - len(ret)
        selected = [data[i] for i in indices[:to_select] if data[i] not in ret and (current is None or data[i] not in current)]
        ret.extend(selected)
    return ret


class DartsDataset():
    def __init__(self, known_points, sample_size, sampling_seed=None, dataset_file=None):
        import eagle.models.darts as darts
        globals()['Genotype'] = darts.Genotype

        if known_points:
            with open(known_points, 'rb') as f:
                self.known_points = pickle.load(f)
                self.known_points = [(eval(p[0]), *p[1:]) for p in self.known_points]
        else:
            self.known_points = []

        self.train_set = self.known_points
        self.valid_set = self.train_set
        self.valid_pts = None

        self.known_points = set(base_utils.freeze(p[0]) for p in self.train_set)
        print(f'Training set contains: {len(self.train_set)} points')

        # full_dataset is used to do final scoring - we use that moment to
        # search for new points, in order to do that we will sample a random
        # subset of points from the darts search space (since sorting all of them
        # would be too expensive) - then, by looking at the results, we can decide
        # which should be the new models to train out of those we selected here
        # In summary: we will do something like stochastic predictor-based search
        self.full_dataset = self.sample_random(sample_size, sampling_seed, dataset_file)

    def sample_random(self, sample_size, seed, dataset_file):
        dataset_file = pathlib.Path(dataset_file)
        if dataset_file.exists():
            print(f'Loading pre-computed dataset from: {dataset_file}')
            with dataset_file.open('rb') as f:
                saved_seed, sample = pickle.load(f)
                if saved_seed != seed:
                    raise ValueError('Different seed!')

                full_size = len(sample)
                if full_size != sample_size:
                    raise ValueError('Different size!')
                sample = [p for p in sample if p not in self.known_points]
                unknown_size = len(sample)
                print(f'    Loaded {full_size} points from which {unknown_size} are not trained yet')
                return sample

        print(f'Sampling {sample_size} random points from DARTS search space (using seed: {seed})...')
        import random
        if seed is not None:
            random.seed(seed)

        sample = []
        import tqdm
        pbar = tqdm.tqdm(total=sample_size)
        import eagle.models.darts as darts
        # inefficient but simple and does its job...
        while len(sample) < sample_size:
            model = darts.get_random_model()
            if model not in self.known_points:
                sample.append((model, -1))
                self.known_points.add(model)
                pbar.update(1)

        pbar.close()
        print('    Done')
        if dataset_file:
            print(f'    Saving the dataset to: {dataset_file}')
            with dataset_file.open('wb') as f:
                pickle.dump((seed, sample), f)
        return sample

