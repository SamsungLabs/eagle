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
import statistics
import collections

import numpy as np
import eagle.utils as utils


def recursive_iter(data):
    if isinstance(data, list):
        for e in data:
            for sube in recursive_iter(e):
                yield sube
    else:
        yield data


def calc_stats(values):
    averages = []
    for subvalues in values:
        q25 = np.percentile(subvalues, 25)
        q75 = np.percentile(subvalues, 75)
        subvalues_filtered = list(filter(lambda x : (x >= q25) and (x <= q75), subvalues))
        averages.append(np.mean(subvalues_filtered))
    q25 = np.percentile(averages, 25)
    q75 = np.percentile(averages, 75)
    averages_filtered = list(filter(lambda x : (x >= q25) and (x <= q75), averages))
    return np.mean(averages_filtered)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', help='File with results -- you can use "combine.py" script to combine multiple files into one')
    parser.add_argument('--out', '-o', required=True, help='Output file')
    args = parser.parse_args()

    data = pickle.loads(pathlib.Path(args.results).read_bytes())
    all_results = []
    results = {}
    invalid = 0
    failed = 0
    for point, value in data:
        if value == 'invalid':
            invalid += 1
        elif not isinstance(value, list):
            failed += 1
        else:
            all_results.append(value)
            point = utils.freeze(point)
            if point in results:
                raise KeyError('Duplicated entry: {}'.format(point))
            results[point] = calc_stats(value)

    print('All results:', len(data))
    print('  Invalid:  ', invalid)
    print('   Failed:  ', failed)
    print('    Valid:  ', len(results))

    if not all_results:
        print('No valid results found')
    else:
        to_save = results
        pathlib.Path(args.out).write_bytes(pickle.dumps(to_save))
