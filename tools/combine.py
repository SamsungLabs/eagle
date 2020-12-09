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

import re
import sys
import pickle
import pathlib
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_folder', help='Folder with measurements files')
    parser.add_argument('--pattern', '-p', default='results*.pickle', help='Results pattern')
    parser.add_argument('--out', '-o', required=True, help='Name of the output file to store combined results')
    args = parser.parse_args()

    data = []
    f = pathlib.Path(args.results_folder)
    for f in f.glob(args.pattern):
        print('Processing file:', f)
        data.extend(pickle.loads(f.read_bytes()))

    if data:
        print('Got {} points in total, saving them to {!r}'.format(len(data), args.out))
        pathlib.Path(args.out).write_bytes(pickle.dumps(data))
    else:
        print('Nothing has been found - will not write to the output file')
