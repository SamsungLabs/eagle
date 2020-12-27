#!/usr/bin/env python
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

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

import importlib.util
from pathlib import Path

package_name = 'eagle'

version_file = Path(__file__).parent.joinpath(package_name, 'version.py')
spec = importlib.util.spec_from_file_location('{}.version'.format(package_name), version_file)
package_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_version)

class build_maybe_inplace(build_py):
    def run(self):
        _dist_file = version_file.parent.joinpath('_dist_info.py')
        assert not _dist_file.exists()
        _dist_file.write_text('\n'.join(map(lambda attr_name: attr_name+' = '+repr(getattr(package_version, attr_name)), package_version.__all__)) + '\n')
        return super().run()


setup(name='Eagle',
      version=package_version.version,
      description='Efficient and AGiLE performance estimator',
      author='SAIC-Cambridge, On-Device Team, ML-Tooling Group',
      author_email='on.device@samsung.com',
      url='https://github.com/thomasccp/eagle',
      download_url='https://github.com/thomasccp/eagle',
      python_requires='>=3.6.0',
      install_requires=[],
      packages=find_packages(where='.', exclude=['tests', 'tools']),
      package_dir={ '': '.' },
      cmdclass={
          'build_py': build_maybe_inplace
      }
)
