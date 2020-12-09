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

from ...models import model_info
from . import runner

def get_model_requirements():
    return model_info.ModelRequirements(
        frameworks=['tf'],
        data_types_and_formats=[
            ('float32', 'float32', None)
        ],
        max_dims=4
    )

# Expose function which are expected to tbe at the package level for external use,
# but it's more convenient to keep them in the runner module internally
get_supported_metrics = runner.get_supported_metrics
get_runner = runner.get_runner
