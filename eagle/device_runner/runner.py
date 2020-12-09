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

import sys
import functools


class DeviceRunner():
    def __init__(self,
                metric,
                impl_func,
                separate_process=False,
                target_args=None,
                device_addr=None,
                spawn_server=True,
                device_user=None,
                device_passwd=None,
                device_wdir=None,
                timeout=5):
        ''' Creates a generic runner which will execute the the target python function on-device.

            Target device can be specified to be:
                 - local machine, using current thread to execute the function (not device_addr and not separate_process),
        '''
        self.metric = metric
        self.impl_func = impl_func
        self.target_args = target_args
        self.separate_process = separate_process

        self.device_addr = device_addr
        self.spawn_server = spawn_server
        self.device_user = device_user
        self.device_passwd = device_passwd
        self.device_wdir = device_wdir
        self.timeout = timeout

        self.server_ctx = None
        self.server = None
        self.worker = None

    def __enter__(self):
        if self.device_addr is not None and self.spawn_server:
            if not isinstance(self.device_addr, str):
                host, port = self.device_addr
            else:
                host, port = self.device_addr, None

            self.server_ctx = server.tmp_ssh_server(
                host=host,
                user=self.device_user,
                passwd=self.device_passwd,
                wdir=self.device_wdir,
                server_port=port
            )
            self.server = self.server_ctx.__enter__() # pylint: disable=no-member

        try:
            self.worker = functools.partial(self.impl_func, **self.target_args)
        except:
            if self.server_ctx:
                self.server_ctx.__exit__(*sys.exc_info()) # pylint: disable=no-member
                self.server_ctx = None

            raise

        return self

    def __exit__(self, *args):
        if self.worker:
            try:
                self.worker.close()
                if not self.worker.wait(timeout=self.timeout):
                    self.worker.terminate()
            except:
                pass

        if self.server_ctx:
            try:
                self.server_ctx.__exit__(*args)
            except:
                pass
            self.server_ctx = None

    def run(self, *args, **kwargs):
        return self.worker(self.metric, *args, **kwargs)
