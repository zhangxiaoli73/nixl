# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ucx_backend_dep = declare_dependency(link_with: ucx_backend_lib, include_directories: [nixl_inc_dirs, '../../../../src/plugins/ucx'])

if cuda_dep.found()
    cuda_dependencies = [cuda_dep]
    cpp_args = '-DHAVE_CUDA'
else
    cuda_dependencies = []
    cpp_args = '-UHAVE_CUDA'
endif

ucx_backend_test = executable('ucx_backend_test',
        'ucx_backend_test.cpp',
        dependencies: [nixl_dep, nixl_infra, nixl_common_deps, ucx_backend_dep, ucx_dep, thread_dep] + cuda_dependencies,
        include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/ucx'],
        cpp_args : cpp_args,
        install: true)

level_zero_inc = include_directories('/usr/include/')
level_zero_libdir = '/usr/lib/x86_64-linux-gnu'

level_zero_dep = declare_dependency(
  include_directories: level_zero_inc,
  link_args: ['-L' + level_zero_libdir, '-lze_loader']
)


ucx_backend_test_xpu = executable('ucx_backend_test_xpu',
    'ucx_backend_test_xpu.cpp',
    dependencies: [nixl_dep, nixl_infra, nixl_common_deps, ucx_backend_dep, ucx_dep, thread_dep] + cuda_dependencies + [level_zero_dep],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/ucx', level_zero_inc],
    cpp_args: cpp_args,
    install: true
)

ucx_backend_multi = executable('ucx_backend_multi',
           'ucx_backend_multi.cpp',
           dependencies: [nixl_dep, nixl_infra, nixl_common_deps, ucx_backend_dep, ucx_dep, thread_dep] + cuda_dependencies,
           include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/ucx', ],
           cpp_args : cpp_args,
           install: true)
