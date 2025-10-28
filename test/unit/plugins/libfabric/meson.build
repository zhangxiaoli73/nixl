# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Intel Corporation. All rights reserved.
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


# Libfabric plugin integration tests

libfabric_backend_dep = declare_dependency(
    link_with: libfabric_backend_lib,
    include_directories: [nixl_inc_dirs, '../../../../src/plugins/libfabric']
)

# Collect compile flags
compile_flags = []
additional_deps = []

if cuda_dep.found()
    additional_deps += [cuda_dep]
    compile_flags += ['-DHAVE_CUDA']
endif

if synapseai_dep.found()
    additional_deps += [synapseai_dep]
    compile_flags += ['-DHAVE_SYNAPSEAI']
endif

if sycl_dep.found()
    additional_deps += [sycl_dep]
    compile_flags += ['-DHAVE_SYCL -I/usr/include -L/usr/lib -lze_loader']
endif

libfabric_test_deps = [
    nixl_dep,
    nixl_infra,
    nixl_common_deps,
    libfabric_utils_dep,
    libfabric_backend_dep,
    libfabric_dep,
    thread_dep,
] + additional_deps

# Integration test: Multi-descriptor offset handling with actual backend
test_backend_integration = executable(
    'test_libfabric_backend_integration',
    'libfabric_backend_integration_test.cpp',
    dependencies: libfabric_test_deps,
    cpp_args: compile_flags,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/libfabric'],
    install: false,
)

test_backend_integration_xpu = executable(
    'test_libfabric_backend_integration_xpu',
    'libfabric_backend_integration_test_xpu.cpp',
    dependencies: libfabric_test_deps,
    cpp_args: compile_flags,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/libfabric'],
    install: false,
)

test_fi_info = executable(
    'test_fi_info',
    'test_fi_info.cpp',
    dependencies: libfabric_test_deps,
    cpp_args: compile_flags,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/libfabric'],
    install: false,
)

test_mr_register = executable(
    'test_mr_register',
    'test_mr_register.cpp',
    dependencies: libfabric_test_deps,
    cpp_args: compile_flags,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/libfabric'],
    install: false,
)

test_mr_register_verbs = executable(
    'test_mr_register_verbs',
    'test_mr_register_verbs.cpp',
    dependencies: libfabric_test_deps,
    cpp_args: compile_flags,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/libfabric'],
    install: false,
)

test('libfabric_backend_integration',
     test_backend_integration,
     suite: ['unit', 'plugins', 'libfabric', 'integration'],
     timeout: 120,
)

# Future integration tests can be added here following the same pattern:
# - Multi-GPU transfers
# - Different memory types (DRAM, VRAM)
# - Notification flow
# - Connection management

