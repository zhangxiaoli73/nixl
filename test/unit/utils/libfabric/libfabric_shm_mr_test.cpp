/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file libfabric_shm_mr_test.cpp
 * @brief Unit test for libfabric provider memory registration with unique keys.
 *
 * This test verifies that multiple memory registrations succeed by using
 * unique requested_key values. Providers that use application-selected keys
 * (not FI_MR_PROV_KEY) require each registration to have a unique key.
 *
 * Usage: libfabric_shm_mr_test [provider_name]
 *   If provider_name is not specified, tries shm first, then tcp as fallback.
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#include "common/nixl_log.h"

// Number of memory regions to register (tests unique key generation)
constexpr int NUM_MR_REGISTRATIONS = 10;
constexpr size_t BUFFER_SIZE = 4096;

// List of providers to try (in order of preference)
const char* PROVIDERS_TO_TRY[] = {"shm", "tcp", "sockets", nullptr};

int main(int argc, char *argv[]) {
    struct fi_info *hints = nullptr;
    struct fi_info *info = nullptr;
    struct fid_fabric *fabric = nullptr;
    struct fid_domain *domain = nullptr;
    std::vector<void*> buffers;
    std::vector<struct fid_mr*> mrs;
    int ret = 0;
    const char *provider_name = nullptr;

    NIXL_INFO << "=== Testing Multiple Memory Registration with Unique Keys ===";
    NIXL_INFO << "This test verifies that multiple fi_mr_regattr calls succeed";
    NIXL_INFO << "by using unique requested_key values for each registration.";

    // Setup hints
    hints = fi_allocinfo();
    if (!hints) {
        NIXL_ERROR << "Failed to allocate fi_info";
        return 1;
    }

    hints->ep_attr->type = FI_EP_RDM;
    hints->caps = FI_MSG | FI_RMA;
    hints->mode = 0;
    // Let the provider choose mr_mode, or use FI_MR_VIRT_ADDR which shm requires
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR;

    // Use FI_VERSION(1, 20) for broader compatibility
    uint32_t fi_version = FI_VERSION(1, 20);

    // Try to find an available provider
    if (argc > 1) {
        // User specified a provider
        provider_name = argv[1];
        hints->fabric_attr->prov_name = strdup(provider_name);
        NIXL_INFO << "1. Getting fabric info for specified provider: " << provider_name << "...";
        ret = fi_getinfo(fi_version, nullptr, nullptr, 0, hints, &info);
    } else {
        // Try providers in order
        NIXL_INFO << "1. Searching for available provider...";
        for (int i = 0; PROVIDERS_TO_TRY[i] != nullptr; i++) {
            if (hints->fabric_attr->prov_name) {
                free(hints->fabric_attr->prov_name);
            }
            hints->fabric_attr->prov_name = strdup(PROVIDERS_TO_TRY[i]);
            NIXL_INFO << "   Trying provider: " << PROVIDERS_TO_TRY[i] << "...";
            ret = fi_getinfo(fi_version, nullptr, nullptr, 0, hints, &info);
            if (ret == 0) {
                provider_name = PROVIDERS_TO_TRY[i];
                break;
            }
            NIXL_INFO << "   " << PROVIDERS_TO_TRY[i] << " not available: " << fi_strerror(-ret);
        }
    }

    if (ret) {
        NIXL_ERROR << "fi_getinfo failed: " << fi_strerror(-ret);
        NIXL_ERROR << "No suitable provider available on this system";
        fi_freeinfo(hints);
        return 1;
    }
    NIXL_INFO << "   SUCCESS: Got info for provider: " << info->fabric_attr->prov_name;

    NIXL_INFO << "2. Creating fabric...";
    ret = fi_fabric(info->fabric_attr, &fabric, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_fabric failed: " << fi_strerror(-ret);
        fi_freeinfo(info);
        fi_freeinfo(hints);
        return 1;
    }
    NIXL_INFO << "   SUCCESS: Fabric created";

    NIXL_INFO << "3. Creating domain...";
    ret = fi_domain(fabric, info, &domain, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_domain failed: " << fi_strerror(-ret);
        fi_close(&fabric->fid);
        fi_freeinfo(info);
        fi_freeinfo(hints);
        return 1;
    }
    NIXL_INFO << "   SUCCESS: Domain created";

    NIXL_INFO << "4. Registering " << NUM_MR_REGISTRATIONS << " memory regions...";

    // Allocate buffers
    for (int i = 0; i < NUM_MR_REGISTRATIONS; i++) {
        void *buf = aligned_alloc(4096, BUFFER_SIZE);
        if (!buf) {
            NIXL_ERROR << "Failed to allocate buffer " << i;
            ret = 1;
            goto cleanup;
        }
        memset(buf, 0, BUFFER_SIZE);
        buffers.push_back(buf);
    }

    // Register memory regions with unique keys
    for (int i = 0; i < NUM_MR_REGISTRATIONS; i++) {
        struct fid_mr *mr = nullptr;
        struct fi_mr_attr mr_attr = {};
        struct iovec iov = {};

        iov.iov_base = buffers[i];
        iov.iov_len = BUFFER_SIZE;

        mr_attr.mr_iov = &iov;
        mr_attr.iov_count = 1;
        mr_attr.access = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;

        // KEY FIX: Use unique requested_key for each registration
        // Without this, the second registration would fail with -FI_ENOKEY or similar
        mr_attr.requested_key = static_cast<uint64_t>(i + 1);

        ret = fi_mr_regattr(domain, &mr_attr, 0, &mr);
        if (ret) {
            NIXL_ERROR << "   FAILED: fi_mr_regattr for region " << i << ": " << fi_strerror(-ret);
            NIXL_ERROR << "   This indicates the unique key fix is not working!";
            goto cleanup;
        }

        uint64_t key = fi_mr_key(mr);
        NIXL_INFO << "   Region " << i << ": registered with key=" << key;
        mrs.push_back(mr);
    }

    NIXL_INFO << "   SUCCESS: All " << NUM_MR_REGISTRATIONS << " memory regions registered!";
    NIXL_INFO << "";
    NIXL_INFO << "=== TEST PASSED ===";
    NIXL_INFO << "Multiple memory registrations with shm provider succeeded.";
    NIXL_INFO << "The unique requested_key fix is working correctly.";
    ret = 0;

cleanup:
    // Cleanup MRs
    for (auto mr : mrs) {
        if (mr) fi_close(&mr->fid);
    }

    // Cleanup buffers
    for (auto buf : buffers) {
        if (buf) free(buf);
    }

    // Cleanup libfabric resources
    if (domain) fi_close(&domain->fid);
    if (fabric) fi_close(&fabric->fid);
    if (info) fi_freeinfo(info);
    if (hints) fi_freeinfo(hints);

    return ret;
}

