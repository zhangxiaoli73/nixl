#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <iostream>
#include "libfabric_common.h"

// Provider configuration structure
struct ProviderConfigTest {
    std::string name;
    uint64_t caps;
    uint64_t mode;
    uint64_t mr_mode;
    fi_resource_mgmt resource_mgmt;
    fi_threading threading;
};

static const ProviderConfigTest PROVIDER_CONFIGS[] = {
    {
        "efa",
        FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM,
        FI_CONTEXT | FI_CONTEXT2,
        0,  // let provider choose
        FI_RM_UNSPEC,
        FI_THREAD_SAFE
    },
    {
        "verbs",  // Matches both "verbs" and "verbs;ofi_rxm"
        FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_MULTI_RECV | FI_LOCAL_COMM | FI_REMOTE_COMM | FI_HMEM,
        0,  // no mode flags required
        FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_HMEM,
        FI_RM_ENABLED,
        FI_THREAD_SAFE
    },
    {
        "tcp",
        FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM,
        FI_CONTEXT | FI_CONTEXT2,
        0,  // basic MR mode, overridden in rail.cpp
        FI_RM_UNSPEC,
        FI_THREAD_UNSPEC
    },
    {
        "sockets",
        FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM,
        0,
        0,  // let provider choose
        FI_RM_UNSPEC,
        FI_THREAD_UNSPEC  // default threading
    },
    {
        "shm",
        FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_MULTI_RECV | FI_LOCAL_COMM | FI_RMA_EVENT | FI_SOURCE | FI_DIRECTED_RECV | FI_HMEM,
        0,
        0, //FI_MR_VIRT_ADDR| FI_MR_HMEM,
        FI_RM_ENABLED,
        FI_THREAD_SAFE  // default threading
    }
};

static const size_t NUM_PROVIDER_CONFIGS = sizeof(PROVIDER_CONFIGS) / sizeof(PROVIDER_CONFIGS[0]);


void configureHintsForProvider(struct fi_info* hints, const std::string& provider_name) {
    const ProviderConfigTest* config = nullptr;

    // Find matching config
    // Match order: 1) exact match, 2) prefix match for composite providers (e.g., "verbs;ofi_rxm")
    for (size_t i = 0; i < NUM_PROVIDER_CONFIGS; ++i) {
        const std::string& config_name = PROVIDER_CONFIGS[i].name;

        // Exact match
        if (provider_name == config_name) {
            config = &PROVIDER_CONFIGS[i];
            break;
        }

        // Composite provider match (e.g., "verbs;ofi_rxm" matches "verbs")
        // Check if provider_name starts with config_name followed by ";"
        if (provider_name.rfind(config_name + ";", 0) == 0) {
            config = &PROVIDER_CONFIGS[i];
            break;
        }
    }

    if (!config) {
        // Default configuration
        std::cout << "No specific config for provider '" << provider_name << "', using defaults" << std::endl;
        hints->caps = FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM;
        hints->mode = 0;
        hints->ep_attr->type = FI_EP_RDM;
        return;
    }

    // Apply provider-specific configuration
    hints->caps = config->caps;
    hints->mode = config->mode;
    hints->ep_attr->type = FI_EP_RDM;

    if (config->resource_mgmt != FI_RM_UNSPEC) {
        hints->domain_attr->resource_mgmt = config->resource_mgmt;
    }

    if (config->mr_mode != 0) {
        hints->domain_attr->mr_mode = config->mr_mode;
    }

    if (config->threading != FI_THREAD_UNSPEC) {
        hints->domain_attr->threading = config->threading;
    }
}

int init_provider(char* provider_name) {

        // Get fabric device info with PCIe addresses from libfabric
    struct fi_info *hints, *info;

    hints = fi_allocinfo();
    if (!hints) {
        std::cerr << "Failed to alloc fi_info" << std::endl;
        return -1;
    }

    // Configure hints for the discovered provider
    // This ensures consistency between device discovery and PCIe mapping
    hints->fabric_attr->prov_name = strdup(provider_name);
    configureHintsForProvider(hints, provider_name);

    // Use FI_VERSION(1, 18) for DMABUF and HMEM support
    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
    if (ret) {
                std::cerr << "fi_getinfo failed with provider " << provider_name << std::endl;
        fi_freeinfo(hints);
        return -1;
    }


    std::cout << "verbs provider initialized successfully." << std::endl;

    fi_freeinfo(info);
    fi_freeinfo(hints);
    return 0;
}

int main(int argc, char **argv) {
    char *provider_name = NULL;
        provider_name = argv[1];
        std::cout << "Input provider as " << provider_name << std::endl;
    auto network_device = LibfabricUtils::getAvailableNetworkDevices();
    return 0;
    //return init_provider(provider_name);
}


