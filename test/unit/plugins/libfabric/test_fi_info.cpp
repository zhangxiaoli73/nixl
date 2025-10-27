#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <iostream>
#include "libfabric_common.h"
#include "libfabric_rail.h"

#include <level_zero/ze_api.h>
using namespace std;


static std::vector<ze_device_handle_t> devices_list;
ze_context_handle_t global_context = nullptr;

int initializeXPU() {
     ze_result_t status;

    // 1️⃣ 初始化 Level Zero
    status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (status != ZE_RESULT_SUCCESS) {
        std::cerr << "zeInit failed\n";
        return -1;
    }

    // 2️⃣ 获取 driver
    uint32_t driverCount = 0;
    status = zeDriverGet(&driverCount, nullptr);
    if (status != ZE_RESULT_SUCCESS || driverCount == 0) {
        std::cerr << "No L0 drivers found\n";
        return -1;
    }

    std::vector<ze_driver_handle_t> drivers(driverCount);
    status = zeDriverGet(&driverCount, drivers.data());
    if (status != ZE_RESULT_SUCCESS) {
        std::cerr << "zeDriverGet failed\n";
        return -1;
    } else {
        std::cout << "Find driver number is " << driverCount << std::endl;
    }

    ze_driver_handle_t driver = drivers[0]; // 使用第一个 driver

    // 3️⃣ 获取 GPU device
    uint32_t deviceCount = 0;
    status = zeDeviceGet(driver, &deviceCount, nullptr);
    if (status != ZE_RESULT_SUCCESS || deviceCount == 0) {
        std::cerr << "No devices found\n";
        return -1;
    }

    std::vector<ze_device_handle_t> devices(deviceCount);
    status = zeDeviceGet(driver, &deviceCount, devices.data());
    if (status != ZE_RESULT_SUCCESS) {
        std::cerr << "zeDeviceGet failed\n";
        return -1;
    }

    for (auto device : devices) {
        ze_device_properties_t props;
        status = zeDeviceGetProperties(device, &props);
        if (status == ZE_RESULT_SUCCESS && props.type == ZE_DEVICE_TYPE_GPU) {
            std::cout << "Using GPU: " << props.name << "\n";
            devices_list.emplace_back(device);
        }
    }

    std::cout << "!!! Devices vector length: " << devices_list.size() << std::endl;

      // 4️⃣ 创建 context
    ze_context_desc_t context_desc = {};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    context_desc.pNext = nullptr;
    context_desc.flags = 0;

    status = zeContextCreate(driver, &context_desc, &global_context);
    if (status != ZE_RESULT_SUCCESS) {
        std::cerr << "zeContextCreate failed\n";
        return -1;
    }
    std::cout << "!!! Context create done" << std::endl;
    return 0;
}

void* allocate_device_memory(size_t len, int dev_id) {
    ze_device_mem_alloc_desc_t device_desc = {};
    device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    device_desc.ordinal = 0;
    device_desc.flags = 0;

    void* device_ptr = nullptr;
    ze_result_t status = zeMemAllocDevice(global_context, &device_desc, len, 1, devices_list[dev_id], &device_ptr);
    if (status != ZE_RESULT_SUCCESS) {
        std::cerr << "zeMemAllocDevice failed\n";
        zeContextDestroy(global_context);
        return nullptr;
    }
    return device_ptr;
}

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

    // Test rail
    ret = initializeXPU();
    if (ret < 0) return -1;

    size_t length = 2048;
    auto xpu_device_ptr = allocate_device_memory(length, 0);

    struct fid_mr *mr;
    uint64_t key;

    auto fabric_rail = nixlLibfabricRail("shm", "shm", static_cast<uint16_t>(0));
    nixl_status_t status = fabric_rail.registerMemory(xpu_device_ptr, length, "ze", 0, &mr, &key);
     if (status != NIXL_SUCCESS) {
         std::cout << "Failed \n" << std::endl;
     } else {
         std::cout << "Passed \n" << std::endl;
     }

    fi_freeinfo(info);
    fi_freeinfo(hints);
    return 0;
}

int main(int argc, char **argv) {
    char *provider_name = NULL;
        provider_name = argv[1];
        std::cout << "Input provider as " << provider_name << std::endl;
//    auto network_device = LibfabricUtils::getAvailableNetworkDevices();
//    return 0;
    return init_provider(provider_name);
}


