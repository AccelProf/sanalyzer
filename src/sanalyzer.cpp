#include "sanalyzer.h"
#include "tools/code_check.h"
#include "tools/app_metric.h"
#include "tools/mem_trace.h"
#include "tools/tool.h"
#include "utils/event.h"


#include <memory>
#include <map>
#include <iostream>

using namespace yosemite;

static std::map<AnalysisTool_t, std::shared_ptr<Tool>> _tools;


YosemiteResult_t yosemite_tool_enable(AnalysisTool_t& tool) {
    const char* tool_name = std::getenv("YOSEMITE_TOOL_NAME");
    if (!tool_name) {
        fprintf(stdout, "No tool name specified.\n");
        return YOSEMITE_NOT_IMPLEMENTED;
    }

    if (std::string(tool_name) == "code_check") {
        tool = CODE_CHECK;
        _tools.emplace(CODE_CHECK, std::make_shared<CodeCheck>());
    } else if (std::string(tool_name) == "app_metric") {
        tool = APP_METRICE;
        _tools.emplace(APP_METRICE, std::make_shared<AppMetrics>());
    } else if (std::string(tool_name) == "mem_trace") {
        tool = MEM_TRACE;
        _tools.emplace(MEM_TRACE, std::make_shared<MemTrace>());
    } else
    {
        fprintf(stdout, "Tool not found.\n");
        return YOSEMITE_NOT_IMPLEMENTED;
    }

    fprintf(stdout, "Enabling %s tool.\n", tool_name);
    fflush(stdout);
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_tool_disable() {
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_flush() {
    for (auto &tool : _tools) {
        tool.second->flush();
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_torch_prof_enable() {
    return YOSEMITE_SUCCESS;
}


/****************************************************************************************
 ********************************** Interface functions *********************************
****************************************************************************************/


YosemiteResult_t yosemite_alloc_callback(uint64_t ptr, uint64_t size, int type) {
    for (auto &tool : _tools) {
        auto mem_alloc = std::make_shared<MemAlloc_t>(ptr, size, type);
        tool.second->evt_callback(mem_alloc);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_free_callback(uint64_t ptr) {
    if (ptr == 0) {
        return YOSEMITE_CUDA_MEMFREE_ZERO;
    }
    for (auto &tool : _tools) {
        auto mem_free = std::make_shared<MemFree_t>(ptr);
        tool.second->evt_callback(mem_free);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_memcpy_callback(uint64_t dst, uint64_t src, uint64_t size, bool is_async, uint32_t direction) {
    for (auto &tool : _tools) {
        auto mem_cpy = std::make_shared<MemCpy_t>(dst, src, size, is_async, direction);
        tool.second->evt_callback(mem_cpy);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_memset_callback(uint64_t dst, uint32_t size, int value, bool is_async) {
    for (auto &tool : _tools) {
        auto mem_set = std::make_shared<MemSet_t>(dst, size, value, is_async);
        tool.second->evt_callback(mem_set);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_kernel_start_callback(std::string kernel_name) {
    for (auto &tool : _tools) {
        auto kernel = std::make_shared<KernelLauch_t>(kernel_name); 
        tool.second->evt_callback(kernel);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_kernel_end_callback(std::string kernel_name) {
    for (auto &tool : _tools) {
        auto kernel = std::make_shared<KernelEnd_t>(kernel_name);
        tool.second->evt_callback(kernel);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_gpu_data_analysis(void* data, uint64_t size) {
    for (auto &tool : _tools) {
        tool.second->gpu_data_analysis(data, size);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_init(SanitizerOptions_t& options) {
    AnalysisTool_t tool;
    yosemite_tool_enable(tool);

    if (tool == CODE_CHECK) {
        options.enable_access_tracking = false;
    } else if (tool == APP_METRICE) {
        options.enable_access_tracking = false;
    } else if (tool == MEM_TRACE) {
        options.enable_access_tracking = true;
    }

    // check env var YOSEMITE_TORCH_PROFILE is 0 or 1
    const char* torch_prof = std::getenv("YOSEMITE_TORCH_PROFILE");
    if (torch_prof && std::string(torch_prof) == "1") {
        fprintf(stdout, "Enabling torch profiler.\n");
        yosemite_torch_prof_enable();
    }

    fprintf(stdout, "================================================================================\n");
    fflush(stdout);
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_terminate() {
    yosemite_flush();
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_tensor_malloc_callback(uint64_t ptr, int64_t alloc_size,
                                    int64_t total_allocated, int64_t total_reserved) {
    for (auto &tool : _tools) {
        auto ten_alloc = std::make_shared<TenAlloc_t>(ptr, alloc_size);
        tool.second->evt_callback(ten_alloc);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_tensor_free_callback(uint64_t ptr, int64_t alloc_size,
                                    int64_t total_allocated, int64_t total_reserved) {
    for (auto &tool : _tools) {
        auto ten_free = std::make_shared<TenFree_t>(ptr);
        tool.second->evt_callback(ten_free);
    }
    return YOSEMITE_SUCCESS;
}
