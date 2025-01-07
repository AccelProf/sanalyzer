
#include "tools/code_check.h"
#include "utils/helper.h"
#include "utils/hash.h"
#include "gpu_patch.h"
#include "cxx_backtrace.h"
#include "python_frame.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <iostream>


using namespace yosemite;



static Timer_t _timer;


static std::map<uint64_t, std::shared_ptr<KernelLauch_t>> kernel_events;
static std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
static std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;


std::string vector2str(std::vector<std::string> &vec, int skip_first = 0, int skip_last = 0) {
    if (skip_first + skip_last > vec.size()) {
        printf("Skip first and skip last are larger than the vector size\n");
        return "";
    }
    std::string str;
    for (size_t i = skip_first; i < vec.size() - skip_last; i++) {
        str += vec[i] + "\n";
    }
    return str;
}

void CodeCheck::init() {
    const char* env_name = std::getenv("CU_PROF_HOME");
    std::string lib_path;
    if (env_name) {
        lib_path = std::string(env_name) + "/lib/libcompute_sanitizer.so";
    }
    init_backtrace(lib_path.c_str());

}


void CodeCheck::evt_callback(EventPtr_t evt) {
    switch (evt->evt_type) {
        case EventType_KERNEL_LAUNCH:
            kernel_start_callback(std::dynamic_pointer_cast<KernelLauch_t>(evt));
            break;
        case EventType_KERNEL_END:
            kernel_end_callback(std::dynamic_pointer_cast<KernelEnd_t>(evt));
            break;
        case EventType_MEM_ALLOC:
            mem_alloc_callback(std::dynamic_pointer_cast<MemAlloc_t>(evt));
            break;
        case EventType_MEM_FREE:
            mem_free_callback(std::dynamic_pointer_cast<MemFree_t>(evt));
            break;
        case EventType_MEM_COPY:
            mem_cpy_callback(std::dynamic_pointer_cast<MemCpy_t>(evt));
            break;
        case EventType_MEM_SET:
            mem_set_callback(std::dynamic_pointer_cast<MemSet_t>(evt));
            break;
        case EventType_TEN_ALLOC:
            ten_alloc_callback(std::dynamic_pointer_cast<TenAlloc_t>(evt));
            break;
        case EventType_TEN_FREE:
            ten_free_callback(std::dynamic_pointer_cast<TenFree_t>(evt));
            break;
        default:
            break;
    }
}


void CodeCheck::kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel) {

    _timer.increment(true);
}


void CodeCheck::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
}


void CodeCheck::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    

    _timer.increment(true);
}


void CodeCheck::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
   

    _timer.increment(true);
}


void CodeCheck::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
    
    std::cout << "Memory copy detected" << std::endl;
    std::cout << mem->src_addr << " " << mem->dst_addr << " " << mem->size << std::endl;
    std::cout << "cudaMemcpy is async: " << mem->is_async << std::endl;
    std::cout << "Direction: " << mem->direction << std::endl;

    auto backtraces = get_backtrace();
    auto py_frames = get_pyframes();
    auto bt_str = vector2str(backtraces);
    auto pf_str = vector2str(py_frames);

    std::cout << bt_str << std::endl;
    std::cout << pf_str << std::endl;
    std::cout << "Backtrace: " << sha256(bt_str) << std::endl;
    std::cout << "Python frames: " << sha256(pf_str) << std::endl;

    _timer.increment(true);
}


void CodeCheck::mem_set_callback(std::shared_ptr<MemSet_t> mem) {

    std::cout << "Memory set detected" << std::endl;
    std::cout << mem->addr << " " << mem->size << " " << mem->value << std::endl;
    std::cout << "cudaMemset is async: " << mem->is_async << std::endl;
    std::cout << "Set value: " << mem->value << std::endl;

    // auto backtraces = get_backtrace();
    // auto py_frames = get_pyframes();
    // auto bt_str = vector2str(backtraces);
    // auto pf_str = vector2str(py_frames);

    // std::cout << bt_str << std::endl;
    // std::cout << pf_str << std::endl;
    // std::cout << "Backtrace: " << sha256(bt_str) << std::endl;
    // std::cout << "Python frames: " << sha256(pf_str) << std::endl;

    _timer.increment(true);
}


void CodeCheck::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    std::cout << "[Tensor malloc] " << ten->addr << " " << ten->size << std::endl;
}


void CodeCheck::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    std::cout << "[Tensor free] " << ten->addr << " " << ten->size << std::endl;
}


void CodeCheck::gpu_data_analysis(void* data, uint64_t size) {
    
}


void CodeCheck::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {

}


void CodeCheck::flush() {
    
}