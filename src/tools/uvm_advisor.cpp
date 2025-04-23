#include "tools/uvm_advisor.h"
#include "utils/helper.h"
#include "utils/hash.h"
#include "gpu_patch.h"
#include "cpp_trace.h"
#include "py_frame.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <vector>
#include <string>
#include <stack>
#include <memory>
#include <iostream>


using namespace yosemite;

typedef enum {
    MEMCPY_UNKNOWN = 0,
    MEMCPY_H2H = 1,
    MEMCPY_H2D = 2,
    MEMCPY_D2H = 3,
    MEMCPY_D2D = 4,
} MemcpyDirection_t;

static Timer_t _timer;

struct CpyStats {
    uint64_t count = 0;
    uint64_t size = 0;
};

struct SetStats {
    uint64_t count = 0;
    uint64_t size = 0;
};

struct MemStats {
    uint64_t alloc_count = 0;
    uint64_t alloc_size = 0;
    uint64_t free_count = 0;
    uint64_t free_size = 0;
};

struct TenStats {
    uint64_t alloc_count = 0;
    uint64_t alloc_size = 0;
    uint64_t free_count = 0;
    uint64_t free_size = 0;
};

struct OpStats {
    uint64_t count = 0;
    uint64_t group_count = 0;
    uint64_t pending_kernels = 0;
};


static std::map<MemcpyDirection_t, CpyStats> cpy_stats;
static SetStats set_stats;
static MemStats mem_stats;
static TenStats ten_stats;
static uint64_t kernel_count = 0;
static std::stack<std::string> op_stack;
static OpStats op_stats;


inline std::string vector2str(std::vector<std::string> &vec, int skip_first = 0, int skip_last = 0) {
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


UVMAdvisor::UVMAdvisor() : Tool(UVM_ADVISOR) {
    init();

    // out_fp = fopen("uvm_advisor.txt", "w");
    out_fp = stdout;
}


UVMAdvisor::~UVMAdvisor() {
    // fclose(out_fp);
}

void UVMAdvisor::init() {
    const char* env_name = std::getenv("ACCEL_PROF_HOME");
    std::string lib_path;
    if (env_name) {
        lib_path = std::string(env_name) + "/lib/libcompute_sanitizer.so";
    }
    init_backtrace(lib_path.c_str());

}


void UVMAdvisor::evt_callback(EventPtr_t evt) {
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
        case EventType_OP_START:
            op_start_callback(std::dynamic_pointer_cast<OpStart_t>(evt));
            break;
        case EventType_OP_END:
            op_end_callback(std::dynamic_pointer_cast<OpEnd_t>(evt));
            break;
        default:
            break;
    }
}


void UVMAdvisor::kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel) {
    kernel_count++;
    fprintf(out_fp, "[Kernel] id: %lu, name: %s\n", kernel_count, kernel->kernel_name.c_str());
    fflush(out_fp);
    op_stats.pending_kernels++;
    _timer.increment(true);
}


void UVMAdvisor::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
}


void UVMAdvisor::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    mem_stats.alloc_count++;
    mem_stats.alloc_size += mem->size;
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);

    _timer.increment(true);
}


void UVMAdvisor::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    mem_stats.free_count++;
    mem_stats.free_size += mem->size;

    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}



void UVMAdvisor::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
    // auto backtraces = get_backtrace();
    // auto py_frames = get_pyframes();
    // auto bt_str = vector2str(backtraces);
    // auto pf_str = vector2str(py_frames);

    // std::cout << "Backtrace hash: " << sha256(bt_str) << std::endl;
    // std::cout << bt_str << std::endl;
    // std::cout << "Python frame hash: " << sha256(pf_str) << std::endl;
    // std::cout << pf_str << std::endl;

    MemcpyDirection_t direction = (MemcpyDirection_t)mem->direction;
    if (cpy_stats.find(direction) == cpy_stats.end()) {
        cpy_stats[direction] = CpyStats{0, 0};
    }
    cpy_stats[direction].count++;
    cpy_stats[direction].size += mem->size;

    _timer.increment(true);
}


void UVMAdvisor::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
    set_stats.count++;
    set_stats.size += mem->size;

    _timer.increment(true);
}


void UVMAdvisor::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    ten_stats.alloc_count++;
    ten_stats.alloc_size += ten->size;

    tenalloc_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void UVMAdvisor::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    ten_stats.free_count++;
    ten_stats.free_size += -ten->size;

    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void UVMAdvisor::op_start_callback(std::shared_ptr<OpStart_t> op) {
    op_stack.push(op->op_name);
    op_stats.count++;

    // if (op->op_name == "aten::matmul") {
    //     auto backtraces = get_backtrace();
    //     auto py_frames = get_pyframes();
    //     auto bt_str = vector2str(backtraces);
    //     auto pf_str = vector2str(py_frames);

    //     std::cout << "Backtrace hash: " << sha256(bt_str) << std::endl;
    //     std::cout << bt_str << std::endl;
    //     std::cout << "Python frame hash: " << sha256(pf_str) << std::endl;
    //     std::cout << pf_str << std::endl;
    // }

    _timer.increment(true);
}


void UVMAdvisor::op_end_callback(std::shared_ptr<OpEnd_t> op) {
    op_stack.pop();
    if (op_stack.empty()) {
        op_stats.group_count++;
        fprintf(out_fp, "op_name: %s, gid: %lu, oid: %lu ------------------+++++-------------,kid: %lu, p_k: %lu\n",
                op->op_name.c_str(), op_stats.group_count, op_stats.count, op_stats.pending_kernels, kernel_count);
        fflush(out_fp);
        op_stats.pending_kernels = 0;
    }

    _timer.increment(true);
}

void UVMAdvisor::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)data;
    MemoryAccessState* states = tracker->access_state;
    TensorAccessState* tensor_states = tracker->tensor_access_state;
    
    for (uint32_t i = 0; i < states->size; i++) {
        if (states->touch[i] == 1) {
            fprintf(out_fp, "Memory access: %lu, size: %lu\n", states->start_end[i].start, states->start_end[i].end - states->start_end[i].start);
        }
    }

    for (uint32_t i = 0; i < tensor_states->size; i++) {
        if (tensor_states->touch[i] == 1) {
            fprintf(out_fp, "Tensor access: %lu, size: %lu\n", tensor_states->start_end[i].start, tensor_states->start_end[i].end - tensor_states->start_end[i].start);
        }
    }
}


void UVMAdvisor::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    MemoryRange* _ranges = (MemoryRange*)ranges;
    *count = 0;
    for (auto mem : active_memories) {
        _ranges[*count].start = mem.second->addr;
        _ranges[*count].end = mem.second->addr + mem.second->size;
        (*count)++;
        if (*count >= limit) {
            fprintf(out_fp, "Warning: query_ranges limit reached\n");
            break;
        }
    }
}

void UVMAdvisor::query_tensors(void* ranges, uint32_t limit, uint32_t* count) {
    MemoryRange* _ranges = (MemoryRange*)ranges;
    *count = 0;
    for (auto ten : active_tensors) {
        _ranges[*count].start = ten.second->addr;
        _ranges[*count].end = ten.second->addr + ten.second->size;
        (*count)++;
        if (*count >= limit) {
            fprintf(out_fp, "Warning: query_tensors limit reached\n");
            break;
        }
    }
}


void UVMAdvisor::flush() {
    
    fprintf(out_fp, "--------------------------------------------------------------------------------\n");
    fprintf(out_fp, "%-12s count: %-10lu\n", "[Kernel]", kernel_count);
    fprintf(out_fp, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[MemMalloc]", mem_stats.alloc_count, mem_stats.alloc_size, format_size(mem_stats.alloc_size).c_str());
    fprintf(out_fp, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[MemFree]", mem_stats.free_count, mem_stats.free_size, format_size(mem_stats.free_size).c_str());
    fprintf(out_fp, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[Memset]", set_stats.count, set_stats.size, format_size(set_stats.size).c_str());

    for (auto& it : cpy_stats) {
        const char* direction = it.first == MEMCPY_H2H ? "H2H" : 
                              it.first == MEMCPY_H2D ? "H2D" :
                              it.first == MEMCPY_D2H ? "D2H" : 
                              it.first == MEMCPY_D2D ? "D2D" : "N/A";
        fprintf(out_fp, "[Memcpy-%s] count: %-10lu, size: %lu (%s)\n",
                direction, it.second.count, it.second.size, format_size(it.second.size).c_str());
    }

    fprintf(out_fp, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[TenMalloc]", ten_stats.alloc_count, ten_stats.alloc_size, format_size(ten_stats.alloc_size).c_str());
    fprintf(out_fp, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[TenFree]", ten_stats.free_count, ten_stats.free_size, format_size(ten_stats.free_size).c_str());
    fprintf(out_fp, "%-12s count: %-10lu\n", "[Op]", op_stats.count);
    fprintf(out_fp, "%-12s count: %-10lu\n", "[OpGroup]", op_stats.group_count);
    fprintf(out_fp, "--------------------------------------------------------------------------------\n");
}
