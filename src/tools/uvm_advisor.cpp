#include "tools/uvm_advisor.h"
#include "utils/helper.h"
#include "utils/hash.h"
#include "gpu_patch.h"
#include "cpp_trace.h"
#include "py_frame.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <iostream>


using namespace yosemite;

#define SANITIZER_UVM_MEMORY_FLAG 0x6
#define LARGE_TENSOR_THRESHOLD 1048576

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

    out_fp = fopen("uvm_advisor.log", "w");
    // out_fp = stdout;
}


UVMAdvisor::~UVMAdvisor() {
    fclose(out_fp);
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
    opt_keys.kernel_id ++;
    kernel->key = opt_keys.kernel_id;
    kernel->timestamp = _timer.get();
    kernel_events.push_back(kernel);
    op_stats.pending_kernels++;
    _timer.increment(true);
}


void UVMAdvisor::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    _timer.increment(true);
}


void UVMAdvisor::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    if (mem->alloc_type != SANITIZER_UVM_MEMORY_FLAG) {
        return;
    }
    opt_keys.mem_id ++;
    mem->key = opt_keys.mem_id;
    mem->timestamp = _timer.get();
    mem_stats.alloc_count++;
    mem_stats.alloc_size += mem->size;
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);

    _timer.increment(true);
}


void UVMAdvisor::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    if (mem->alloc_type != SANITIZER_UVM_MEMORY_FLAG) {
        return;
    }

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
    if (ten->size <= LARGE_TENSOR_THRESHOLD) {
        return;
    }
    opt_keys.ten_id ++;
    ten->key = opt_keys.ten_id;
    ten_stats.alloc_count++;
    ten_stats.alloc_size += ten->size;

    ten->timestamp = _timer.get();
    tenalloc_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void UVMAdvisor::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    if (-ten->size <= LARGE_TENSOR_THRESHOLD) {
        return;
    }

    ten_stats.free_count++;
    ten_stats.free_size += -ten->size;

    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void UVMAdvisor::op_start_callback(std::shared_ptr<OpStart_t> op) {
    opt_keys.op_id ++;
    op->key = opt_keys.op_id;
    op->timestamp = _timer.get();
    op_stack.push(op);
    op_stats.count++;
    op_stats.pending_ops++;

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
    auto op_start = op_stack.top();
    op_stack.pop();
    if (op_stack.empty()) {
        if (op_stats.pending_kernels > 0) {
            assert(op_tables.find(op_start->timestamp) == op_tables.end());
            op_start->end_time = _timer.get();
            op_start->pending_kernels = op_stats.pending_kernels;
            op_start->pending_ops = op_stats.pending_ops;
            op_tables[op_start->timestamp] = std::make_pair(op_start, kernel_resources);
        }
        op_stats.group_count++;
        op_stats.pending_kernels = 0;
        op_stats.pending_ops = 0;
        kernel_resources.clear();
    }

    _timer.increment(true);
}

void UVMAdvisor::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)data;
    MemoryAccessState* states = tracker->access_state;
    TensorAccessState* tensor_states = tracker->tensor_access_state;

    MemAllocVec mem_alloc_vec;
    TenAllocVec ten_alloc_vec;

    for (uint32_t i = 0; i < states->size; i++) {
        if (states->touch[i] == 1) {
            auto mem = active_memories.find(states->start_end[i].start);
            mem_alloc_vec.push_back(mem->second);
        }
    }

    for (uint32_t i = 0; i < tensor_states->size; i++) {
        if (tensor_states->touch[i] == 1) {
            auto ten = active_tensors.find(tensor_states->start_end[i].start);
            ten_alloc_vec.push_back(ten->second);
        }
    }

    auto kernel = kernel_events.back();
    kernel_resources.push_back(std::make_tuple(kernel, mem_alloc_vec, ten_alloc_vec));
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

    for (auto& it : op_tables) {
        auto op = it.second.first;
        fprintf(out_fp, "Op - %.30s, timestamp: %lu, pending_ops: %lu, pending_kernels: %lu\n", 
                op->op_name.c_str(), op->timestamp, op->pending_ops, op->pending_kernels);
        for (auto& kernel_tuple : it.second.second) {
            auto kernel = std::get<0>(kernel_tuple);
            auto mem_alloc_vec = std::get<1>(kernel_tuple);
            auto ten_alloc_vec = std::get<2>(kernel_tuple);
            fprintf(out_fp, "   Kernel: %.30s, timestamp: %lu\n", kernel->kernel_name.c_str(), kernel->timestamp);
            fprintf(out_fp, "       MemAlloc (%lu): ", mem_alloc_vec.size());
            for (auto& mem : mem_alloc_vec) {
                fprintf(out_fp, "(%p, %lu) ", mem->addr, mem->size);
            }
            fprintf(out_fp, "\n");
            fprintf(out_fp, "       TenAlloc (%lu): ", ten_alloc_vec.size());
            for (auto& ten : ten_alloc_vec) {
                fprintf(out_fp, "(%p, %lu) ", ten->addr, ten->size);
            }
            fprintf(out_fp, "\n");
        }
    }

    fprintf(out_fp, "================================================================================\n");
     for (auto& it : op_tables) {
        auto op = it.second.first;
        fprintf(out_fp, "Op - %.30s, op_id: %lu, pending_ops: %lu, pending_kernels: %lu\n", 
                op->op_name.c_str(), op->key, op->pending_ops, op->pending_kernels);
        for (auto& kernel_tuple : it.second.second) {
            auto kernel = std::get<0>(kernel_tuple);
            auto mem_alloc_vec = std::get<1>(kernel_tuple);
            auto ten_alloc_vec = std::get<2>(kernel_tuple);
            fprintf(out_fp, "   Kernel: %.30s, kernel_id: %lu\n", kernel->kernel_name.c_str(), kernel->key);
            fprintf(out_fp, "       MemAlloc (%lu): ", mem_alloc_vec.size());
            for (auto& mem : mem_alloc_vec) {
                fprintf(out_fp, "%lu:(%lu, %lu), ", mem->key, mem->addr, mem->size);
            }
            fprintf(out_fp, "\n");
            fprintf(out_fp, "       TenAlloc (%lu): ", ten_alloc_vec.size());
            for (auto& ten : ten_alloc_vec) {
                fprintf(out_fp, "%lu:(%lu, %lu), ", ten->key, ten->addr, ten->size);
            }
            fprintf(out_fp, "\n");
        }
    }
}
