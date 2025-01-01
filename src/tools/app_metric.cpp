
#include "tools/app_metric.h"
#include "utils/helper.h"
#include "gpu_patch.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <memory>


using namespace yosemite;

typedef struct Stats{
    uint64_t num_allocs;
    uint64_t num_kernels;
    uint64_t cur_mem_usage;
    uint64_t max_mem_usage;
    uint64_t max_mem_accesses_per_kernel;
    uint64_t avg_mem_accesses;
    uint64_t tot_mem_accesses;
    std::string max_mem_accesses_kernel;

    Stats() = default;

    ~Stats() = default;
} Stats_t;

static Stats_t _stats;

static Timer_t _timer;


static std::map<uint64_t, std::shared_ptr<KernelLauch_t>> kernel_events;
static std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
static std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;

static std::map<std::string, uint32_t> kernel_invocations;



void AppMetrics::evt_callback(EventPtr_t evt) {
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
        default:
            break;
    }
}


void AppMetrics::kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel) {
    kernel_events.emplace(_timer.get(), kernel);
    if (kernel_invocations.find(kernel->kernel_name) == kernel_invocations.end()) {
        kernel_invocations.emplace(kernel->kernel_name, 1);
    } else {
        kernel_invocations[kernel->kernel_name]++;
    }

    _stats.num_kernels++;

    _timer.increment(true);
}


void AppMetrics::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
}


void AppMetrics::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);
    
    _stats.num_allocs++;
    _stats.cur_mem_usage += mem->size;
    _stats.max_mem_usage = std::max(_stats.max_mem_usage, _stats.cur_mem_usage);

    _timer.increment(true);
}


void AppMetrics::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    _stats.cur_mem_usage -= it->second->size;
    active_memories.erase(it);

    _timer.increment(true);
}


void AppMetrics::gpu_data_analysis(void* data, size_t size) {
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)data;

    auto event = std::prev(kernel_events.end())->second;
    event->mem_accesses = tracker->accessCount;
}


void AppMetrics::flush() {
    const char* env_filename = std::getenv("APP_NAME");
    std::string filename;
    if (env_filename) {
        fprintf(stdout, "APP_NAME: %s\n", env_filename);
        filename = std::string(env_filename) + "_" + get_current_date_n_time();
    } else {
        filename = "metrics_" + get_current_date_n_time();
        fprintf(stdout, "No filename specified. Using default filename: %s\n",
                filename.c_str());
    }
    filename += ".log";
    printf("Dumping traces to %s\n", filename.c_str());

    std::ofstream out(filename);

    int count = 0;
    for (auto event : alloc_events) {
        out << "Alloc(" << event.second->alloc_type << ") " << count << ":\t"
            << event.second->addr << " " << event.second->size
            << " (" << format_size(event.second->size) << ")" << std::endl;
        count++;
    }
    out << std::endl;

    count = 0;
    for (auto event : kernel_events) {
        out << "Kernel " << count << " (refs=" << event.second->mem_accesses
            << "):\t" << event.second->kernel_name << std::endl;
        _stats.tot_mem_accesses += event.second->mem_accesses;
        if (_stats.max_mem_accesses_per_kernel < event.second->mem_accesses) {
            _stats.max_mem_accesses_kernel = event.second->kernel_name;
            _stats.max_mem_accesses_per_kernel = event.second->mem_accesses;
        }
        count++;
    }
    out << std::endl;

    // sort kernel_invocations by number of invocations in descending order
    std::vector<std::pair<std::string, uint32_t>> sorted_kernel_invocations(
                        kernel_invocations.begin(), kernel_invocations.end());
    std::sort(sorted_kernel_invocations.begin(), sorted_kernel_invocations.end(),
                                [](const std::pair<std::string, uint32_t>& a,
                                   const std::pair<std::string, uint32_t>& b) {
        return a.second > b.second;
    });
    for (auto kernel : sorted_kernel_invocations) {
        out << "InvCount=" << kernel.second << "\t" << kernel.first << std::endl;
    }
    out << std::endl;

    _stats.avg_mem_accesses = _stats.tot_mem_accesses / _stats.num_kernels;
    out << "Number of allocations: " << _stats.num_allocs << std::endl;
    out << "Number of kernels: " << _stats.num_kernels << std::endl;
    out << "Maximum memory usage: " << _stats.max_mem_usage
        << "B (" << format_size(_stats.max_mem_usage) << ")" << std::endl;
    out << "Maximum memory accesses kernel: " << _stats.max_mem_accesses_kernel
        << std::endl;
    out << "Maximum memory accesses per kernel: " << _stats.max_mem_accesses_per_kernel
        << " (" << format_number(_stats.max_mem_accesses_per_kernel) << ")" << std::endl;
    out << "Average memory accesses per kernel: " << _stats.avg_mem_accesses
        << " (" << format_number(_stats.avg_mem_accesses) << ")"  << std::endl;
    out << "Total memory accesses: " << _stats.tot_mem_accesses
        << " (" << format_number(_stats.tot_mem_accesses) << ")"  << std::endl;
    out.close();
}