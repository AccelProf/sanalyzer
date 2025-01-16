#include "tools/hot_analysis.h"
#include <cstring>
#include "utils/helper.h"
#include "gpu_patch.h"

#include <map>
#include <vector>
#include <cassert>
#include <cstring>


using namespace yosemite;

constexpr uint32_t RANGE_GRANULARITY = 2 * 1024 * 1024;

static std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;


HotAnalysis::HotAnalysis() : Tool(HOT_ANALYSIS) {
}

HotAnalysis::~HotAnalysis() {
}

void HotAnalysis::kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel) {
}

void HotAnalysis::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
}

uint64_t device_size = 0;
void HotAnalysis::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    if (mem->alloc_type != 0) {
        return;
    }

    active_memories.emplace(mem->addr, mem);
}

void HotAnalysis::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    if (mem->alloc_type != 0) {
        return;
    }

    active_memories.erase(mem->addr);
}

void HotAnalysis::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
}

void HotAnalysis::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
}

void HotAnalysis::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
}

void HotAnalysis::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
}

void HotAnalysis::evt_callback(EventPtr_t evt) {
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

void HotAnalysis::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccessState* state = (MemoryAccessState*)data;
    for (uint32_t i = 0; i < size; ++i) {
        MemoryRange range = state->start_end[i];
        printf("range: %lu - %lu, count: %lu, size: %lu (%s)\n", range.start, range.end, state->touch[i], range.end - range.start, format_size(range.end - range.start).c_str());
    }
}

void HotAnalysis::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    std::vector<MemoryRange> active_memory_ranges;
    for (auto active_mem : active_memories) {
        auto mem = active_mem.second;
        if (mem->size <= RANGE_GRANULARITY) {
            MemoryRange range = {mem->addr, mem->addr + mem->size};
            active_memory_ranges.push_back(range);
        } else {
            uint64_t start = mem->addr;
            uint64_t end = mem->addr + mem->size;
            while (start < end) {
                if (start + RANGE_GRANULARITY > end) {
                    MemoryRange range = {start, end};
                    active_memory_ranges.push_back(range);
                } else {
                    MemoryRange range = {start, start + RANGE_GRANULARITY};
                    active_memory_ranges.push_back(range);
                }
                start += RANGE_GRANULARITY;
            }
        }
    }
    assert(active_memory_ranges.size() <= MAX_NUM_MEMORY_RANGES);

    MemoryRange* temp_ranges = active_memory_ranges.data();
    memcpy(ranges, temp_ranges, sizeof(MemoryRange) * active_memory_ranges.size());
    *count = active_memory_ranges.size();
}

void HotAnalysis::flush() {
}
