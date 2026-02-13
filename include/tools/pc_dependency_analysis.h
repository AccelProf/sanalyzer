#ifndef YOSEMITE_TOOL_PC_DEPENDENCY_ANALYSIS_H
#define YOSEMITE_TOOL_PC_DEPENDENCY_ANALYSIS_H


#include "tools/tool.h"
#include "utils/event.h"
#include "gpu_patch.h"

#include <map>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <array>
#include <cstdint>
#include <string>
#include <memory>
#include <cassert>


#ifndef SANITIZER_MEMORY_DEVICE_FLAG_READ
#define SANITIZER_MEMORY_DEVICE_FLAG_READ 0x1
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_WRITE
#define SANITIZER_MEMORY_DEVICE_FLAG_WRITE 0x2
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_RED
#define SANITIZER_MEMORY_DEVICE_FLAG_RED 0x3
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_ATOMIC
#define SANITIZER_MEMORY_DEVICE_FLAG_ATOMIC 0x4
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_PREFETCH
#define SANITIZER_MEMORY_DEVICE_FLAG_PREFETCH 0x8
#endif

#ifndef SANITIZER_MEMORY_GLOBAL
#define SANITIZER_MEMORY_GLOBAL 0x10
#endif

#ifndef SANITIZER_MEMORY_SHARED
#define SANITIZER_MEMORY_SHARED 0x20
#endif

#ifndef SANITIZER_MEMORY_LOCAL
#define SANITIZER_MEMORY_LOCAL 0x40
#endif

namespace yosemite {

/* we choose to use PC offset instead of PC because the PC is too long for shadow memory and it is not necessary to track the original PC.
The offset will be calculated during trace collection.

Every memory allocation will cause a shadow memory to be created.
Every memory deallocation will cause a shadow memory to be destroyed.
Shadow memory bitmask will be reset when a kernel finished. (to avoid mass shadow memory reset)

The gpu data analysis will 
1.iterate the trace buffer and query the shadow memory to get the corresponding shadow memory entry.
2. compare the last access information with the current access information with the rules below:
    0. if bitmask of this access is 0, it means the current access is a cold miss set it's acient pc to 0xFFFFFFFF.
    1. if last access and current access are from the same thread, then it is an intra thread access.
    2. if last access and current access are from the same warp, then it is an intra warp access.
    3. if last access and current access are from the same block, then it is an intra block access.
    4. if last access and current access are from the same grid, then it is an intra grid access.
3. update the pc_statistics with the current pc, ancient pc and the distance.
4. update the shadow memory entry with the current pc and the flat thread id.
*/


class memory_region{
public:
    memory_region() : start(0), end(0) {};
    memory_region(uint64_t start, uint64_t end) : start(start), end(end) {};
    ~memory_region() {};

    bool contains(uint64_t ptr) const {
        return ptr >= start && ptr < end;
    };

    bool operator==(const memory_region& other) const {
        return start == other.start && end == other.end;
    };

    bool operator<(const memory_region& other) const {
        // strict-weak-ordering: compare both start and end
        if (start != other.start) return start < other.start;
        return end < other.end;
    };

    uint64_t get_start() const {
        return start;
    };
    uint64_t get_end() const {
        return end;
    };

private:
    uint64_t start;
    uint64_t end;
};

class shadow_memory_entry{
public:
    shadow_memory_entry() {};
    ~shadow_memory_entry() {};

    uint32_t last_pc = 0xFFFFFFFFu; // using offset of pc instead of original pc to save space and keep alignment;
    uint32_t last_flat_thread_id = 0xFFFFFFFFu; // 0-5 bits for lane id, 6-10 bits for warp id, 11-31 bits for block id to save space;
};

class shadow_memory{
public:
    shadow_memory(uint64_t size) 
    :_size(size),
    _size_celled((size + 3) / 4 * 4),
    _stride(_size_celled / 4),
    _shadow_memory_entries(std::make_unique<shadow_memory_entry[]>(_size_celled)), 
      _shadow_memory_bitmap(std::vector<uint8_t>((size + 7) / 8, 0)) {
        printf("[PC_DEPENDENCY] Shadow memory entries: %lu\n", size);
        printf("[PC_DEPENDENCY] Shadow memory per entry size: %lu\n", sizeof(shadow_memory_entry));
        printf("[PC_DEPENDENCY] Shadow memory size: %lu\n", size*sizeof(shadow_memory_entry));
        printf("[PC_DEPENDENCY] Shadow memory bitmap size: %lu\n", _shadow_memory_bitmap.size());
      };
    ~shadow_memory() = default;
    void reset_bitmap() {
        std::fill(_shadow_memory_bitmap.begin(), _shadow_memory_bitmap.end(), 0);
    };
    shadow_memory_entry& get_entry(uint64_t offset) {
        assert(offset < _size);
        //update layout: use offset/4 + offset%4 * _size/4 to make every 4 bytes adjacent in one cache line
        return _shadow_memory_entries[(offset/4) + (offset%4) * _stride];
        // return _shadow_memory_entries[offset];
    }
    bool is_valid(uint64_t ptr) {
        return _shadow_memory_bitmap[ptr / 8] & (1u << (ptr % 8)); 
    }
    void set_valid(uint64_t ptr) {
        _shadow_memory_bitmap[ptr / 8] |= (1u << (ptr % 8));
    }
    uint64_t _size;
    uint64_t _size_celled;
    uint64_t _stride;
    std::unique_ptr<shadow_memory_entry[]> _shadow_memory_entries;
    std::vector<uint8_t> _shadow_memory_bitmap;
};


class PC_statisitics{
public:
    std::array<uint64_t, 4> dist = {0, 0, 0, 0}; 
    // 0: intra thread
    // 1: intra warp
    // 2: intra block
    // 3: intra grid
};

class PcDependency final : public Tool {
public:
    PcDependency();

    ~PcDependency();

    void gpu_data_analysis(void* data, uint64_t size);

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count) override {};

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count) override {};

    void allocation_callback(uint64_t ptr, uint64_t size);

    void deallocation_callback(uint64_t ptr);

    void evt_callback(EventPtr_t evt);

    void flush();

private:
    void kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);

    void ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten);

    void ten_free_callback(std::shared_ptr<TenFree_t> ten);

    void kernel_trace_flush(std::shared_ptr<KernelLaunch_t> kernel);

    void unit_access(uint64_t ptr, uint32_t pc_offset, uint64_t current_block_id, uint32_t current_warp_id, uint32_t current_lane_id, memory_region& memory_region_target, int access_size);

    void unit_access_shared(uint64_t ptr, uint32_t pc_offset, uint64_t current_block_id, uint32_t current_warp_id, uint32_t current_lane_id, int access_size);

    void unit_access_local(uint64_t ptr, uint32_t pc_offset, uint64_t current_block_id, uint32_t current_warp_id, uint32_t current_lane_id, int access_size);


/*
********************************* variables *********************************
*/
    Timer_t _timer;

    std::string output_directory;
    uint32_t kernel_id = 0;


    std::map<uint64_t, std::shared_ptr<KernelLaunch_t>> kernel_events;
    std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;

    std::map<uint64_t, std::shared_ptr<TenAlloc>> tensor_events;
    std::map<DevPtr, std::shared_ptr<TenAlloc>> active_tensors;


    std::vector<memory_region> _memory_regions;

    std::map<memory_region, std::unique_ptr<shadow_memory>> _shadow_memories; // memory region, shadow memory
    std::unordered_map<uint64_t, shadow_memory_entry> _shadow_memory_shared; // shared memory address (packed as block_id << 32 | address low 32 bits to reduce aliasing), shadow memory shared
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, PC_statisitics>> _pc_statistics; // current pc offset, ancient pc offset, PC_statisitics
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> _pc_flags; // pc offset, flags, size of the access
    // Index [0..31] stores distinct sector count 1..32.
    // Index [32..64] stores active lane count 0..32.
    std::unordered_map<uint32_t, std::array<uint64_t, 65>> _distinct_sector_count; // pc offset, distinct sector distribution

};

}   // yosemite
#endif // YOSEMITE_TOOL_PC_DEPENDENCY_ANALYSIS_H
