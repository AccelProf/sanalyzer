#include "tools/mem_trace.h"
#include "utils/helper.h"
#include "utils/event.h"
#include "gpu_patch.h"

#include <cstdint>
#include <map>
#include <list>
#include <fstream>
#include <memory>
#include <cassert>
#include <iostream>


using namespace yosemite;


typedef struct TraceEntry
{
    uint64_t addr;
    uint64_t page_no;
    uint32_t size;
    uint64_t timestampe;
    uint32_t access_type; // 0: read 1: write
    MemoryAccessType mem_type;

    TraceEntry() = default;

    ~TraceEntry() = default;

    TraceEntry(uint64_t addr, uint32_t size, uint32_t access_type)
        : addr(addr), size(size), access_type(access_type) {
            timestampe = 0;
            page_no = addr >> 12;
    }
}TraceEntry_t;


static Timer_t _timer;

static bool first_kernel_finished = false;
std::string trace_folder_name;
uint32_t kernel_id = 0;

std::map<uint64_t, std::shared_ptr<KernelLauch_t>> kernel_events;
std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;

std::map<uint64_t, std::shared_ptr<TenAlloc>> tensor_events;
std::map<DevPtr, std::shared_ptr<TenAlloc>> active_tensors;

std::list<TraceEntry> _traces;


static std::string GetMemoryRWString(uint32_t flags)
{
    auto SANITIZER_MEMORY_DEVICE_FLAG_READ = 0x1;
    auto SANITIZER_MEMORY_DEVICE_FLAG_WRITE = 0x2;
    const bool isWrite = !!(flags & SANITIZER_MEMORY_DEVICE_FLAG_WRITE);
    const bool isRead = !!(flags & SANITIZER_MEMORY_DEVICE_FLAG_READ);

    if (isWrite && isRead) {return "Atomic";}
    else if (isRead) {return "Read";}
    else if (isWrite) {return "Write";}
    else {return "Unknown";}
}


static std::string GetMemoryTypeString(MemoryAccessType type)
{
    if (type == MemoryAccessType::Local) {return "local";}
    else if (type == MemoryAccessType::Shared) {return "shared";}
    else {return "global";}
}


MemTrace::MemTrace() : Tool(MEM_TRACE) {
    const char* torch_prof = std::getenv("YOSEMITE_TORCH_PROFILE");
    if (torch_prof && std::string(torch_prof) == "1") {
        fprintf(stdout, "Enabling torch profiler in MemTrace.\n");
        _torch_enabled = true;
    }
}


MemTrace::~MemTrace() {}


void MemTrace::kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel) {

    kernel->kernel_id = kernel_id++;
    kernel_events.emplace(_timer.get(), kernel);
    _traces = std::list<TraceEntry>();

    _timer.increment(true);
}


void MemTrace::kernel_trace_flush(std::shared_ptr<KernelLauch_t> kernel) {
    std::string filename = trace_folder_name + "/kernel_"
                            + std::to_string(kernel->kernel_id) + ".txt";
    printf("Dumping traces to %s\n", filename.c_str());

    std::ofstream out(filename);

    for (auto& trace : _traces) {
        out << trace.page_no << " "
            << trace.addr << " "
            << trace.size << " "
            << trace.timestampe << " "
            << trace.access_type << " "
            << (uint32_t) trace.mem_type
            << std::endl;
    }

    for (auto evt : active_memories) {
        out << "ALLOCATION: " << " " << evt.second->addr
            << " " << evt.second->size << std::endl;
    }

    for (auto evt : active_tensors) {
        out << "TENSOR: " << " " << evt.second->addr
            << " " << evt.second->size << std::endl;
    }

    out << "KERNEL: " << kernel->timestamp << " " << kernel->end_time << std::endl;

    out.close();
}


void MemTrace::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    if (!first_kernel_finished) {
        const char* env_trace_folder_name = std::getenv("APP_NAME");
        if (env_trace_folder_name != nullptr) {
            fprintf(stdout, "APP_NAME: %s\n", env_trace_folder_name);
            trace_folder_name = "traces_" + std::string(env_trace_folder_name)
                                + "_" + get_current_date_n_time();
        } else {
            fprintf(stdout, "No trace_folder_name specified.\n");
            trace_folder_name = "traces_" + get_current_date_n_time();
        }
        check_folder_existance(trace_folder_name);
        first_kernel_finished = true;
    }

    auto evt = std::prev(kernel_events.end())->second;
    evt->end_time = _timer.get();

    kernel_trace_flush(evt);

    _timer.increment(true);
}


void MemTrace::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);

    _timer.increment(true);
}


void MemTrace::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}


void MemTrace::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    tensor_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void MemTrace::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void MemTrace::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccess* accesses_buffer = (MemoryAccess*)data;
    for (int i = 0; i < size; i++) {
        auto& accesses = accesses_buffer[i];
        for (int j = 0; j < GPU_WARP_SIZE; j++) {
            if (accesses.addresses[j] != 0) {
                TraceEntry entry;
                entry.addr = accesses.addresses[j];
                entry.page_no = entry.addr >> 12;
                entry.size = accesses.accessSize;
                entry.timestampe = _timer.get();
                entry.access_type = accesses.flags;
                entry.mem_type = accesses.type;
                _traces.push_back(entry);
            }
        }
    }

}


void MemTrace::evt_callback(EventPtr_t evt) {
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


void MemTrace::flush() {
}
