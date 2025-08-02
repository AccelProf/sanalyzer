#include "tools/event_trace.h"

using namespace yosemite;

#define YOSEMITE_VERBOSE 1

#if YOSEMITE_VERBOSE
#define PRINT(...) do { fprintf(stdout, __VA_ARGS__); fflush(stdout); } while (0)
#else
#define PRINT(...)
#endif

EventTrace::EventTrace() : Tool(EVENT_TRACE) {}

EventTrace::~EventTrace() {}

void EventTrace::evt_callback(EventPtr_t evt) {}

void EventTrace::flush() {}

void EventTrace::init() {}

void EventTrace::kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel) {
    PRINT("[YOSEMITE INFO] Kernel start: %s\n", kernel->kernel_name.c_str());
    _timer.increment(true);
}

void EventTrace::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    PRINT("[YOSEMITE INFO] Kernel end: %s\n", kernel->kernel_name.c_str());
    _timer.increment(true);
}

void EventTrace::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    PRINT("[YOSEMITE INFO] Mem alloc: %lu, size: %lu\n", mem->addr, mem->size);
    _timer.increment(true);
}

void EventTrace::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    PRINT("[YOSEMITE INFO] Mem free: %lu, size: %lu\n", mem->addr, mem->size);
    _timer.increment(true);
}

void EventTrace::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
    PRINT("[YOSEMITE INFO] Mem copy: %lu -> %lu, size: %lu, direction: %d\n", mem->src_addr, mem->dst_addr, mem->size, mem->direction);
    _timer.increment(true);
}

void EventTrace::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
    PRINT("[YOSEMITE INFO] Mem set: %lu, size: %lu, value: %d\n", mem->addr, mem->size, mem->value);
    _timer.increment(true);
}
