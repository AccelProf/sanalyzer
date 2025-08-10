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

void EventTrace::evt_callback(EventPtr_t evt) {
    switch (evt->evt_type) {
        case EventType_KERNEL_LAUNCH:
            kernel_start_callback(std::dynamic_pointer_cast<KernelLaunch_t>(evt));
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

void EventTrace::flush() {}

void EventTrace::init() {}

void EventTrace::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
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

void EventTrace::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    PRINT("[YOSEMITE INFO] Ten alloc: %lu, size: %lu\n", ten->addr, ten->size);
    _timer.increment(true);
}

void EventTrace::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    PRINT("[YOSEMITE INFO] Ten free: %lu, size: %lu\n", ten->addr, ten->size);
    _timer.increment(true);
}

void EventTrace::op_start_callback(std::shared_ptr<OpStart_t> op) {
    PRINT("[YOSEMITE INFO] Op start: %s\n", op->op_name.c_str());
    _timer.increment(true);
}

void EventTrace::op_end_callback(std::shared_ptr<OpEnd_t> op) {
    PRINT("[YOSEMITE INFO] Op end: %s\n", op->op_name.c_str());
    _timer.increment(true);
}
