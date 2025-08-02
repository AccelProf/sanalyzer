#ifndef YOSEMITE_TOOL_EVENT_TRACE_H
#define YOSEMITE_TOOL_EVENT_TRACE_H

#include "tools/tool.h"
#include "utils/event.h"
#include <map>

namespace yosemite {

class EventTrace final : public Tool {
    public:
    EventTrace();

    ~EventTrace();

    void gpu_data_analysis(void* data, uint64_t size) override {};

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count) override {};

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count) override {};

    void evt_callback(EventPtr_t evt);

    void flush();

private:
    void init();

    void kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);

    void mem_cpy_callback(std::shared_ptr<MemCpy_t> mem);

    void mem_set_callback(std::shared_ptr<MemSet_t> mem);


/*
********************************* variables *********************************
*/
    Timer_t _timer;

    std::map<uint64_t, std::shared_ptr<KernelLauch_t>> kernel_events;
    std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;

    std::map<uint64_t, std::shared_ptr<TenAlloc>> tensor_events;
    std::map<DevPtr, std::shared_ptr<TenAlloc>> active_tensors;

};

} // namespace yosemite
#endif // YOSEMITE_TOOL_EVENT_TRACE_H
