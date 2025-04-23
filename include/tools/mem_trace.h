#ifndef YOSEMITE_TOOL_MEM_TRACE_H
#define YOSEMITE_TOOL_MEM_TRACE_H


#include "tools/tool.h"
#include "utils/event.h"

namespace yosemite {

class MemTrace final : public Tool {
public:
    MemTrace();

    ~MemTrace();

    void kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);

    void ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten);

    void ten_free_callback(std::shared_ptr<TenFree_t> ten);

    void gpu_data_analysis(void* data, uint64_t size);

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count) override {};

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count) override {};

    void evt_callback(EventPtr_t evt);

    void flush();

private:
    void kernel_trace_flush(std::shared_ptr<KernelLauch_t> kernel);
};

}   // yosemite
#endif // YOSEMITE_TOOL_MEM_TRACE_H
