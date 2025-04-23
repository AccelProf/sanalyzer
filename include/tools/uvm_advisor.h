#ifndef YOSEMITE_TOOL_UVM_ADVISOR_H
#define YOSEMITE_TOOL_UVM_ADVISOR_H


#include "tools/tool.h"
#include "utils/event.h"

#include <map>
namespace yosemite {

class UVMAdvisor final : public Tool {
public:
    UVMAdvisor();

    ~UVMAdvisor();

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, uint64_t size);

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count);

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count);

    void flush();

private :
    void init();

    void kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);

    void mem_cpy_callback(std::shared_ptr<MemCpy_t> mem);

    void mem_set_callback(std::shared_ptr<MemSet_t> mem);

    void ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten);

    void ten_free_callback(std::shared_ptr<TenFree_t> ten);

    void op_start_callback(std::shared_ptr<OpStart_t> op);

    void op_end_callback(std::shared_ptr<OpEnd_t> op);


/*
********************************* variables
*/
    FILE* out_fp;

    Timer_t _timer;

    std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;

    std::map<uint64_t, std::shared_ptr<TenAlloc_t>> tenalloc_events;
    std::map<DevPtr, std::shared_ptr<TenAlloc_t>> active_tensors;

    std::map<uint64_t, std::shared_ptr<KernelLauch_t>> kernel_events;
};  

}   // yosemite
#endif // YOSEMITE_TOOL_UVM_ADVISOR_H
