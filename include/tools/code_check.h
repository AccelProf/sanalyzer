#ifndef YOSEMITE_TOOL_CODE_CHECK_H
#define YOSEMITE_TOOL_CODE_CHECK_H


#include "tools/tool.h"
#include "utils/event.h"

namespace yosemite {

class CodeCheck final : public Tool {
public:
    CodeCheck() : Tool(CODE_CHECK) {}

    ~CodeCheck() {}

    void kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);

    void mem_cpy_callback(std::shared_ptr<MemCpy_t> mem);

    void mem_set_callback(std::shared_ptr<MemSet_t> mem);

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, size_t size);

    void flush();
};

}   // yosemite
#endif // YOSEMITE_TOOL_CODE_CHECK_H