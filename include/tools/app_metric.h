#ifndef YOSEMITE_TOOL_APP_METRICS_H
#define YOSEMITE_TOOL_APP_METRICS_H


#include "tools/tool.h"
#include "utils/event.h"

namespace yosemite {

class AppMetrics final : public Tool {
public:
    AppMetrics() : Tool(APP_METRICE) {}

    ~AppMetrics() {}

    void kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, uint64_t size);

    void flush();
};

}   // yosemite
#endif // YOSEMITE_TOOL_APP_METRICS_H