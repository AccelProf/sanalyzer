#ifndef YOSEMITE_H
#define YOSEMITE_H

#include <cstddef>
#include <cstdint>
#include <string>

typedef enum {
    YOSEMITE_SUCCESS = 0,
    YOSEMITE_ERROR = 1,
    YOSEMITE_NOT_IMPLEMENTED = 2,
    YOSEMITE_CUDA_MEMFREE_ZERO = 3
} YosemiteResult_t;


typedef struct SanitizerOptions {
    bool enable_access_tracking;

    SanitizerOptions() = default;
    ~SanitizerOptions() = default;
} SanitizerOptions_t;


YosemiteResult_t yosemite_alloc_callback(uint64_t ptr, size_t size, int type);

YosemiteResult_t yosemite_free_callback(uint64_t ptr);

YosemiteResult_t yosemite_memcpy_callback();

YosemiteResult_t yosemite_memset_callback();

YosemiteResult_t yosemite_kernel_start_callback(std::string kernel_name);

YosemiteResult_t yosemite_kernel_end_callback(std::string kernel_name);

YosemiteResult_t yosemite_gpu_data_analysis(void* data, size_t size);

YosemiteResult_t yosemite_init(SanitizerOptions_t& options);

YosemiteResult_t yosemite_terminate();

YosemiteResult_t yosemite_tensor_malloc_callback(uint64_t ptr, int64_t alloc_size,
                                int64_t total_allocated, int64_t total_reserved);

YosemiteResult_t yosemite_tensor_free_callback(uint64_t ptr, int64_t alloc_size,
                                int64_t total_allocated, int64_t total_reserved);


#endif // YOSEMITE_H
