#include "tools/pc_dependency_analysis.h"
#include "utils/helper.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <cassert>
#include <iostream>
#include <sstream>
#include <set>
#include <iomanip>


using namespace yosemite;

namespace {
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                // control chars
                if (static_cast<unsigned char>(c) < 0x20) {
                    std::ostringstream oss;
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << (int)static_cast<unsigned char>(c);
                    out += oss.str();
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static std::string hex_u32(uint32_t v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << v;
    return oss.str();
}
} // namespace


PcDependency::PcDependency() : Tool(PC_DEPENDENCY_ANALYSIS) {
    const char* torch_prof = std::getenv("TORCH_PROFILE_ENABLED");
    if (torch_prof && std::string(torch_prof) == "1") {
        fprintf(stdout, "Enabling torch profiler in PcDependency.\n");
        _torch_enabled = true;
    }

    const char* env_app_name = std::getenv("YOSEMITE_APP_NAME");
    if (env_app_name != nullptr) {
        output_directory = "dependency_" + std::string(env_app_name)
                            + "_" + get_current_date_n_time();
    } else {
        output_directory = "dependency_" + get_current_date_n_time();
    }
    check_folder_existance(output_directory);
}


PcDependency::~PcDependency() {}


void PcDependency::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {

    kernel->kernel_id = kernel_id++;
    kernel_events.emplace(_timer.get(), kernel);
    _pc_statistics.clear();
    _pc_flags.clear();
    for (auto& shadow_memory_iter : _shadow_memories) {
        shadow_memory_iter.second->reset_bitmap();
    }
    printf("[PC_DEPENDENCY] Resetting shadow memory bitmap\n");
    _timer.increment(true);
}


void PcDependency::kernel_trace_flush(std::shared_ptr<KernelLaunch_t> kernel) {
    std::string filename = output_directory + "/kernel_"
                            + std::to_string(kernel->kernel_id) + ".csv";
    printf("Dumping pc dependency to %s\n", filename.c_str());

    std::ofstream out(filename);
    out << "current_pc_offset,ancient_pc_offset,flags,intra_thread,intra_warp,intra_block,intra_grid\n";

    std::vector<std::pair<uint32_t, std::unordered_map<uint32_t, PC_statisitics>>> outer(
        _pc_statistics.begin(), _pc_statistics.end());
    std::sort(outer.begin(), outer.end(),
              [](auto& a, auto& b){ return a.first < b.first; });

    for (auto& [cur_pc, inner_map] : outer) {
        std::vector<std::pair<uint32_t, PC_statisitics>> inner(inner_map.begin(), inner_map.end());
        std::sort(inner.begin(), inner.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });

        uint32_t flags = 0;
        auto fit = _pc_flags.find(cur_pc);
        if (fit != _pc_flags.end()) flags = fit->second;

        for (auto& [anc_pc, st] : inner) {
            out << "0x" << std::hex << cur_pc
                << ",0x" << anc_pc
                << ",0x" << flags
                << std::dec
                << "," << st.dist[0]
                << "," << st.dist[1]
                << "," << st.dist[2]
                << "," << st.dist[3]
                << "\n";
        }
    }

    // JSON output for building PC dependency graph (joinable with CFG)
    std::string json_filename = output_directory + "/kernel_"
                                + std::to_string(kernel->kernel_id) + ".json";
    std::ofstream jout(json_filename);
    jout << "{\n";
    jout << "  \"tool\": \"pc_dependency_analysis\",\n";
    jout << "  \"kernel\": {\n";
    jout << "    \"kernel_id\": " << kernel->kernel_id << ",\n";
    jout << "    \"kernel_name\": \"" << json_escape(kernel->kernel_name) << "\",\n";
    jout << "    \"device_id\": " << kernel->device_id << ",\n";
    jout << "    \"kernel_pc\": " << kernel->kernel_pc << ",\n";
    jout << "    \"kernel_pc_hex\": \"" << hex_u32((uint32_t)kernel->kernel_pc) << "\"\n";
    jout << "  },\n";
    jout << "  \"shadow_memory_granularity_bytes\": 1,\n";
    jout << "  \"sample_stride_bytes\": 4,\n";

    // Collect nodes (all current PCs + all non-cold ancient PCs)
    std::set<uint32_t> nodes;
    for (const auto& [cur_pc, inner_map] : _pc_statistics) {
        nodes.insert(cur_pc);
        for (const auto& [anc_pc, st] : inner_map) {
            (void)st;
            if (anc_pc != 0xFFFFFFFFu) nodes.insert(anc_pc);
        }
    }

    jout << "  \"nodes\": [\n";
    {
        bool first = true;
        for (uint32_t pc : nodes) {
            if (!first) jout << ",\n";
            first = false;
            auto fit = _pc_flags.find(pc);
            bool has_flags = (fit != _pc_flags.end());
            uint32_t flags = has_flags ? fit->second : 0;
            jout << "    {\"pc\": " << pc
                 << ", \"pc_hex\": \"" << hex_u32(pc) << "\"";
            if (has_flags) {
                jout << ", \"flags\": " << flags
                     << ", \"flags_hex\": \"" << hex_u32(flags) << "\"";
            } else {
                jout << ", \"flags\": null, \"flags_hex\": null";
            }
            jout << "}";
        }
        jout << "\n";
    }
    jout << "  ],\n";

    // Edges: ancient_pc -> current_pc, with per-scope counts.
    jout << "  \"edges\": [\n";
    {
        // Stable order: sort by current pc then ancient pc
        std::vector<std::pair<uint32_t, std::unordered_map<uint32_t, PC_statisitics>>> outer2(
            _pc_statistics.begin(), _pc_statistics.end());
        std::sort(outer2.begin(), outer2.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });

        bool first_edge = true;
        for (auto& [cur_pc, inner_map] : outer2) {
            std::vector<std::pair<uint32_t, PC_statisitics>> inner2(inner_map.begin(), inner_map.end());
            std::sort(inner2.begin(), inner2.end(),
                      [](auto& a, auto& b){ return a.first < b.first; });

            // current flags if available
            auto cfit = _pc_flags.find(cur_pc);
            bool has_cflags = (cfit != _pc_flags.end());
            uint32_t cflags = has_cflags ? cfit->second : 0;

            for (auto& [anc_pc, st] : inner2) {
                if (!first_edge) jout << ",\n";
                first_edge = false;

                bool cold_miss = (anc_pc == 0xFFFFFFFFu);

                jout << "    {\"current_pc\": " << cur_pc
                     << ", \"current_pc_hex\": \"" << hex_u32(cur_pc) << "\""
                     << ", \"ancient_pc\": ";
                if (cold_miss) {
                    jout << "null";
                } else {
                    jout << anc_pc;
                }
                jout << ", \"ancient_pc_hex\": ";
                if (cold_miss) {
                    jout << "null";
                } else {
                    jout << "\"" << hex_u32(anc_pc) << "\"";
                }
                jout << ", \"cold_miss\": " << (cold_miss ? "true" : "false");

                if (has_cflags) {
                    jout << ", \"current_flags\": " << cflags
                         << ", \"current_flags_hex\": \"" << hex_u32(cflags) << "\"";
                } else {
                    jout << ", \"current_flags\": null, \"current_flags_hex\": null";
                }

                jout << ", \"dist\": {"
                     << "\"intra_thread\": " << st.dist[0]
                     << ", \"intra_warp\": " << st.dist[1]
                     << ", \"intra_block\": " << st.dist[2]
                     << ", \"intra_grid\": " << st.dist[3]
                     << "}}";
            }
        }
        jout << "\n";
    }
    jout << "  ]\n";
    jout << "}\n";
    printf("Dumping pc dependency graph json to %s\n", json_filename.c_str());
}


void PcDependency::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    auto evt = std::prev(kernel_events.end())->second;
    evt->end_time = _timer.get();

    kernel_trace_flush(evt);

    _timer.increment(true);
}


void PcDependency::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    // TODO： add shadow memory allocation here
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);
    memory_region memory_region_current = memory_region((uint64_t)mem->addr, (uint64_t)(mem->addr + mem->size));
    _memory_regions.push_back(memory_region_current);
    _shadow_memories.emplace(memory_region_current, std::make_unique<shadow_memory>(mem->size));

    printf("[PC_DEPENDENCY] Allocating shadow memory for memory region: %p - %p, size: %lu\n", (void*)memory_region_current.get_start(), (void*)memory_region_current.get_end(), mem->size);
    _timer.increment(true);
}

void PcDependency::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());

    uint64_t sz = it->second->size;   // 从 alloc 事件拿 size
    active_memories.erase(it);

    memory_region r((uint64_t)mem->addr, (uint64_t)mem->addr + sz);

    auto vit = std::find(_memory_regions.begin(), _memory_regions.end(), r);
    if (vit != _memory_regions.end()) _memory_regions.erase(vit);

    _shadow_memories.erase(r);
    printf("[PC_DEPENDENCY] Freeing shadow memory for memory region: %p - %p, size: %lu\n", (void*)r.get_start(), (void*)r.get_end(), sz);
    _timer.increment(true);
}


void PcDependency::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    tensor_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void PcDependency::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}

void PcDependency::unit_access(uint64_t ptr, uint32_t pc_offset, uint64_t current_block_id, uint64_t current_warp_id, uint64_t current_lane_id, memory_region& memory_region_target, int access_size) {
    // auto& shadow_memory = this->_shadow_memories[memory_region_target];
    auto shadow_memory_it = this->_shadow_memories.find(memory_region_target);
    if (shadow_memory_it == this->_shadow_memories.end()) {
        printf("shadow memory not found for memory region: %lu - %lu\n", memory_region_target.get_start(), memory_region_target.get_end());
        return;
    }
    auto& shadow_memory = *(shadow_memory_it->second);

    for (int i = 0; i < access_size; i += 4) {
        auto addr = ptr + i;
        // Byte-granularity shadow memory: addr is byte offset within allocation.
        // Bound check to avoid OOB on allocations at end boundary or odd sizes.
        if (addr >= shadow_memory._size) {
            break;
        }
        if (shadow_memory.is_valid(addr) == false) {
            // cold miss
            _pc_statistics[pc_offset][0xFFFFFFFF].dist[0] += 1;
            shadow_memory.set_valid(addr);
            auto& shadow_memory_entry = shadow_memory.get_entry(addr);
            shadow_memory_entry.last_pc = pc_offset;
            shadow_memory_entry.last_flat_thread_id = (current_block_id << 10) | (current_warp_id << 5) | current_lane_id;
            continue;
        }
        auto& last_access = shadow_memory.get_entry(addr);
        uint64_t last_block_id = last_access.last_flat_thread_id >> 10;
        uint64_t last_warp_id = (last_access.last_flat_thread_id >> 5) & 0x1F;
        uint64_t last_lane_id = last_access.last_flat_thread_id & 0x1F;

        uint32_t last_pc = last_access.last_pc;
        if (last_block_id != current_block_id) {
            this->_pc_statistics[pc_offset][last_pc].dist[3] += 1;
        }else if (last_warp_id != current_warp_id) {
            this->_pc_statistics[pc_offset][last_pc].dist[2] += 1;
        }else if (last_lane_id != current_lane_id) {
            this->_pc_statistics[pc_offset][last_pc].dist[1] += 1;
        }else {
            this->_pc_statistics[pc_offset][last_pc].dist[0] += 1;
        }
        last_access.last_pc = pc_offset;
        last_access.last_flat_thread_id = (current_block_id << 10) | (current_warp_id << 5) | current_lane_id;
    }
}


void PcDependency::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccess* accesses_buffer = (MemoryAccess*)data;
    for (uint64_t i = 0; i < size; i++) {
        MemoryAccess trace = accesses_buffer[i];
        uint32_t pc_offset = trace.pc;
        this->_pc_flags[pc_offset] = trace.flags;
        if (trace.type != MemoryType::Global) {
            //only analyze global memory accesses currently
            continue;
        }
        uint32_t access_size = trace.accessSize;
        memory_region memory_region_target;
        uint64_t first_valid_address = 0;

        for (int j = 0; j < GPU_WARP_SIZE; j++) {
            if (trace.active_mask & (1u << j)) {
                first_valid_address = trace.addresses[j];
                break;
            }
        }
        
        
        assert(first_valid_address != 0);
        for (auto memory_region_iter : this->_memory_regions) {
            if (memory_region_iter.contains(first_valid_address)) {
                memory_region_target = memory_region_iter;
                break;
            }
        }
        uint64_t memory_region_start = memory_region_target.get_start();
        assert(memory_region_start != 0);
        for ( int j = 0; j < GPU_WARP_SIZE; j++) {
            if (trace.active_mask & (1u << j)) {
                unit_access(trace.addresses[j] - memory_region_start, pc_offset, trace.ctaId, trace.warpId, j, memory_region_target, access_size);
            }
        }
    }

}


void PcDependency::evt_callback(EventPtr_t evt) {
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


void PcDependency::flush() {
}
