#ifndef YOSEMITE_UTILS_EVENT_H
#define YOSEMITE_UTILS_EVENT_H

#include <cstdint>
#include <string>
#include <memory>

typedef unsigned long long DevPtr;

typedef struct Timer{
    uint64_t access_timer;
    uint64_t event_timer;

    Timer() {
        access_timer = 0;
        event_timer = 0;
    }

    void increment(bool is_event) {
        if (is_event) {
            event_timer++;
        } else {
            access_timer++;
        }
    }

    uint64_t get() {
        return access_timer + event_timer;
    }
} Timer_t;

namespace yosemite {

typedef enum EventType {
    EventType_KERNEL_LAUNCH = 0,
    EventType_KERNEL_END = 1,
    EventType_MEM_ALLOC = 2,
    EventType_MEM_FREE = 3,
    EventType_MEM_COPY = 4,
    EventType_MEM_SET = 5,
    EventType_TEN_ALLOC = 6,
    EventType_TEN_FREE = 7,
    EventTypeCount = 8,
}EventType_t;


typedef struct Event
{
    uint64_t timestamp;
    EventType_t evt_type;

    Event() = default;
    virtual ~Event() = default;

    bool operator<(const Event &other) const { return timestamp < other.timestamp; }
}Event_t;

typedef std::shared_ptr<Event> EventPtr_t;


typedef struct KernelLauch : public Event {
    uint64_t end_time;
    std::string kernel_name;
    uint32_t kernel_id;
    uint64_t mem_accesses;

    KernelLauch() {
        evt_type = EventType_KERNEL_LAUNCH;
    }

    KernelLauch(std::string kernel_name)
        : kernel_name(kernel_name) {
            evt_type = EventType_KERNEL_LAUNCH;
        }

    ~KernelLauch() = default;
}KernelLauch_t;

typedef struct KernelEnd : public Event {
    uint64_t end_time;
    std::string kernel_name;
    uint64_t mem_accesses;

    KernelEnd() {
        evt_type = EventType_KERNEL_END;
    }

    KernelEnd(std::string kernel_name)
        : kernel_name(kernel_name) {
            evt_type = EventType_KERNEL_END;
        }

    ~KernelEnd() = default;
}KernelEnd_t;

typedef struct MemAlloc : public Event {
    DevPtr addr;
    uint64_t size;
    uint64_t release_time;
    int alloc_type;

    MemAlloc() {
        evt_type = EventType_MEM_ALLOC;
    }

    MemAlloc(DevPtr addr, uint64_t size, int alloc_type)
        : addr(addr), size(size), alloc_type(alloc_type) {
            evt_type = EventType_MEM_ALLOC;
        }

    ~MemAlloc() = default;
}MemAlloc_t;

typedef struct MemFree : public Event {
    DevPtr addr;
    uint64_t size;

    MemFree() {
        evt_type = EventType_MEM_FREE;
    }

    MemFree(DevPtr addr)
        : addr(addr) {
            evt_type = EventType_MEM_FREE;
        }

    ~MemFree() = default;
}MemFree_t;

typedef struct MemCpy : public Event {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t size;

    MemCpy() {
        evt_type = EventType_MEM_COPY;
    }
    
    MemCpy(uint64_t src_addr, uint64_t dst_addr, uint64_t size)
        : src_addr(src_addr), dst_addr(dst_addr), size(size) {
            evt_type = EventType_MEM_COPY;
        }

    ~MemCpy() = default;
}MemCpy_t;

typedef struct MemSet : public Event {

    uint64_t addr;
    uint64_t size;
    uint8_t value;

    MemSet() {
        evt_type = EventType_MEM_SET;
    }

    MemSet(uint64_t addr, uint64_t size, uint8_t value)
        : addr(addr), size(size), value(value) {
            evt_type = EventType_MEM_SET;
        }

    ~MemSet() = default;
}MemSet_t;

typedef struct TenAlloc : public Event {
    DevPtr addr;
    uint64_t size;
    uint64_t release_time;
    int alloc_type;

    TenAlloc() {
        evt_type = EventType_TEN_ALLOC;
    }

    TenAlloc(DevPtr addr, uint64_t size)
        : addr(addr), size(size) {
            evt_type = EventType_TEN_ALLOC;
        }

    ~TenAlloc() = default;
}TenAlloc_t;

typedef struct TenFree : public Event {
    DevPtr addr;
    uint64_t size;

    TenFree() {
        evt_type = EventType_TEN_FREE;
    }

    TenFree(DevPtr addr)
        : addr(addr) {
            evt_type = EventType_TEN_FREE;
        }

    ~TenFree() = default;
}TenFree_t;

}   // yosemite

#endif // YOSEMITE_UTILS_EVENT_H