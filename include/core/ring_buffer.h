#pragma once
#include "snapshot.h"
#include <atomic>
#include <memory>
#include <optional>
#include <cstdint>

constexpr uint32_t RING_BUFFER_SIZE = 256;

class RingBuffer {
public:
    RingBuffer();
    ~RingBuffer();

    bool push(const LayerSnapshot& snap);
    std::optional<LayerSnapshot> pop();
    uint32_t available() const;
    uint64_t dropped_count() const;

private:
    LayerSnapshot*            buffer_;
    std::atomic<uint32_t>     head_{0};
    std::atomic<uint32_t>     tail_{0};
    std::atomic<uint64_t>     drops_{0};
};