#pragma once
#include "snapshot.h"
#include <atomic>
#include <array>
#include <cstdint>
#include <optional>

// Must be a power of 2
constexpr uint32_t RING_BUFFER_SIZE = 256;

// Thread-safe single-producer single-consumer ring buffer
// Hook thread pushes, TUI thread pops
class RingBuffer {
public:
    // Push a snapshot - called from hook thread
    // Returns false if something went wrong
    bool push(const LayerSnapshot& snap);

    // Pop next snapshot - called from TUI thread
    // Returns empty optional if buffer is empty
    std::optional<LayerSnapshot> pop();

    // How many snapshots are waiting to be read
    uint32_t available() const;

private:
    std::array<LayerSnapshot, RING_BUFFER_SIZE> buffer_;
    std::atomic<uint32_t> head_{0};   // where next write goes
    std::atomic<uint32_t> tail_{0};   // where next read comes from
};