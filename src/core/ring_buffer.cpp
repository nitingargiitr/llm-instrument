#include "core/ring_buffer.h"
#include <cstring>
#include <new>

RingBuffer::RingBuffer() {
    buffer_ = new LayerSnapshot[RING_BUFFER_SIZE];
    memset(buffer_, 0, sizeof(LayerSnapshot) * RING_BUFFER_SIZE);
}

RingBuffer::~RingBuffer() {
    delete[] buffer_;
}

bool RingBuffer::push(const LayerSnapshot& snap) {
    uint32_t head = head_.load(std::memory_order_relaxed);
    uint32_t next_head = (head + 1) & (RING_BUFFER_SIZE - 1);

    if (next_head == tail_.load(std::memory_order_acquire)) {
        tail_.fetch_add(1, std::memory_order_release);
    }

    buffer_[head] = snap;
    head_.store(next_head, std::memory_order_release);
    return true;
}

std::optional<LayerSnapshot> RingBuffer::pop() {
    uint32_t tail = tail_.load(std::memory_order_relaxed);

    if (tail == head_.load(std::memory_order_acquire)) {
        return std::nullopt;
    }

    LayerSnapshot snap = buffer_[tail];
    tail_.store((tail + 1) & (RING_BUFFER_SIZE - 1),
                std::memory_order_release);
    return snap;
}

uint32_t RingBuffer::available() const {
    uint32_t h = head_.load(std::memory_order_acquire);
    uint32_t t = tail_.load(std::memory_order_acquire);
    return (h - t) & (RING_BUFFER_SIZE - 1);
}