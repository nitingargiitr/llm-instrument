#pragma once

#include "core/ring_buffer.h"
#include "core/snapshot.h"

class HookManager {
public:
    explicit HookManager(RingBuffer& buffer);

    void install();
    void uninstall();

    void capture_layer(
        int layer_index,
        LayerType type,
        ComputeDevice device,
        const float* data,
        int count,
        float latency_ms
    );

private:
    RingBuffer& buffer_;
    bool installed_ = false;
};