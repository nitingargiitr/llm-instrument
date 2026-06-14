#include "core/ring_buffer.h"
#include "hooks/hook_manager.h"
#include "core/snapshot.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    std::cout << "Starting hook manager tests...\n";

    RingBuffer ring_buffer;
    HookManager hooks(ring_buffer);
    hooks.install();

    float dense_data[128];
    for (int i = 0; i < 128; i++) {
        dense_data[i] = std::sin(i * 0.3f) * 4.0f;
    }

    float hot_data[128];
    for (int i = 0; i < 128; i++) {
        hot_data[i] = 7.5f;
    }

    float sparse_a[128];
    float sparse_b[128];
    for (int i = 0; i < 128; i++) {
        sparse_a[i] = (i % 2 == 0) ? 0.0f : 1.0f;  // ~50% sparse
        sparse_b[i] = 1.0f;                         // 0% sparse
    }

    // Capture a few normal layers
    for (int layer = 0; layer < 4; layer++) {
        LayerType type;
        switch (layer % 4) {
            case 0: type = LayerType::Embedding; break;
            case 1: type = LayerType::Attention; break;
            case 2: type = LayerType::MLP; break;
            default: type = LayerType::LayerNorm; break;
        }
        ComputeDevice dev = (layer % 4 == 3)
            ? ComputeDevice::CPU
            : ComputeDevice::CUDA;
        hooks.capture_layer(layer, type, dev, dense_data, 128,
                            0.8f + layer * 0.05f);
    }

    // High activation capture: baseline then spike
    hooks.capture_layer(10, LayerType::MLP, ComputeDevice::CUDA,
                        hot_data, 128, 1.2f);
    float hotter_data[128];
    for (int i = 0; i < 128; i++) hotter_data[i] = 40.0f;
    hooks.capture_layer(10, LayerType::MLP, ComputeDevice::CUDA,
                        hotter_data, 128, 1.3f);

    // Sparsity drop on same layer index
    hooks.capture_layer(20, LayerType::Attention, ComputeDevice::CUDA,
                        sparse_a, 128, 1.0f);
    hooks.capture_layer(20, LayerType::Attention, ComputeDevice::CUDA,
                        sparse_b, 128, 1.1f);

    assert(ring_buffer.available() == 8);
    std::cout << "PASS: captures pushed to ring buffer\n";

    bool saw_cuda_fallback = false;
    bool saw_high_act = false;
    bool saw_sparsity_drop = false;

    while (auto snap = ring_buffer.pop()) {
        if (snap->anomaly_flags & LayerSnapshot::FLAG_CUDA_FALLBACK)
            saw_cuda_fallback = true;
        if (snap->anomaly_flags & LayerSnapshot::FLAG_HIGH_ACTIVATION)
            saw_high_act = true;
        if (snap->anomaly_flags & LayerSnapshot::FLAG_SPARSITY_DROP)
            saw_sparsity_drop = true;
    }

    assert(saw_cuda_fallback);
    std::cout << "PASS: CUDA fallback detected\n";

    assert(saw_high_act);
    std::cout << "PASS: high activation detected\n";

    assert(saw_sparsity_drop);
    std::cout << "PASS: sparsity drop detected\n";

    hooks.uninstall();
    std::cout << "\nAll hook manager tests passed.\n";
    return 0;
}
