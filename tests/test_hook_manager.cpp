#include "core/ring_buffer.h"
#include "hooks/hook_manager.h"
#include "core/snapshot.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

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

    char labels[2][16] = {};
    std::snprintf(labels[0], sizeof(labels[0]), "Hello");
    std::snprintf(labels[1], sizeof(labels[1]), "world");
    hooks.set_token_labels(labels, 2);
    hooks.append_token_label("!");
    hooks.capture_layer(7, LayerType::Attention, ComputeDevice::CUDA,
                        dense_data, 128, 1.0f);
    if (auto snap = ring_buffer.pop()) {
        assert(snap->n_token_labels == 3);
        assert(std::string(snap->token_labels[0]) == "Hello");
        assert(std::string(snap->token_labels[1]) == "world");
        assert(std::string(snap->token_labels[2]) == "!");
        std::cout << "PASS: token labels attached to snapshot\n";
    }

    float weights[LayerSnapshot::ATTN_CAP][LayerSnapshot::ATTN_CAP]{};
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            weights[r][c] = (r == c) ? 0.5f : 0.05f;
        }
    }
    int wshape[4] = {8, 8, 1, 1};
    hooks.set_attn_weights(7, 0, weights, 8, 8, 8, wshape, 4);
    hooks.capture_layer(7, LayerType::Attention, ComputeDevice::CUDA,
                        dense_data, 128, 1.0f);
    if (auto snap = ring_buffer.pop()) {
        assert(snap->attn_rows == 8);
        assert(snap->attn_cols == 8);
        assert(snap->attn_matrix[0][0] > 0.4f);
        std::cout << "PASS: attention weights merged into snapshot\n";
    }

    float row[LayerSnapshot::ATTN_CAP][LayerSnapshot::ATTN_CAP]{};
    for (int c = 0; c < 9; c++) {
        row[0][c] = 0.1f * static_cast<float>(c + 1);
    }
    hooks.set_attn_weights(7, 0, row, 1, 9, 9, wshape, 4);
    hooks.capture_layer(7, LayerType::Attention, ComputeDevice::CUDA,
                        dense_data, 128, 1.0f, "layers.7.attn");
    if (auto snap = ring_buffer.pop()) {
        assert(snap->attn_rows == 9);
        assert(snap->attn_cols == 9);
        assert(snap->attn_matrix[7][7] > 0.4f);
        assert(snap->attn_matrix[8][8] > 0.8f);
        std::cout << "PASS: single-token row merged into attention grid\n";
    }

    hooks.uninstall();
    std::cout << "\nAll hook manager tests passed.\n";
    return 0;
}
