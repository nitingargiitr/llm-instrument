#include "hooks/hook_manager.h"
#include "core/snapshot.h"
#include <chrono>
#include <iostream>
#include <cstring>

static auto app_start = std::chrono::high_resolution_clock::now();
static uint32_t global_seq = 0;

static int64_t now_us() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>
           (now - app_start).count();
}

HookManager::HookManager(RingBuffer& buffer) : buffer_(buffer) {}

void HookManager::install() {
    if (installed_) return;
    installed_ = true;
    std::cout << "[HookManager] Hooks installed\n";
}

void HookManager::uninstall() {
    if (!installed_) return;
    installed_ = false;
    std::cout << "[HookManager] Hooks uninstalled\n";
}

void HookManager::capture_layer(int layer_index, LayerType type,
                                  ComputeDevice device,
                                  const float* data, int count,
                                  float latency_ms) {
    if (!installed_) return;

    LayerSnapshot snap;
    memset(&snap, 0, sizeof(snap));

    snap.sequence_id  = global_seq++;
    snap.timestamp_us = now_us();
    snap.layer_index  = layer_index;
    snap.type         = type;
    snap.device       = device;
    snap.latency_ms   = latency_ms;
    snap.ndim         = 1;
    snap.shape[0]     = count;
    strncpy(snap.dtype, "float32", sizeof(snap.dtype) - 1);

    // Compute metrics in one pass
    float sum = 0, max_abs = 0, zeros = 0;
    for (int i = 0; i < count; i++) {
        float v = data[i];
        float av = v < 0 ? -v : v;
        sum += v;
        if (av > max_abs) max_abs = av;
        if (av < 1e-6f) zeros++;
    }
    snap.mean     = count > 0 ? sum / count : 0;
    snap.max_abs  = max_abs;
    snap.sparsity = count > 0 ? zeros / count : 0;

    // Anomaly flags
    if (max_abs > 6.0f)
        snap.anomaly_flags |= LayerSnapshot::FLAG_HIGH_ACTIVATION;
    if (device == ComputeDevice::CPU)
        snap.anomaly_flags |= LayerSnapshot::FLAG_CUDA_FALLBACK;

    // If this is an attention layer, fill a fake-but-real attention matrix
    // from the data itself (8x8 sample) so Panel 3 has something to show
    if (type == LayerType::Attention && count >= 64) {
        snap.attn_rows = 8;
        snap.attn_cols = 8;
        float local_max = 0.0f;
        for (int i = 0; i < 64; i++) {
            float av = data[i] < 0 ? -data[i] : data[i];
            if (av > local_max) local_max = av;
        }
        if (local_max < 1e-6f) local_max = 1.0f;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                float v = data[r * 8 + c] < 0
                          ? -data[r * 8 + c]
                          :  data[r * 8 + c];
                snap.attn_matrix[r][c] = v / local_max;
            }
        }
    }

    buffer_.push(snap);
}