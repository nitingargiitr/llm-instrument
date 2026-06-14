#include "hooks/hook_manager.h"
#include "core/metrics.h"
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

void HookManager::set_model_name(const char* name) {
    if (!name) {
        model_name_[0] = '\0';
        return;
    }
    strncpy(model_name_, name, sizeof(model_name_) - 1);
    model_name_[sizeof(model_name_) - 1] = '\0';
}

void HookManager::install() {
    if (installed_) return;
    installed_ = true;
    layer_prev_ = {};
    std::cout << "[HookManager] Hooks installed\n";
}

void HookManager::uninstall() {
    if (!installed_) return;
    installed_ = false;
    std::cout << "[HookManager] Hooks uninstalled\n";
}

int HookManager::layer_slot(int layer_index, LayerType type) {
    if (layer_index < 0 || layer_index >= kMaxTrackedLayers) return -1;
    const int type_idx = static_cast<int>(type);
    if (type_idx < 0 || type_idx >= kLayerTypeCount) return -1;
    return layer_index * kLayerTypeCount + type_idx;
}

void HookManager::capture_layer(int layer_index, LayerType type,
                                ComputeDevice device,
                                const float* data, int count,
                                float latency_ms,
                                const char* layer_name,
                                const int* shape, int ndim,
                                const char* dtype) {
    if (!installed_ || !data || count <= 0) return;

    LayerSnapshot snap;
    memset(&snap, 0, sizeof(snap));

    snap.sequence_id  = global_seq++;
    snap.timestamp_us = now_us();
    snap.layer_index  = layer_index;
    snap.type         = type;
    snap.device       = device;
    snap.latency_ms   = latency_ms;
    strncpy(snap.model_name, model_name_, sizeof(snap.model_name) - 1);

    if (layer_name) {
        strncpy(snap.layer_name, layer_name, sizeof(snap.layer_name) - 1);
    } else {
        snprintf(snap.layer_name, sizeof(snap.layer_name),
                 "layer.%d", layer_index);
    }

    if (shape && ndim > 0) {
        snap.ndim = ndim;
        for (int i = 0; i < ndim && i < 4; i++) {
            snap.shape[i] = shape[i];
        }
    } else {
        snap.ndim = 1;
        snap.shape[0] = count;
    }

    if (dtype) {
        strncpy(snap.dtype, dtype, sizeof(snap.dtype) - 1);
    }

    snap.sparsity = compute_sparsity(data, count);
    snap.mean     = compute_mean(data, count);
    snap.max_abs  = compute_max_abs(data, count);

    const int slot = layer_slot(layer_index, type);

    float prev_sparsity = 0.0f;
    float prev_max_ema  = 0.0f;
    bool  has_baseline  = false;
    ComputeDevice prev_device = ComputeDevice::CUDA;
    if (slot >= 0 && layer_prev_[slot].seen) {
        prev_sparsity = layer_prev_[slot].sparsity;
        prev_max_ema  = layer_prev_[slot].max_abs_ema;
        prev_device   = layer_prev_[slot].device;
        has_baseline  = true;
    }

    snap.anomaly_flags = detect_anomalies(
        snap.max_abs, snap.sparsity, prev_sparsity,
        prev_max_ema, has_baseline,
        device, prev_device, type);

    if (slot >= 0) {
        auto& prev = layer_prev_[slot];
        prev.sparsity = snap.sparsity;
        prev.device   = device;
        prev.max_abs_ema = prev.seen
            ? (0.9f * prev.max_abs_ema + 0.1f * snap.max_abs)
            : snap.max_abs;
        prev.seen = true;
    }

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
                float v = data[r * 8 + c];
                snap.attn_matrix[r][c] = (v < 0 ? -v : v) / local_max;
            }
        }
    }

    buffer_.push(snap);
}
