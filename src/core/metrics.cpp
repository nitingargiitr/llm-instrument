#include "core/metrics.h"
#include <cmath>
#include <algorithm>

float compute_sparsity(const float* data, int count, float threshold) {
    if (count <= 0) return 0.0f;
    int zero_count = 0;
    for (int i = 0; i < count; i++) {
        if (std::abs(data[i]) < threshold) zero_count++;
    }
    return static_cast<float>(zero_count) / count;
}

float compute_mean(const float* data, int count) {
    if (count <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < count; i++) sum += data[i];
    return sum / count;
}

float compute_max_abs(const float* data, int count) {
    if (count <= 0) return 0.0f;
    float max_val = 0.0f;
    for (int i = 0; i < count; i++) {
        max_val = std::max(max_val, std::abs(data[i]));
    }
    return max_val;
}

uint32_t detect_anomalies(float max_abs, float sparsity,
                          float prev_sparsity, float prev_max_abs_ema,
                          bool has_max_baseline, ComputeDevice device,
                          ComputeDevice prev_device, LayerType type) {
    uint32_t flags = 0;

    if (has_max_baseline &&
        max_abs > METRICS_HIGH_ACTIVATION_FLOOR &&
        max_abs > prev_max_abs_ema * METRICS_HIGH_ACTIVATION_SPIKE_RATIO) {
        flags |= LayerSnapshot::FLAG_HIGH_ACTIVATION;
    }

    if (device == ComputeDevice::CPU &&
        (prev_device == ComputeDevice::CUDA ||
         prev_device == ComputeDevice::Metal ||
         type == LayerType::LayerNorm)) {
        flags |= LayerSnapshot::FLAG_CUDA_FALLBACK;
    }

    if (prev_sparsity > 0.0f &&
        std::abs(sparsity - prev_sparsity) > METRICS_SPARSITY_DROP_DELTA) {
        flags |= LayerSnapshot::FLAG_SPARSITY_DROP;
    }

    return flags;
}
