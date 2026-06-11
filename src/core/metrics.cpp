#include "core/metrics.h"
#include <cmath>
#include <algorithm>

// Compute sparsity - fraction of values near zero
float compute_sparsity(const float* data, int count, float threshold) {
    int zero_count = 0;
    for (int i = 0; i < count; i++) {
        if (std::abs(data[i]) < threshold) zero_count++;
    }
    return static_cast<float>(zero_count) / count;
}

// Compute mean of array
float compute_mean(const float* data, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) sum += data[i];
    return sum / count;
}

// Compute max absolute value
float compute_max_abs(const float* data, int count) {
    float max_val = 0.0f;
    for (int i = 0; i < count; i++) {
        max_val = std::max(max_val, std::abs(data[i]));
    }
    return max_val;
}

// Check anomaly flags based on computed metrics
uint32_t detect_anomalies(float max_abs, float sparsity,
                           float prev_sparsity, ComputeDevice device,
                           ComputeDevice prev_device) {
    uint32_t flags = 0;

    if (max_abs > 6.0f)
        flags |= LayerSnapshot::FLAG_HIGH_ACTIVATION;

    if (device == ComputeDevice::CPU && prev_device == ComputeDevice::CUDA)
        flags |= LayerSnapshot::FLAG_CUDA_FALLBACK;

    if (prev_sparsity > 0.0f &&
        std::abs(sparsity - prev_sparsity) > 0.3f)
        flags |= LayerSnapshot::FLAG_SPARSITY_DROP;

    return flags;
}