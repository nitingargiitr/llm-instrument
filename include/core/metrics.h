#pragma once
#include "snapshot.h"

constexpr float METRICS_SPARSITY_THRESHOLD          = 1e-6f;
constexpr float METRICS_HIGH_ACTIVATION_FLOOR       = 6.0f;
constexpr float METRICS_HIGH_ACTIVATION_SPIKE_RATIO = 4.0f;
constexpr float METRICS_SPARSITY_DROP_DELTA         = 0.3f;

float    compute_sparsity(const float* data, int count,
                          float threshold = METRICS_SPARSITY_THRESHOLD);
float    compute_mean    (const float* data, int count);
float    compute_max_abs (const float* data, int count);
uint32_t detect_anomalies(float max_abs, float sparsity,
                          float prev_sparsity,
                          float prev_max_abs_ema, bool has_max_baseline,
                          ComputeDevice device,
                          ComputeDevice prev_device,
                          LayerType type = LayerType::Unknown);
