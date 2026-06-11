#pragma once
#include "snapshot.h"

float    compute_sparsity(const float* data, int count,
                          float threshold = 1e-6f);
float    compute_mean    (const float* data, int count);
float    compute_max_abs (const float* data, int count);
uint32_t detect_anomalies(float max_abs, float sparsity,
                          float prev_sparsity,
                          ComputeDevice device,
                          ComputeDevice prev_device);