#include "core/metrics.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    std::cout << "Starting metrics tests...\n";

    // Test 1: sparsity all zeros
    {
        float data[4] = {0, 0, 0, 0};
        float s = compute_sparsity(data, 4);
        assert(std::abs(s - 1.0f) < 1e-5f);
        std::cout << "PASS: sparsity all zeros\n";
    }

    // Test 2: sparsity none zero
    {
        float data[4] = {1, 2, 3, 4};
        float s = compute_sparsity(data, 4);
        assert(std::abs(s - 0.0f) < 1e-5f);
        std::cout << "PASS: sparsity dense tensor\n";
    }

    // Test 3: mean
    {
        float data[4] = {1, 2, 3, 4};
        float m = compute_mean(data, 4);
        assert(std::abs(m - 2.5f) < 1e-5f);
        std::cout << "PASS: compute_mean\n";
    }

    // Test 4: max_abs
    {
        float data[4] = {-1, 2, -7, 4};
        float m = compute_max_abs(data, 4);
        assert(std::abs(m - 7.0f) < 1e-5f);
        std::cout << "PASS: compute_max_abs\n";
    }

    // Test 5: high activation requires baseline + spike
    {
        uint32_t flags = detect_anomalies(
            7.0f, 0.1f, 0.1f, 0.0f, false,
            ComputeDevice::CUDA, ComputeDevice::CUDA,
            LayerType::MLP);
        assert(flags == 0);
        std::cout << "PASS: no high act without baseline\n";

        flags = detect_anomalies(
            40.0f, 0.1f, 0.1f, 7.0f, true,
            ComputeDevice::CUDA, ComputeDevice::CUDA,
            LayerType::MLP);
        assert(flags & LayerSnapshot::FLAG_HIGH_ACTIVATION);
        std::cout << "PASS: high activation spike anomaly\n";
    }

    // Test 6: CUDA fallback on CPU LayerNorm
    {
        uint32_t flags = detect_anomalies(
            1.0f, 0.2f, 0.2f, 1.0f, true,
            ComputeDevice::CPU, ComputeDevice::CUDA,
            LayerType::LayerNorm);
        assert(flags & LayerSnapshot::FLAG_CUDA_FALLBACK);
        std::cout << "PASS: CUDA fallback on CPU norm\n";
    }

    // Test 7: sparsity drop
    {
        uint32_t flags = detect_anomalies(
            1.0f, 0.1f, 0.8f, 1.0f, true,
            ComputeDevice::CUDA, ComputeDevice::CUDA,
            LayerType::Attention);
        assert(flags & LayerSnapshot::FLAG_SPARSITY_DROP);
        std::cout << "PASS: sparsity drop anomaly\n";
    }

    // Test 8: no false positives on stable layer
    {
        uint32_t flags = detect_anomalies(
            1.0f, 0.5f, 0.45f, 1.0f, true,
            ComputeDevice::CUDA, ComputeDevice::CUDA,
            LayerType::MLP);
        assert(flags == 0);
        std::cout << "PASS: no false positive anomalies\n";
    }

    std::cout << "\nAll metrics tests passed.\n";
    return 0;
}
