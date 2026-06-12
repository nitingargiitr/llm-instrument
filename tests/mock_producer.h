#pragma once
#include "core/ring_buffer.h"
#include <thread>
#include <cmath>
#include <cstring>
#include <cstdlib>

inline LayerSnapshot make_fake_snapshot(uint32_t seq_id, int layer_idx) {
    LayerSnapshot snap;
    memset(&snap, 0, sizeof(snap));

    snap.sequence_id  = seq_id;
    snap.layer_index  = layer_idx;
    snap.timestamp_us = seq_id * 1000;
    snap.ndim         = 3;
    snap.shape[0]     = 1;
    snap.shape[1]     = 32;
    snap.shape[2]     = 4096;
    snap.latency_ms   = 0.8f + (layer_idx % 4) * 0.3f;
    snap.sparsity     = 0.4f + 0.1f * std::sin(seq_id * 0.1f);
    snap.mean         = 0.01f * layer_idx;
    snap.max_abs      = 3.5f + (rand() % 100) / 100.0f;
    snap.anomaly_flags = 0;
    strncpy(snap.dtype, "float16", sizeof(snap.dtype) - 1);

    switch (layer_idx % 4) {
        case 0:
            snap.type   = LayerType::Embedding;
            snap.device = ComputeDevice::CUDA; break;
        case 1:
            snap.type   = LayerType::Attention;
            snap.device = ComputeDevice::CUDA; break;
        case 2:
            snap.type   = LayerType::MLP;
            snap.device = ComputeDevice::CUDA; break;
        case 3:
            snap.type   = LayerType::LayerNorm;
            snap.device = ComputeDevice::CPU;  break;
    }

    snap.attn_rows = 8;
    snap.attn_cols = 8;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            snap.attn_matrix[r][c] =
                std::exp(-std::abs(r - c) * 0.5f);

    if (snap.max_abs > 4.0f)
        snap.anomaly_flags |= LayerSnapshot::FLAG_HIGH_ACTIVATION;
    if (snap.device == ComputeDevice::CPU)
        snap.anomaly_flags |= LayerSnapshot::FLAG_CUDA_FALLBACK;

    return snap;
}

inline void run_mock_producer(RingBuffer& buffer, bool& running) {
    uint32_t seq   = 0;
    int      layer = 0;
    while (running) {
        buffer.push(make_fake_snapshot(seq++, layer));
        layer = (layer + 1) % 32;
        std::this_thread::sleep_for(
            std::chrono::milliseconds(50));
    }
}
