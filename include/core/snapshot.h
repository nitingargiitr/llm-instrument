#pragma once
#include <cstdint>
#include <array>

enum class LayerType {
    Embedding,
    Attention,
    MLP,
    LayerNorm,
    Unknown
};

enum class ComputeDevice {
    CPU,
    CUDA
};

struct LayerSnapshot {
    uint32_t      sequence_id;
    int64_t       timestamp_us;
    int           layer_index;
    LayerType     type;
    ComputeDevice device;

    std::array<int, 4> shape;
    int           ndim;
    char          dtype[16];        // fixed array instead of std::string

    float         latency_ms;
    float         sparsity;
    float         mean;
    float         max_abs;

    static constexpr int ATTN_CAP = 64;
    float         attn_matrix[ATTN_CAP][ATTN_CAP];
    int           attn_rows;
    int           attn_cols;

    uint32_t      anomaly_flags;
    static constexpr uint32_t FLAG_HIGH_ACTIVATION = 1 << 0;
    static constexpr uint32_t FLAG_CUDA_FALLBACK   = 1 << 1;
    static constexpr uint32_t FLAG_SPARSITY_DROP   = 1 << 2;
};