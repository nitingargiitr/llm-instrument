#pragma once
#include <cstdint>
#include <string>
#include <array>

// What kind of layer produced this snapshot
enum class LayerType {
    Embedding,
    Attention,
    MLP,
    LayerNorm,
    Unknown
};

// Where the computation ran
enum class ComputeDevice {
    CPU,
    CUDA
};

// One snapshot = one layer captured during one forward pass
struct LayerSnapshot {
    uint32_t      sequence_id;        // increments every capture, globally unique
    int64_t       timestamp_us;       // microseconds since app started
    int           layer_index;        // which layer number (0, 1, 2 ...)
    LayerType     type;               // what kind of layer
    ComputeDevice device;             // cpu or gpu

    // Tensor information
    std::array<int, 4> shape;         // tensor dimensions, unused = 0
    int           ndim;               // how many dimensions are valid
    std::string   dtype;              // "float16", "float32"

    // Performance metrics
    float         latency_ms;         // how long this layer took in milliseconds
    float         sparsity;           // 0.0 = dense, 1.0 = all zeros
    float         mean;               // average activation value
    float         max_abs;            // largest absolute activation value

    // Attention matrix (capped at 64x64 to save memory)
    static constexpr int ATTN_CAP = 64;
    float         attn_matrix[ATTN_CAP][ATTN_CAP];
    int           attn_rows;          // how many rows are stored
    int           attn_cols;          // how many cols are stored

    // Anomaly flags - use bitwise OR to combine
    uint32_t      anomaly_flags;
    static constexpr uint32_t FLAG_HIGH_ACTIVATION = 1 << 0;  // max > threshold
    static constexpr uint32_t FLAG_CUDA_FALLBACK   = 1 << 1;  // gpu fell back to cpu
    static constexpr uint32_t FLAG_SPARSITY_DROP   = 1 << 2;  // sudden sparsity change
};