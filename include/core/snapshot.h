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
    Metal,
    CUDA,
};

struct LayerSnapshot {
    uint32_t      sequence_id;
    int           decode_step;      // 0 = prefill, >=1 = nth generated token
    int64_t       timestamp_us;
    int           layer_index;
    LayerType     type;
    ComputeDevice device;

    char          layer_name[64];   // e.g. "layers.1.attn"
    char          model_name[64];   // e.g. "llama-3-8b"

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
    int           attn_head;
    int           attn_label_offset;  // token_labels index for matrix col 0
    int           attn_row_offset;    // token_labels index for matrix row 0
    int           n_context_tokens;   // tokens in KV when matrix was captured
    std::array<int, 4> attn_weight_shape;
    int           attn_weight_ndim;

    static constexpr int TOKEN_LABEL_CAP = 16;
    int           n_token_labels;
    char          token_labels[TOKEN_LABEL_CAP][16];

    uint32_t      anomaly_flags;
    static constexpr uint32_t FLAG_HIGH_ACTIVATION = 1 << 0;
    static constexpr uint32_t FLAG_CUDA_FALLBACK   = 1 << 1;
    static constexpr uint32_t FLAG_SPARSITY_DROP   = 1 << 2;
};

void print_snapshot(const LayerSnapshot& snap);