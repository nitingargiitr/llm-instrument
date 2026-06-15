#pragma once

#include "core/ring_buffer.h"
#include "core/snapshot.h"

#include <array>

class HookManager {
public:
    explicit HookManager(RingBuffer& buffer);

    void install();
    void uninstall();
    void set_model_name(const char* name);
    void set_token_labels(const char labels[][16], int count);
    void append_token_label(const char* label);

    void set_attn_weights(int layer_index, int head,
                          const float weights[][LayerSnapshot::ATTN_CAP],
                          int rows, int cols,
                          int n_context_tokens,
                          const int* weight_shape = nullptr,
                          int weight_ndim = 0);
    void reset_attn_caches();
    void set_decode_step(int step);

    void capture_layer(
        int layer_index,
        LayerType type,
        ComputeDevice device,
        const float* data,
        int count,
        float latency_ms,
        const char* layer_name = nullptr,
        const int* shape = nullptr,
        int ndim = 0,
        const char* dtype = "float32"
    );

private:
    struct LayerPrevState {
        float         sparsity     = 0.0f;
        float         max_abs_ema  = 0.0f;
        ComputeDevice device       = ComputeDevice::CUDA;
        bool          seen         = false;
    };

    static constexpr int kMaxTrackedLayers = 128;
    static constexpr int kLayerTypeCount   = 5;

    static int layer_slot(int layer_index, LayerType type);

    RingBuffer& buffer_;
    bool installed_ = false;
    char model_name_[64]{};
    int  n_token_labels_ = 0;
    char token_labels_[LayerSnapshot::TOKEN_LABEL_CAP][16]{};
    int  decode_step_ = 0;
    std::array<LayerPrevState, kMaxTrackedLayers * kLayerTypeCount> layer_prev_{};

    struct AttnWeightCache {
        float matrix[LayerSnapshot::ATTN_CAP][LayerSnapshot::ATTN_CAP]{};
        int   rows     = 0;
        int   cols     = 0;
        int   head     = 0;
        bool  valid    = false;
        std::array<int, 4> weight_shape{};
        int   weight_ndim = 0;
        int   label_offset = 0;
        int   row_offset   = 0;
        int   n_context    = 0;
    };
    std::array<AttnWeightCache, kMaxTrackedLayers> attn_weights_{};
};
