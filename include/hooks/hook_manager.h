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
    std::array<LayerPrevState, kMaxTrackedLayers * kLayerTypeCount> layer_prev_{};
};
