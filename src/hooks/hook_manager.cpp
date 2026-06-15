#include "hooks/hook_manager.h"
#include "core/metrics.h"
#include "core/snapshot.h"
#include <chrono>
#include <iostream>
#include <cstring>

static auto app_start = std::chrono::high_resolution_clock::now();
static uint32_t global_seq = 0;

static int64_t now_us() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>
           (now - app_start).count();
}

HookManager::HookManager(RingBuffer& buffer) : buffer_(buffer) {}

void HookManager::set_model_name(const char* name) {
    if (!name) {
        model_name_[0] = '\0';
        return;
    }
    strncpy(model_name_, name, sizeof(model_name_) - 1);
    model_name_[sizeof(model_name_) - 1] = '\0';
}

void HookManager::append_token_label(const char* label) {
    if (!label || label[0] == '\0') return;
    if (n_token_labels_ >= LayerSnapshot::TOKEN_LABEL_CAP) {
        for (int i = 0; i < LayerSnapshot::TOKEN_LABEL_CAP - 1; i++) {
            strncpy(token_labels_[i], token_labels_[i + 1],
                    sizeof(token_labels_[i]) - 1);
            token_labels_[i][sizeof(token_labels_[i]) - 1] = '\0';
        }
        n_token_labels_ = LayerSnapshot::TOKEN_LABEL_CAP - 1;
    }
    strncpy(token_labels_[n_token_labels_], label,
            sizeof(token_labels_[n_token_labels_]) - 1);
    token_labels_[n_token_labels_][sizeof(token_labels_[n_token_labels_]) - 1] =
        '\0';
    n_token_labels_++;
}

void HookManager::set_attn_weights(int layer_index, int head,
                                   const float weights[][LayerSnapshot::ATTN_CAP],
                                   int rows, int cols,
                                   int n_context_tokens,
                                   const int* weight_shape,
                                   int weight_ndim) {
    if (layer_index < 0 || layer_index >= kMaxTrackedLayers) return;
    if (rows <= 0 || cols <= 0) return;

    auto& cache = attn_weights_[layer_index];
    cache.head        = head;
    cache.n_context   = n_context_tokens;
    cache.weight_ndim = weight_ndim;
    for (int d = 0; d < 4; d++) {
        cache.weight_shape[d] = (weight_shape && d < weight_ndim)
            ? weight_shape[d] : 0;
    }

    if (rows > 1) {
        cache.rows = std::min(rows, LayerSnapshot::ATTN_CAP);
        cache.cols = std::min(cols, LayerSnapshot::ATTN_CAP);
        for (int r = 0; r < cache.rows; r++) {
            for (int c = 0; c < cache.cols; c++) {
                cache.matrix[r][c] = weights[r][c];
            }
        }
    } else {
        const int cap = LayerSnapshot::ATTN_CAP;
        const int win_cols = std::min(cols, cap);
        const int win_rows = std::min(n_context_tokens, cap);
        const int kv_start = n_context_tokens - cols;

        if (cache.valid && cache.rows > 0 && cache.cols > 0) {
            if (cache.rows >= cap && n_context_tokens > cache.n_context) {
                for (int r = 0; r < cap - 1; r++) {
                    for (int c = 0; c < cache.cols; c++) {
                        cache.matrix[r][c] = cache.matrix[r + 1][c];
                    }
                }
                cache.rows = cap;
            } else {
                cache.rows = std::min(std::max(cache.rows, win_rows), cap);
            }
            cache.cols = std::min(std::max(cache.cols, win_cols), cap);

            const int q_abs   = n_context_tokens - 1;
            const int row_base = n_context_tokens - cache.rows;
            const int col_base = n_context_tokens - cache.cols;
            const int local_r  = q_abs - row_base;

            if (local_r >= 0 && local_r < cache.rows) {
                for (int c = 0; c < cols; c++) {
                    const int abs_key = kv_start + c;
                    const int local_c = abs_key - col_base;
                    if (local_c >= 0 && local_c < cache.cols) {
                        cache.matrix[local_r][local_c] = weights[0][c];
                    }
                }
            }
        } else {
            cache.rows = 1;
            cache.cols = win_cols;
            for (int c = 0; c < cache.cols; c++) {
                cache.matrix[0][c] = weights[0][c];
            }
        }
    }

    cache.label_offset = std::max(0, n_token_labels_ - cache.cols);
    cache.row_offset   = std::max(0, n_token_labels_ - cache.rows);
    cache.valid = cache.rows > 0 && cache.cols > 0;
}

void HookManager::reset_attn_caches() {
    for (auto& cache : attn_weights_) {
        cache = {};
    }
}

void HookManager::set_decode_step(int step) {
    decode_step_ = step < 0 ? 0 : step;
}

void HookManager::set_token_labels(const char labels[][16], int count) {
    n_token_labels_ = 0;
    if (!labels || count <= 0) return;
    n_token_labels_ = count;
    if (n_token_labels_ > LayerSnapshot::TOKEN_LABEL_CAP) {
        n_token_labels_ = LayerSnapshot::TOKEN_LABEL_CAP;
    }
    for (int i = 0; i < n_token_labels_; i++) {
        strncpy(token_labels_[i], labels[i], sizeof(token_labels_[i]) - 1);
        token_labels_[i][sizeof(token_labels_[i]) - 1] = '\0';
    }
}

void HookManager::install() {
    if (installed_) return;
    installed_ = true;
    layer_prev_ = {};
    std::cout << "[HookManager] Hooks installed\n";
}

void HookManager::uninstall() {
    if (!installed_) return;
    installed_ = false;
    std::cout << "[HookManager] Hooks uninstalled\n";
}

int HookManager::layer_slot(int layer_index, LayerType type) {
    if (layer_index < 0 || layer_index >= kMaxTrackedLayers) return -1;
    const int type_idx = static_cast<int>(type);
    if (type_idx < 0 || type_idx >= kLayerTypeCount) return -1;
    return layer_index * kLayerTypeCount + type_idx;
}

void HookManager::capture_layer(int layer_index, LayerType type,
                                ComputeDevice device,
                                const float* data, int count,
                                float latency_ms,
                                const char* layer_name,
                                const int* shape, int ndim,
                                const char* dtype) {
    if (!installed_ || !data || count <= 0) return;

    LayerSnapshot snap;
    memset(&snap, 0, sizeof(snap));

    snap.sequence_id  = global_seq++;
    snap.decode_step  = decode_step_;
    snap.timestamp_us = now_us();
    snap.layer_index  = layer_index;
    snap.type         = type;
    snap.device       = device;
    snap.latency_ms   = latency_ms;
    strncpy(snap.model_name, model_name_, sizeof(snap.model_name) - 1);

    if (layer_name) {
        strncpy(snap.layer_name, layer_name, sizeof(snap.layer_name) - 1);
    } else {
        snprintf(snap.layer_name, sizeof(snap.layer_name),
                 "layer.%d", layer_index);
    }

    if (shape && ndim > 0) {
        snap.ndim = ndim;
        for (int i = 0; i < ndim && i < 4; i++) {
            snap.shape[i] = shape[i];
        }
    } else {
        snap.ndim = 1;
        snap.shape[0] = count;
    }

    if (dtype) {
        strncpy(snap.dtype, dtype, sizeof(snap.dtype) - 1);
    }

    snap.sparsity = compute_sparsity(data, count);
    snap.mean     = compute_mean(data, count);
    snap.max_abs  = compute_max_abs(data, count);

    const int slot = layer_slot(layer_index, type);

    float prev_sparsity = 0.0f;
    float prev_max_ema  = 0.0f;
    bool  has_baseline  = false;
    ComputeDevice prev_device = ComputeDevice::CUDA;
    if (slot >= 0 && layer_prev_[slot].seen) {
        prev_sparsity = layer_prev_[slot].sparsity;
        prev_max_ema  = layer_prev_[slot].max_abs_ema;
        prev_device   = layer_prev_[slot].device;
        has_baseline  = true;
    }

    snap.anomaly_flags = detect_anomalies(
        snap.max_abs, snap.sparsity, prev_sparsity,
        prev_max_ema, has_baseline,
        device, prev_device, type);

    if (slot >= 0) {
        auto& prev = layer_prev_[slot];
        prev.sparsity = snap.sparsity;
        prev.device   = device;
        prev.max_abs_ema = prev.seen
            ? (0.9f * prev.max_abs_ema + 0.1f * snap.max_abs)
            : snap.max_abs;
        prev.seen = true;
    }

    snap.n_token_labels = n_token_labels_;
    for (int i = 0; i < n_token_labels_; i++) {
        strncpy(snap.token_labels[i], token_labels_[i],
                sizeof(snap.token_labels[i]) - 1);
        snap.token_labels[i][sizeof(snap.token_labels[i]) - 1] = '\0';
    }

    if (type == LayerType::Attention &&
        layer_index >= 0 && layer_index < kMaxTrackedLayers &&
        attn_weights_[layer_index].valid) {
        const auto& w = attn_weights_[layer_index];
        snap.attn_rows = w.rows;
        snap.attn_cols = w.cols;
        snap.attn_head = w.head;
        snap.attn_label_offset = w.label_offset;
        snap.attn_row_offset   = w.row_offset;
        snap.n_context_tokens  = w.n_context;
        snap.attn_weight_ndim = w.weight_ndim;
        for (int d = 0; d < 4; d++) {
            snap.attn_weight_shape[d] = w.weight_shape[d];
        }
        for (int r = 0; r < w.rows && r < LayerSnapshot::ATTN_CAP; r++) {
            for (int c = 0; c < w.cols && c < LayerSnapshot::ATTN_CAP; c++) {
                snap.attn_matrix[r][c] = w.matrix[r][c];
            }
        }
    }

    buffer_.push(snap);
}
