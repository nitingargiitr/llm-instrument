#include "tui/tui_app.h"
#include "core/metrics.h"
#include "core/snapshot.h"
#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ftxui;

namespace {

// Monochrome theme with salmon accent (mockup)
constexpr const char* kCaptureLabel = " [Active Capture Target]";

static bool verbose_logging() {
    const char* v = std::getenv("LLM_INSTRUMENT_VERBOSE");
    return v && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

// Background threads (model runner, llama.cpp) write to stderr while the TUI
// paints the primary screen — each line flashes above the header until redraw.
class StderrSilencer {
public:
    explicit StderrSilencer(bool enable) {
        if (!enable) return;
        std::fflush(stderr);
        saved_fd_ = dup(STDERR_FILENO);
        if (saved_fd_ < 0) return;
        const int null_fd = open("/dev/null", O_WRONLY);
        if (null_fd < 0) return;
        dup2(null_fd, STDERR_FILENO);
        close(null_fd);
        active_ = true;
    }

    ~StderrSilencer() {
        if (!active_ || saved_fd_ < 0) return;
        std::fflush(stderr);
        dup2(saved_fd_, STDERR_FILENO);
        close(saved_fd_);
    }

private:
    int  saved_fd_ = -1;
    bool active_   = false;
};

static Color accent_color() { return Color::RGB(255, 120, 150); }

static std::string padded_event_count(size_t n) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%5zu events", n);
    return buf;
}

static float attn_weight_sparsity(const LayerSnapshot* sel) {
    if (!sel || sel->attn_rows <= 0 || sel->attn_cols <= 0) return 0.0f;
    std::vector<float> weights;
    weights.reserve(static_cast<size_t>(sel->attn_rows) * sel->attn_cols);
    for (int r = 0; r < sel->attn_rows; r++) {
        for (int c = 0; c < sel->attn_cols; c++) {
            weights.push_back(sel->attn_matrix[r][c]);
        }
    }
    return compute_sparsity(weights.data(), static_cast<int>(weights.size()));
}

// Attention matrix labels (token piece when available, else tN).
constexpr int kMatrixColW = 6;

static std::string matrix_pos_label(int abs_pos) {
    return "t" + std::to_string(abs_pos);
}

static std::string matrix_token_label(const LayerSnapshot* sel, int label_idx,
                                      int abs_pos_fallback) {
    if (sel && label_idx >= 0 && label_idx < sel->n_token_labels &&
        sel->token_labels[label_idx][0] != '\0') {
        return sel->token_labels[label_idx];
    }
    return matrix_pos_label(abs_pos_fallback);
}

static std::string matrix_col_label(const LayerSnapshot* sel, int col) {
    const int kv_pos = sel->n_context_tokens - sel->attn_cols + col;
    return matrix_token_label(sel, sel->attn_label_offset + col, kv_pos);
}

static std::string matrix_row_label(const LayerSnapshot* sel, int row) {
    const int q_pos = sel->n_context_tokens - sel->attn_rows + row;
    return matrix_token_label(sel, sel->attn_row_offset + row, q_pos);
}

static std::string matrix_pad_center(const std::string& s, int width) {
    if ((int)s.size() >= width) return s.substr(0, width);
    const int pad = width - static_cast<int>(s.size());
    const int left = pad / 2;
    return std::string(left, ' ') + s + std::string(pad - left, ' ');
}

static std::string matrix_header_row(int col_start, int col_end,
                                     const LayerSnapshot* sel) {
    std::string s = std::string(kMatrixColW, ' ');
    if (!sel) return s;
    for (int c = col_start; c < col_end; c++) {
        s += matrix_pad_center(matrix_col_label(sel, c), kMatrixColW);
    }
    return s;
}

static std::string matrix_glyph_str(float v, float threshold);

static std::string matrix_data_row(int row, int col_start, int col_end,
                                   const LayerSnapshot* sel, float threshold) {
    std::string s = matrix_pad_center(matrix_row_label(sel, row), kMatrixColW);
    float row_max = 0.0f;
    for (int c = col_start; c < col_end; c++) {
        if (c < sel->attn_cols) {
            row_max = std::max(row_max, sel->attn_matrix[row][c]);
        }
    }
    if (row_max < 1e-8f) row_max = 1.0f;
    for (int c = col_start; c < col_end; c++) {
        const float norm = (c < sel->attn_cols)
            ? sel->attn_matrix[row][c] / row_max : 0.0f;
        s += matrix_pad_center(matrix_glyph_str(norm, threshold), kMatrixColW);
    }
    return s;
}

static std::string matrix_glyph_str(float v, float threshold) {
    const float t0 = threshold * 0.4f;
    const float t1 = threshold * 0.8f;
    const float t2 = threshold * 1.2f;
    const float t3 = threshold * 1.6f;
    if (v > t3) return "#";
    if (v > t2) return "+";
    if (v > t1) return ":";
    if (v > t0) return ".";
    return " ";
}

struct TreeLine {
    std::string label;
    int         layer_idx = -1;
    LayerType   type      = LayerType::Unknown;
    bool        is_root   = false;
};

static Element styled_panel(const std::string& title, Element content,
                            bool focused) {
    auto title_el = text(" " + title + " ") | bold;
    return window(title_el, content, focused ? HEAVY : LIGHT);
}

static void sync_scroll(int cursor, int& scroll, int visible, int total) {
    if (total <= visible) {
        scroll = 0;
        return;
    }
    if (cursor < scroll) scroll = cursor;
    if (cursor >= scroll + visible) scroll = cursor - visible + 1;
    scroll = std::max(0, std::min(scroll, total - visible));
}

static int layer_type_order(LayerType type, const char* name) {
    switch (type) {
        case LayerType::Embedding: return 0;
        case LayerType::Attention: return 1;
        case LayerType::MLP:       return 2;
        case LayerType::LayerNorm:
            if (std::strstr(name, "attn_norm")) return 3;
            if (std::strstr(name, "ffn_norm"))  return 4;
            if (std::strstr(name, "result_norm")) return 100;
            return 5;
        default: return 10;
    }
}

static void rebuild_tree(const std::vector<LayerSnapshot>& layers,
                         std::vector<TreeLine>& tree_lines) {
    tree_lines.clear();
    if (layers.empty()) {
        tree_lines.push_back({"  Waiting for data...", -1,
                              LayerType::Unknown, false});
        return;
    }

    const char* root = layers[0].model_name[0] ? layers[0].model_name : "model";
    tree_lines.push_back({std::string("\xe2\x94\x9c\xe2\x94\x80 ") + root, -1,
                          LayerType::Unknown, true});

    std::vector<int> order(layers.size());
    for (int i = 0; i < (int)layers.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const auto& la = layers[a];
        const auto& lb = layers[b];
        if (la.layer_index != lb.layer_index)
            return la.layer_index < lb.layer_index;
        const int oa = layer_type_order(la.type, la.layer_name);
        const int ob = layer_type_order(lb.type, lb.layer_name);
        if (oa != ob) return oa < ob;
        return std::strcmp(la.layer_name, lb.layer_name) < 0;
    });

    for (int idx : order) {
        const auto& snap = layers[idx];
        int depth = 1;
        if (std::strncmp(snap.layer_name, "layers.", 7) == 0) depth = 2;
        std::string indent(depth * 2, ' ');
        tree_lines.push_back({indent + snap.layer_name, idx, snap.type, false});
    }
}

static int layer_slot(int layer_index, LayerType type) {
    return layer_index * 5 + static_cast<int>(type);
}

static std::string format_timestamp_us(int64_t us) {
    const int64_t total_ms = us / 1000;
    const int     ms       = static_cast<int>(total_ms % 1000);
    const int64_t total_s  = total_ms / 1000;
    const int     s        = static_cast<int>(total_s % 60);
    const int     m        = static_cast<int>((total_s / 60) % 60);
    const int     h        = static_cast<int>((total_s / 3600) % 24);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", h, m, s, ms);
    return buf;
}

static std::string spec_layer_type(LayerType t) {
    switch (t) {
        case LayerType::Embedding: return "Embedding";
        case LayerType::Attention: return "Attn (Self)";
        case LayerType::MLP:       return "MLP (SwiGLU)";
        case LayerType::LayerNorm: return "LayerNorm";
        default:                   return "Unknown";
    }
}

static std::string spec_device(ComputeDevice d) {
    switch (d) {
        case ComputeDevice::Metal: return "Metal [GPU 0]";
        case ComputeDevice::CUDA:  return "CUDA [GPU 0]";
        default:                   return "CPU (Fallback)";
    }
}

static bool matches_capture(const LayerSnapshot& s,
                            const char* capture_name) {
    if (!capture_name || capture_name[0] == '\0') return true;
    return std::strcmp(s.layer_name, capture_name) == 0;
}

static int stream_sort_key(const LayerSnapshot& s) {
    return s.layer_index * 10 + layer_type_order(s.type, s.layer_name);
}

static void push_stream_group_front(std::deque<LayerSnapshot>& stream,
                                    std::vector<LayerSnapshot> snaps) {
    std::sort(snaps.begin(), snaps.end(),
              [](const LayerSnapshot& a, const LayerSnapshot& b) {
                  return stream_sort_key(a) < stream_sort_key(b);
              });
    for (auto it = snaps.rbegin(); it != snaps.rend(); ++it) {
        stream.push_front(*it);
    }
}

static std::string stream_layer_label(const LayerSnapshot& s) {
    std::string lname = s.layer_name;
    if (s.decode_step > 0) {
        lname += " g" + std::to_string(s.decode_step);
    } else {
        lname += " pre";
    }
    if (lname.size() > 16) lname = lname.substr(0, 16);
    return lname;
}

static const LayerSnapshot* find_layer_by_name(
    const std::vector<LayerSnapshot>& layers,
    const char* name) {
    if (!name || name[0] == '\0') return nullptr;
    for (const auto& l : layers) {
        if (std::strcmp(l.layer_name, name) == 0) return &l;
    }
    return nullptr;
}

static std::string float_str(float v, int decimals = 2) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return std::string(buf);
}

static Element render_topology_panel(
    const std::vector<TreeLine>& tree_lines,
    int tree_cursor,
    const char* capture_target_name,
    const std::vector<LayerSnapshot>& layers,
    int& scroll,
    int height) {
    const int visible = height - 2;
    sync_scroll(tree_cursor, scroll, visible, static_cast<int>(tree_lines.size()));

    Elements rows;
    const int end = std::min((int)tree_lines.size(), scroll + visible);
    for (int i = scroll; i < end; i++) {
        const auto& line = tree_lines[i];
        std::string text_line = line.label;
        const bool is_capture =
            line.layer_idx >= 0 &&
            matches_capture(layers[line.layer_idx], capture_target_name);
        if (is_capture) {
            text_line += kCaptureLabel;
        }
        Element row = text(text_line);
        if (line.is_root) {
            row = row | bold;
        }
        if (is_capture) {
            row = row | color(accent_color());
        }
        if (line.layer_idx >= 0 && i == tree_cursor) {
            row = row | inverted;
        }
        rows.push_back(row);
    }
    while ((int)rows.size() < visible) {
        rows.push_back(text(""));
    }
    return vbox(rows) | vscroll_indicator | frame | size(HEIGHT, EQUAL, height);
}

static Element render_stream_panel(const std::deque<LayerSnapshot>& stream,
                                   const char* capture_name,
                                   int& scroll, int height) {
    Elements stream_rows;
    stream_rows.push_back(hbox({
        text(" ID ") | bold | color(accent_color()) | size(WIDTH, EQUAL, 5),
        text(" TIMESTAMP    ") | bold | dim | size(WIDTH, EQUAL, 14),
        text(" LAYER            ") | bold | dim | size(WIDTH, EQUAL, 18),
        text(" TYPE         ") | bold | color(accent_color()) | size(WIDTH, EQUAL, 12),
        text(" DEVICE         ") | bold | dim,
    }));
    stream_rows.push_back(separator());

    const int visible = height - 3;
    const int max_scroll = std::max(0, (int)stream.size() - visible);
    scroll = std::min(scroll, max_scroll);
    const int end = std::min((int)stream.size(), scroll + visible);
    for (int i = scroll; i < end; i++) {
        const auto& s = stream[i];
        const bool is_target = capture_name && capture_name[0] &&
            matches_capture(s, capture_name);
        const bool highlight_type =
            s.type == LayerType::Attention || s.type == LayerType::MLP;
        const std::string lname = stream_layer_label(s);
        stream_rows.push_back(hbox({
            text(" " + std::to_string(s.sequence_id))
                | size(WIDTH, EQUAL, 5)
                | color(is_target ? accent_color() : Color::Default),
            text(" " + format_timestamp_us(s.timestamp_us))
                | size(WIDTH, EQUAL, 14) | color(Color::Yellow),
            text(" " + lname) | size(WIDTH, EQUAL, 18)
                | (is_target ? color(accent_color()) : dim),
            text(" " + spec_layer_type(s.type))
                | size(WIDTH, EQUAL, 12)
                | (highlight_type ? color(accent_color()) : dim),
            text(" " + spec_device(s.device)),
        }));
    }

    return vbox(stream_rows) | vscroll_indicator | frame |
           size(HEIGHT, EQUAL, height);
}

static Element render_matrix_panel(const LayerSnapshot* sel,
                                   int pan_row, int pan_col,
                                   float threshold, bool fullscreen,
                                   bool focused) {
    constexpr int kView = 8;
    const int view_rows = fullscreen ? 16 : kView;
    const int view_cols = fullscreen ? 16 : kView;

    Elements grid;
    if (sel && sel->attn_rows > 0 && sel->type == LayerType::Attention) {
        const int max_r  = sel->attn_rows;
        const int max_c  = sel->attn_cols;
        const bool bar_mode = max_r <= 1;

        if (bar_mode) {
            grid.push_back(text(
                "  Last query token -> key positions (not full NxN matrix)")
                               | dim);
            grid.push_back(text(matrix_header_row(0, max_c, sel))
                               | color(Color::Yellow));
            grid.push_back(text(matrix_data_row(0, 0, max_c, sel, threshold)));
            grid.push_back(text(""));
            const int q_pos = sel->n_context_tokens - 1;
            grid.push_back(text(
                "  query " + matrix_row_label(sel, 0)
                + " (pos " + std::to_string(q_pos) + ")  |  "
                + std::to_string(sel->n_context_tokens) + " ctx tokens  |  "
                + "brighter = stronger weight")
                | dim);
        } else {
            const int vr_end = std::min(pan_row + view_rows, max_r);
            const int vc_end = std::min(pan_col + view_cols, max_c);
            grid.push_back(text(matrix_header_row(pan_col, vc_end, sel))
                               | color(Color::Yellow));
            for (int r = pan_row; r < vr_end; r++) {
                grid.push_back(text(
                    matrix_data_row(r, pan_col, vc_end, sel, threshold)));
            }
            grid.push_back(text(""));
            grid.push_back(text(
                "  " + std::to_string(max_r) + "x" + std::to_string(max_c)
                + " softmax map  |  hjkl pan")
                | dim);
        }
        grid.push_back(hbox({
            text(" [Focus + F]: Fullscreen") | dim,
            text("  [+/-]: Contrast") | dim,
        }));
    } else {
        grid.push_back(
            text("  Select an Attention layer; Space sets capture target") | dim);
    }

    const int box_h = fullscreen ? view_rows + 5 : kView + 5;
    std::string title = fullscreen
        ? "3. ATTENTION WEIGHTS (fullscreen)"
        : "3. ATTENTION WEIGHTS (HEAD "
          + std::to_string(sel && sel->attn_rows > 0 ? sel->attn_head : 0) + ")";
    return styled_panel(title, vbox(grid) | size(HEIGHT, EQUAL, box_h), focused);
}

static Element render_metrics_panel(
    const LayerSnapshot* sel,
    const std::unordered_map<int, float>& latency_ema) {
    Elements metric_rows;
    if (sel) {
        metric_rows.push_back(hbox({
            text("  Layer   : ") | bold,
            text(std::string(sel->layer_name)),
        }));
        {
            std::string shape_str = "[";
            const int dims = sel->ndim > 0 ? sel->ndim : 1;
            for (int d = 0; d < dims && d < 4; d++) {
                if (d > 0) shape_str += ", ";
                shape_str += std::to_string(sel->shape[d]);
            }
            shape_str += "]";
            metric_rows.push_back(hbox({
                text("  Shape (activation): ") | bold,
                text(shape_str),
            }));
        }
        metric_rows.push_back(hbox({
            text("  Dtype   : ") | bold,
            text(std::string(sel->dtype)),
        }));
        if (sel->attn_rows > 0 && sel->attn_cols > 0) {
            metric_rows.push_back(hbox({
                text("  Attn map: ") | bold,
                text(std::to_string(sel->attn_cols) + " keys x "
                     + std::to_string(sel->attn_rows) + " queries, head "
                     + std::to_string(sel->attn_head)
                     + " (" + std::to_string(sel->n_context_tokens) + " tok)"),
            }));
        }
        metric_rows.push_back(hbox({
            text("  Sparsity (activations): ") | bold,
            gauge(sel->sparsity) | size(WIDTH, EQUAL, 16),
            text("  " + std::to_string(static_cast<int>(sel->sparsity * 100)) + "%"),
        }));
        if (sel->attn_rows > 0 && sel->attn_cols > 0) {
            const float attn_sparse = attn_weight_sparsity(sel);
            metric_rows.push_back(hbox({
                text("  Sparsity (attn weights): ") | bold,
                gauge(attn_sparse) | size(WIDTH, EQUAL, 16),
                text("  " + std::to_string(static_cast<int>(attn_sparse * 100))
                     + "%  (near-zero softmax cells)"),
            }));
        }

        const int slot = layer_slot(sel->layer_index, sel->type);
        float ema = sel->latency_ms;
        auto it = latency_ema.find(slot);
        if (it != latency_ema.end()) ema = it->second;

        const float delta = sel->latency_ms - ema;
        const bool normal = std::abs(delta) < 0.5f;
        metric_rows.push_back(hbox({
            text("  Latency : ") | bold,
            text(float_str(sel->latency_ms) + " ms") | color(Color::Yellow),
            text("  (op delta " + float_str(delta, 3) + " ms)"),
        }));
        metric_rows.push_back(hbox({
            text("  Mean    : ") | bold,
            text(float_str(sel->mean, 4)),
        }));
        metric_rows.push_back(hbox({
            text("  Status  : ") | bold,
            text(normal ? "Within Normal Bounds" : "Elevated")
                | dim,
        }));
        metric_rows.push_back(hbox({
            text("  Max abs : ") | bold,
            text(float_str(sel->max_abs, 4)),
        }));
    } else {
        metric_rows.push_back(text("  Space on topology to set capture target") | dim);
    }
    return vbox(metric_rows);
}

static std::string format_anomaly_message(const LayerSnapshot& s) {
    std::string msg = "[" + format_timestamp_us(s.timestamp_us) + "] ";
    if (s.anomaly_flags & LayerSnapshot::FLAG_HIGH_ACTIVATION) {
        msg += "High activation spike (max=" + float_str(s.max_abs, 2) + ")";
    }
    if (s.anomaly_flags & LayerSnapshot::FLAG_CUDA_FALLBACK) {
        if (s.anomaly_flags & LayerSnapshot::FLAG_HIGH_ACTIVATION) msg += "; ";
        msg += "GPU fallback — running on CPU";
    }
    if (s.anomaly_flags & LayerSnapshot::FLAG_SPARSITY_DROP) {
        if (s.anomaly_flags & (LayerSnapshot::FLAG_HIGH_ACTIVATION |
                               LayerSnapshot::FLAG_CUDA_FALLBACK)) {
            msg += "; ";
        }
        msg += "Sparsity shift (now="
            + std::to_string(static_cast<int>(s.sparsity * 100)) + "%)";
    }
    msg += " — Layer " + std::to_string(s.layer_index);
    return msg;
}

static Element render_anomaly_panel(const std::deque<LayerSnapshot>& stream,
                                    const char* capture_name,
                                    int& scroll, int height) {
    Elements anomaly_rows;
    std::vector<std::string> messages;
    for (const auto& s : stream) {
        if (s.anomaly_flags == 0) continue;
        if (!matches_capture(s, capture_name)) continue;
        messages.push_back(format_anomaly_message(s));
    }

    const int visible = height - 2;
    const int max_scroll = std::max(0, (int)messages.size() - visible);
    scroll = std::min(scroll, max_scroll);
    if (messages.empty()) {
        anomaly_rows.push_back(text("  No anomalies") | dim);
    } else {
        const int end = std::min((int)messages.size(), scroll + visible);
        for (int i = scroll; i < end; i++) {
            const std::string& msg = messages[i];
            const auto ts_end = msg.find(']');
            if (ts_end != std::string::npos) {
                const bool is_alert =
                    messages[i].find("High activation") != std::string::npos ||
                    messages[i].find("GPU fallback") != std::string::npos;
                anomaly_rows.push_back(hbox({
                    text("  " + msg.substr(0, ts_end + 1))
                        | color(Color::Yellow),
                    text(msg.substr(ts_end + 1))
                        | (is_alert ? color(accent_color()) : dim),
                }));
            } else {
                anomaly_rows.push_back(text("  " + msg));
            }
        }
    }

    return vbox(anomaly_rows) | vscroll_indicator | frame |
           size(HEIGHT, EQUAL, height);
}

}  // namespace

TuiApp::TuiApp(RingBuffer& buffer) : buffer_(buffer) {}

void TuiApp::run() {
    running_ = true;

    std::vector<LayerSnapshot> layers;
    std::deque<LayerSnapshot>  stream;
    std::vector<TreeLine>      tree_lines;
    std::unordered_map<int, float> latency_ema;

    int tree_cursor         = 0;
    int tree_scroll         = 0;
    char capture_target_name[64]{};
    int focused_panel       = 0;
    int stream_scroll       = 0;
    int anomaly_scroll      = 0;
    int matrix_pan_row      = 0;
    int matrix_pan_col      = 0;
    float matrix_threshold  = 0.25f;
    bool matrix_fullscreen  = false;

    const int MAX_STREAM = 500;
    constexpr int kTopologyHeight = 14;
    constexpr int kStreamHeight   = 12;
    constexpr int kAnomalyHeight  = 8;

    std::mutex data_mutex;
    StderrSilencer stderr_guard(!verbose_logging());
    auto screen = ScreenInteractive::Fullscreen();
    screen.TrackMouse(false);
    const int term_height = screen.dimy();

    auto input = CatchEvent(Renderer([] { return text(""); }),
        [&](Event event) {
            if (event == Event::Tab) {
                focused_panel = (focused_panel + 1) % 5;
                return true;
            }
            if (event == Event::Character('q') ||
                event == Event::Character('Q')) {
                screen.ExitLoopClosure()();
                return true;
            }
            if (matrix_fullscreen &&
                (event == Event::Character('f') ||
                 event == Event::Character('F') ||
                 event == Event::Escape)) {
                matrix_fullscreen = false;
                return true;
            }

            if (focused_panel == 0) {
                if (event == Event::Character('j')) {
                    if (!tree_lines.empty()) {
                        do {
                            tree_cursor = std::min(
                                tree_cursor + 1, (int)tree_lines.size() - 1);
                        } while (tree_cursor < (int)tree_lines.size() &&
                                 tree_lines[tree_cursor].layer_idx < 0 &&
                                 tree_cursor < (int)tree_lines.size() - 1);
                    }
                    return true;
                }
                if (event == Event::Character('k')) {
                    if (!tree_lines.empty()) {
                        do {
                            tree_cursor = std::max(tree_cursor - 1, 0);
                        } while (tree_cursor > 0 &&
                                 tree_lines[tree_cursor].layer_idx < 0);
                    }
                    return true;
                }
                if (event == Event::Character(' ')) {
                    if (tree_cursor >= 0 &&
                        tree_cursor < (int)tree_lines.size() &&
                        tree_lines[tree_cursor].layer_idx >= 0) {
                        const auto& snap =
                            layers[tree_lines[tree_cursor].layer_idx];
                        std::snprintf(capture_target_name,
                                      sizeof(capture_target_name), "%s",
                                      snap.layer_name);
                    }
                    return true;
                }
            }

            if (focused_panel == 1) {
                const int page = kStreamHeight - 3;
                const int max_scroll = std::max(0, (int)stream.size() - page);
                if (event == Event::Character('j')) {
                    stream_scroll = std::min(stream_scroll + 1, max_scroll);
                    return true;
                }
                if (event == Event::Character('k')) {
                    stream_scroll = std::max(stream_scroll - 1, 0);
                    return true;
                }
            }

            if (focused_panel == 2) {
                const LayerSnapshot* cap =
                    find_layer_by_name(layers, capture_target_name);
                const int max_row = cap && cap->attn_rows > 0
                    ? std::max(0, cap->attn_rows - 1) : 7;
                const int max_col = cap && cap->attn_cols > 0
                    ? std::max(0, cap->attn_cols - 1) : 7;
                if (event == Event::Character('f') ||
                    event == Event::Character('F')) {
                    matrix_fullscreen = !matrix_fullscreen;
                    return true;
                }
                if (event == Event::Character('h')) {
                    matrix_pan_col = std::max(matrix_pan_col - 1, 0);
                    return true;
                }
                if (event == Event::Character('l')) {
                    matrix_pan_col = std::min(matrix_pan_col + 1, max_col);
                    return true;
                }
                if (event == Event::Character('j')) {
                    matrix_pan_row = std::min(matrix_pan_row + 1, max_row);
                    return true;
                }
                if (event == Event::Character('k')) {
                    matrix_pan_row = std::max(matrix_pan_row - 1, 0);
                    return true;
                }
                if (event == Event::Character('+') ||
                    event == Event::Character('=')) {
                    matrix_threshold = std::min(matrix_threshold + 0.05f, 0.9f);
                    return true;
                }
                if (event == Event::Character('-') ||
                    event == Event::Character('_')) {
                    matrix_threshold = std::max(matrix_threshold - 0.05f, 0.05f);
                    return true;
                }
            }

            if (focused_panel == 4) {
                if (event == Event::Character('j')) {
                    anomaly_scroll++;
                    return true;
                }
                if (event == Event::Character('k')) {
                    anomaly_scroll = std::max(anomaly_scroll - 1, 0);
                    return true;
                }
            }

            return false;
        });

    std::thread refresh_thread([&] {
        uint64_t last_ring_drops = 0;
        int      last_stream_decode_step = -1;
        std::unordered_set<std::string> prefill_stream_layers;
        int      last_painted_decode_step = -1;
        bool     pending_redraw = false;
        constexpr int kRefreshMs = 200;
        constexpr int kMinRedrawMs = 500;
        auto last_redraw = std::chrono::steady_clock::now();

        auto request_redraw = [&] {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_redraw >=
                std::chrono::milliseconds(kMinRedrawMs)) {
                screen.Post([&] { screen.PostEvent(Event::Custom); });
                last_redraw = now;
                pending_redraw = false;
            } else {
                pending_redraw = true;
            }
        };

        while (running_) {
            std::vector<LayerSnapshot> incoming;
            while (auto snap = buffer_.pop()) {
                incoming.push_back(*snap);
            }

            if (!incoming.empty()) {
                std::lock_guard<std::mutex> lock(data_mutex);
                const size_t stream_before = stream.size();

                std::unordered_map<int,
                    std::unordered_map<std::string, LayerSnapshot>> step_layers;

                for (const auto& snap : incoming) {
                    bool found = false;
                    for (auto& l : layers) {
                        if (std::strcmp(l.layer_name, snap.layer_name) == 0) {
                            l = snap;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        layers.push_back(snap);
                    }

                    const int slot = layer_slot(snap.layer_index, snap.type);
                    auto& ema = latency_ema[slot];
                    ema = ema > 0.0f
                        ? 0.9f * ema + 0.1f * snap.latency_ms
                        : snap.latency_ms;

                    step_layers[snap.decode_step][snap.layer_name] = snap;
                }

                const auto prefill_it = step_layers.find(0);
                if (prefill_it != step_layers.end()) {
                    std::vector<LayerSnapshot> prefill_new;
                    for (const auto& kv : prefill_it->second) {
                        if (prefill_stream_layers.insert(kv.first).second) {
                            prefill_new.push_back(kv.second);
                        }
                    }
                    if (!prefill_new.empty()) {
                        push_stream_group_front(stream, std::move(prefill_new));
                    }
                }

                std::vector<int> gen_steps;
                for (const auto& kv : step_layers) {
                    if (kv.first > last_stream_decode_step) {
                        gen_steps.push_back(kv.first);
                    }
                }
                std::sort(gen_steps.begin(), gen_steps.end());
                for (int step : gen_steps) {
                    const auto it = step_layers.find(step);
                    if (it == step_layers.end() || it->second.empty()) {
                        continue;
                    }
                    std::vector<LayerSnapshot> step_snaps;
                    step_snaps.reserve(it->second.size());
                    for (const auto& kv : it->second) {
                        step_snaps.push_back(kv.second);
                    }
                    push_stream_group_front(stream, std::move(step_snaps));
                    last_stream_decode_step = step;
                }

                while ((int)stream.size() > MAX_STREAM) {
                    stream.pop_back();
                }

                rebuild_tree(layers, tree_lines);
                if (capture_target_name[0] == '\0') {
                    for (const auto& line : tree_lines) {
                        if (line.layer_idx >= 0 &&
                            layers[line.layer_idx].type ==
                                LayerType::Attention) {
                            std::snprintf(capture_target_name,
                                          sizeof(capture_target_name), "%s",
                                          layers[line.layer_idx].layer_name);
                            break;
                        }
                    }
                }
                if (tree_cursor >= (int)tree_lines.size()) {
                    tree_cursor = std::max(0, (int)tree_lines.size() - 1);
                }

                if (stream.size() != stream_before ||
                    last_stream_decode_step != last_painted_decode_step) {
                    last_painted_decode_step = last_stream_decode_step;
                    request_redraw();
                }
            }

            if (pending_redraw) {
                request_redraw();
            }

            const uint64_t drops = buffer_.dropped_count();
            if (drops > last_ring_drops) {
                std::fprintf(stderr,
                    "[TUI] ring buffer dropped %llu snapshot(s) "
                    "(total drops: %llu)\n",
                    static_cast<unsigned long long>(drops - last_ring_drops),
                    static_cast<unsigned long long>(drops));
                last_ring_drops = drops;
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(kRefreshMs));
        }
    });

    auto renderer = Renderer(input, [&] {
        std::lock_guard<std::mutex> lock(data_mutex);

        if (tree_lines.empty()) {
            rebuild_tree(layers, tree_lines);
        }

        const LayerSnapshot* capture =
            find_layer_by_name(layers, capture_target_name);

        if (matrix_fullscreen) {
            return vbox({
                render_matrix_panel(capture, matrix_pan_row, matrix_pan_col,
                                    matrix_threshold, true, true),
                separator(),
                hbox({
                    text("  [F/Esc] Exit  ") | bold | dim,
                    text("[hjkl] Pan  ") | dim,
                    text("[+/-] Contrast  ") | dim,
                    text("[Q] Quit") | dim,
                }),
            });
        }

        const bool f0 = (focused_panel == 0);
        const bool f1 = (focused_panel == 1);
        const bool f2 = (focused_panel == 2);
        const bool f3 = (focused_panel == 3);
        const bool f4 = (focused_panel == 4);

        auto panel1 = styled_panel(
            "1. MODEL TOPOLOGY",
            render_topology_panel(tree_lines, tree_cursor, capture_target_name,
                                  layers, tree_scroll, kTopologyHeight),
            f0) | size(WIDTH, EQUAL, 36) | size(HEIGHT, EQUAL, kTopologyHeight);

        auto panel2 = styled_panel(
            "2. LIVE STREAM",
            render_stream_panel(stream, capture_target_name, stream_scroll,
                                kStreamHeight),
            f1) | flex | size(HEIGHT, EQUAL, kTopologyHeight);

        auto panel3 = render_matrix_panel(capture, matrix_pan_row, matrix_pan_col,
                                          matrix_threshold, false, f2);

        auto panel4 = styled_panel(
            "4. RUNTIME METRICS",
            render_metrics_panel(capture, latency_ema),
            f3) | flex;

        auto panel5 = styled_panel(
            "5. ANOMALY LEDGER",
            render_anomaly_panel(stream, capture_target_name, anomaly_scroll,
                                 kAnomalyHeight),
            f4) | flex | size(HEIGHT, EQUAL, kAnomalyHeight);

        std::string capture_label = "none";
        if (capture_target_name[0]) {
            capture_label = capture_target_name;
        }

        static const char* kPanelNames[] = {
            "Topology", "Stream", "Matrix", "Metrics", "Anomalies"
        };

        auto header = hbox({
            text(" llm-instrument ") | bold,
            text("  layer telemetry  ") | dim,
            filler(),
            text(padded_event_count(stream.size())) | dim,
        });

        auto status = hbox({
            text("  [Tab] ") | dim,
            text(kPanelNames[focused_panel]) | bold,
            text("  [j/k] Nav  [Space] Capture  [F] Matrix  [Q] Quit  ") | dim,
            filler(),
            text("Target: " + capture_label) | dim,
        });

        return vbox({
            header,
            separator(),
            hbox({panel1, panel2}),
            panel3,
            hbox({panel4, panel5}),
            separator(),
            status,
        }) | size(HEIGHT, EQUAL, term_height);
    });

    screen.Loop(renderer);
    running_ = false;
    refresh_thread.join();
}

void TuiApp::stop() { running_ = false; }
