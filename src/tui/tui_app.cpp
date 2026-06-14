#include "tui/tui_app.h"
#include "core/snapshot.h"
#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include <vector>
#include <deque>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_set>

using namespace ftxui;

static std::string layer_type_str(LayerType t) {
    switch (t) {
        case LayerType::Embedding: return "Embed";
        case LayerType::Attention: return "Attn ";
        case LayerType::MLP:       return "MLP  ";
        case LayerType::LayerNorm: return "Norm ";
        default:                   return "???  ";
    }
}

static std::string device_str(ComputeDevice d) {
    return d == ComputeDevice::CUDA ? "CUDA" : "CPU ";
}

static std::string float_str(float v, int decimals = 2) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return std::string(buf);
}

static Element render_attention_panel(const LayerSnapshot* sel) {
    Elements attn_rows;
    constexpr int kMatrixRows = 8;

    if (sel && sel->attn_rows > 0 &&
        sel->type == LayerType::Attention) {
        for (int r = 0; r < sel->attn_rows; r++) {
            std::string row_str;
            row_str.reserve(sel->attn_cols);
            for (int c = 0; c < sel->attn_cols; c++) {
                float v = sel->attn_matrix[r][c];
                if      (v > 0.75f) row_str += '#';
                else if (v > 0.50f) row_str += '+';
                else if (v > 0.25f) row_str += '-';
                else if (v > 0.10f) row_str += '.';
                else                row_str += ' ';
            }
            attn_rows.push_back(text("  " + row_str));
        }
        while ((int)attn_rows.size() < kMatrixRows) {
            attn_rows.push_back(text(""));
        }
    } else {
        attn_rows.push_back(
            text("  Select an Attention layer (j/k)") | dim);
        while ((int)attn_rows.size() < kMatrixRows) {
            attn_rows.push_back(text(""));
        }
    }

    return window(
        text(" 3. ATTENTION MATRIX ") | bold,
        vbox(attn_rows) | size(HEIGHT, EQUAL, kMatrixRows));
}

TuiApp::TuiApp(RingBuffer& buffer) : buffer_(buffer) {}

void TuiApp::run() {
    running_ = true;
    std::vector<LayerSnapshot> layers;
    std::deque<LayerSnapshot>  stream;
    std::vector<std::string>   layer_labels;
    int menu_selected = 0;
    int focused_panel = 0;
    const int MAX_STREAM = 50;
    constexpr int kTopologyHeight = 14;
    constexpr int kStreamHeight     = 12;
    constexpr int kAnomalyHeight    = 8;
    std::mutex data_mutex;
    auto screen = ScreenInteractive::Fullscreen();
    auto layer_menu = Menu(&layer_labels, &menu_selected);
    auto main_component = CatchEvent(layer_menu, [&](Event event) {
        if (event == Event::Tab) {
            focused_panel = (focused_panel + 1) % 2;
            return true;
        }
        if (event == Event::Character('q') ||
            event == Event::Character('Q')) {
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });
    std::thread refresh_thread([&] {
        while (running_) {
            bool updated = false;
            while (auto snap = buffer_.pop()) {
                updated = true;
                std::lock_guard<std::mutex> lock(data_mutex);
                bool found = false;
                for (auto& l : layers) {
                    if (l.layer_index == snap->layer_index &&
                        l.type == snap->type) {
                        l = *snap; found = true; break;
                    }
                }
                if (!found) {
                    layers.push_back(*snap);
                    layer_labels.push_back(
                        "Layer " +
                        std::to_string(snap->layer_index) +
                        "  [" + layer_type_str(snap->type) + "]");
                }
                stream.push_front(*snap);
                if ((int)stream.size() > MAX_STREAM)
                    stream.pop_back();
            }
            if (updated) {
                screen.PostEvent(Event::Custom);
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(100));
        }
    });
    auto renderer = Renderer(main_component, [&] {
        std::lock_guard<std::mutex> lock(data_mutex);

        LayerSnapshot selected_snap{};
        const LayerSnapshot* sel = nullptr;
        if (menu_selected >= 0 &&
            menu_selected < (int)layers.size()) {
            selected_snap = layers[menu_selected];
            sel = &selected_snap;
        }
        auto panel1 = window(
            text(" 1. MODEL TOPOLOGY ") | bold,
            layer_menu->Render() | vscroll_indicator | frame |
                size(HEIGHT, EQUAL, kTopologyHeight)
        ) | size(WIDTH, EQUAL, 36) | size(HEIGHT, EQUAL, kTopologyHeight);

        Elements stream_rows;
        stream_rows.push_back(hbox({
            text(" ID  ") | bold | size(WIDTH, EQUAL, 6),
            text(" LYR ") | bold | size(WIDTH, EQUAL, 5),
            text(" TYPE ") | bold | size(WIDTH, EQUAL, 7),
            text(" DEV ") | bold | size(WIDTH, EQUAL, 6),
            text(" MS  ") | bold | size(WIDTH, EQUAL, 7),
        }));
        stream_rows.push_back(separator());
        for (auto& s : stream) {
            stream_rows.push_back(hbox({
                text(" "+std::to_string(s.sequence_id))
                    | size(WIDTH, EQUAL, 6),
                text(" "+std::to_string(s.layer_index))
                    | size(WIDTH, EQUAL, 5),
                text(" "+layer_type_str(s.type))
                    | size(WIDTH, EQUAL, 7),
                text(" "+device_str(s.device))
                    | size(WIDTH, EQUAL, 6),
                text(" "+float_str(s.latency_ms))
                    | size(WIDTH, EQUAL, 7),
            }));
        }
        auto panel2 = window(
            text(" 2. LIVE STREAM ") | bold,
            vbox(stream_rows) | vscroll_indicator | frame |
                size(HEIGHT, EQUAL, kStreamHeight)
        ) | flex | size(HEIGHT, EQUAL, kTopologyHeight);

        auto panel3 = render_attention_panel(sel);

        Elements metric_rows;
        if (sel) {
            metric_rows.push_back(hbox({
                text("  Shape   : ") | bold,
                text("["+std::to_string(sel->shape[0])+", "
                    +std::to_string(sel->shape[1])+", "
                    +std::to_string(sel->shape[2])+"]"),
            }));
            metric_rows.push_back(hbox({
                text("  Dtype   : ") | bold,
                text(std::string(sel->dtype)),
            }));
            metric_rows.push_back(hbox({
                text("  Sparsity: ") | bold,
                gauge(sel->sparsity) | size(WIDTH, EQUAL, 16),
                text("  "+std::to_string(
                    (int)(sel->sparsity*100))+"%"),
            }));
            metric_rows.push_back(hbox({
                text("  Latency : ") | bold,
                text(float_str(sel->latency_ms)+" ms"),
            }));
            metric_rows.push_back(hbox({
                text("  Max abs : ") | bold,
                text(float_str(sel->max_abs, 4)),
            }));
        } else {
            metric_rows.push_back(
                text("  Select a layer") | dim);
        }
        auto panel4 = window(
            text(" 4. RUNTIME METRICS ") | bold,
            vbox(metric_rows)) | flex;

        Elements anomaly_rows;
        std::unordered_set<int> shown_layers;
        for (auto& s : stream) {
            if (s.anomaly_flags == 0) continue;
            if (!shown_layers.insert(s.layer_index).second) continue;

            std::string msg = "  Layer " +
                std::to_string(s.layer_index) + ": ";
            if (s.anomaly_flags &
                LayerSnapshot::FLAG_HIGH_ACTIVATION)
                msg += "[HIGH ACT] ";
            if (s.anomaly_flags &
                LayerSnapshot::FLAG_CUDA_FALLBACK)
                msg += "[CUDA FALLBACK] ";
            if (s.anomaly_flags &
                LayerSnapshot::FLAG_SPARSITY_DROP)
                msg += "[SPARSITY DROP] ";
            anomaly_rows.push_back(
                text(msg) | color(Color::RedLight));
        }
        if (anomaly_rows.empty())
            anomaly_rows.push_back(
                text("  No anomalies") | dim);
        auto panel5 = window(
            text(" 5. ANOMALY LEDGER ") | bold,
            vbox(anomaly_rows) | vscroll_indicator | frame |
                size(HEIGHT, EQUAL, kAnomalyHeight)
        ) | flex | size(HEIGHT, EQUAL, kAnomalyHeight);

        auto status = hbox({
            text("  [Tab] Switch panel  ") | bold,
            text("[j/k] Select layer  ") | bold,
            text("[Q] Quit  ") | bold,
            filler(),
            text("Focus: Panel " + std::to_string(
                focused_panel + 1)) | bold | color(Color::Cyan),
        });

        auto main_panels = vbox({
            hbox({ panel1, panel2 }),
            panel3,
            hbox({ panel4, panel5 }),
        }) | flex;

        return vbox({
            main_panels,
            separator(),
            status,
        });
    });
    screen.Loop(renderer);
    running_ = false;
    refresh_thread.join();
}

void TuiApp::stop() { running_ = false; }
