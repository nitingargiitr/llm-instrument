#include <iostream>
#include <string>
#include <thread>
#include "core/ring_buffer.h"
#include "hooks/hook_manager.h"
#include "hooks/model_runner.h"
#include "tui/tui_app.h"

static void print_usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " [--simulate]          Run with simulated model data\n"
        << "  " << prog << " --model <path.gguf>   Run with real llama.cpp model\n"
        << "  " << prog << " --prompt \"text\"       Prompt for model mode (optional)\n";
}

int main(int argc, char** argv) {
    std::cout << "llm-instrument starting...\n";

    bool simulate = true;
    std::string model_path;
    std::string prompt = "Hello";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--simulate") {
            simulate = true;
        } else if (arg == "--model" && i + 1 < argc) {
            simulate = false;
            model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    RingBuffer   ring_buffer;
    HookManager  hooks(ring_buffer);
    TuiApp       tui(ring_buffer);

    hooks.install();

    bool model_running = true;
    std::thread model_thread([&] {
        if (simulate) {
            std::cout << "[main] running in simulate mode\n";
            run_simulate(hooks, model_running);
        } else {
            std::cout << "[main] running model: " << model_path << "\n";
            run_model(hooks, model_path, prompt, model_running);
        }
    });

    tui.run();

    model_running = false;
    hooks.uninstall();
    model_thread.join();
    return 0;
}
