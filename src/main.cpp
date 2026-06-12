#include <iostream>
#include <thread>
#include <chrono>
#include "core/ring_buffer.h"
#include "tui/tui_app.h"
#include "../tests/mock_producer.h"

int main() {
    std::cout << "llm-instrument starting...\n";

    RingBuffer ring_buffer;
    TuiApp     tui(ring_buffer);

    bool mock_running = true;
    std::thread mock_thread(
        run_mock_producer,
        std::ref(ring_buffer),
        std::ref(mock_running)
    );

    tui.run();

    mock_running = false;
    mock_thread.join();
    return 0;
}
