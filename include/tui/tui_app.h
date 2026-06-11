#pragma once
#include "core/ring_buffer.h"

class TuiApp {
public:
    explicit TuiApp(RingBuffer& buffer);
    void run();   // starts the TUI render loop, blocks until user quits
    void stop();  // signals the render loop to exit

private:
    RingBuffer& buffer_;
    bool running_ = false;
};