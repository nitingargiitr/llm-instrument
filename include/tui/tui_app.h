#pragma once
#include "core/ring_buffer.h"

class TuiApp {
public:
    explicit TuiApp(RingBuffer& buffer);
    void run();
    void stop();

private:
    RingBuffer& buffer_;
    bool running_ = false;
};
