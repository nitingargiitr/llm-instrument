#pragma once
#include "core/ring_buffer.h"

class HookManager {
public:
    explicit HookManager(RingBuffer& buffer);
    void install();    // register hooks at startup
    void uninstall();  // remove hooks at shutdown

private:
    RingBuffer& buffer_;
    bool installed_ = false;
};