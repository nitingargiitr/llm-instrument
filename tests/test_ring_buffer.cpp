#include "core/ring_buffer.h"
#include <iostream>
#include <cassert>
#include <cstring>

int main() {
    std::cout << "Starting tests..." << std::endl;

    // Test 1: empty buffer returns nullopt
    {
        RingBuffer rb;
        auto result = rb.pop();
        assert(!result.has_value());
        std::cout << "PASS: empty buffer returns nullopt" << std::endl;
    }

    // Test 2: push and pop one item
    {
        RingBuffer rb;
        LayerSnapshot s;
        memset(&s, 0, sizeof(s));
        s.sequence_id = 42;
        rb.push(s);
        auto p = rb.pop();
        assert(p.has_value());
        assert(p->sequence_id == 42);
        std::cout << "PASS: push and pop one item" << std::endl;
    }

    // Test 3: FIFO order
    {
        RingBuffer rb;
        for (int i = 0; i < 5; i++) {
            LayerSnapshot s;
            memset(&s, 0, sizeof(s));
            s.sequence_id = (uint32_t)i;
            rb.push(s);
        }
        for (int i = 0; i < 5; i++) {
            auto s = rb.pop();
            assert(s.has_value());
            assert(s->sequence_id == (uint32_t)i);
        }
        std::cout << "PASS: FIFO order preserved" << std::endl;
    }

    // Test 4: overfill does not crash
    {
        RingBuffer rb;
        LayerSnapshot s;
        memset(&s, 0, sizeof(s));
        for (int i = 0; i < 20; i++) rb.push(s);
        std::cout << "PASS: overfill does not crash" << std::endl;
    }

    // Test 5: available() count
    {
        RingBuffer rb;
        LayerSnapshot s;
        memset(&s, 0, sizeof(s));
        rb.push(s);
        rb.push(s);
        rb.push(s);
        assert(rb.available() == 3);
        std::cout << "PASS: available() returns correct count" << std::endl;
    }

    std::cout << "\nAll tests passed." << std::endl;
    return 0;
}