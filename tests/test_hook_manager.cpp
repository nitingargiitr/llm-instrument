#include "core/ring_buffer.h"
#include "hooks/hook_manager.h"
#include "core/snapshot.h"

#include <iostream>
#include <cmath>

int main() {

    RingBuffer ring_buffer;
    HookManager hooks(ring_buffer);

    hooks.install();

    float fake_data[128];

    for (int i = 0; i < 128; i++) {
        fake_data[i] = std::sin(i * 0.3f) * 4.0f;
    }

    for (int layer = 0; layer < 8; layer++) {

        LayerType type;

        switch (layer % 4) {
            case 0:
                type = LayerType::Embedding;
                break;

            case 1:
                type = LayerType::Attention;
                break;

            case 2:
                type = LayerType::MLP;
                break;

            default:
                type = LayerType::LayerNorm;
                break;
        }

        ComputeDevice dev =
            (layer % 4 == 3)
                ? ComputeDevice::CPU
                : ComputeDevice::CUDA;

        hooks.capture_layer(
            layer,
            type,
            dev,
            fake_data,
            128,
            0.8f + layer * 0.05f
        );
    }

    std::cout
        << "Captured "
        << ring_buffer.available()
        << " snapshots\n";

    int printed = 0;

    while (printed < 5) {

        auto snap = ring_buffer.pop();

        if (!snap)
            break;

        print_snapshot(*snap);
        printed++;
    }

    hooks.uninstall();

    return 0;
}