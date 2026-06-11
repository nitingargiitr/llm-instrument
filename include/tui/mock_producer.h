#pragma once
#include "core/ring_buffer.h"

LayerSnapshot make_fake_snapshot(uint32_t seq_id, int layer_idx);
void run_mock_producer(RingBuffer& buffer, bool& running);
