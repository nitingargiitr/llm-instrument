#include "core/snapshot.h"
#include <iostream>

void print_snapshot(const LayerSnapshot& snap) {
    std::cout << "[seq=" << snap.sequence_id
              << " layer=" << snap.layer_index
              << " latency=" << snap.latency_ms << "ms"
              << " sparsity=" << snap.sparsity
              << " max_abs=" << snap.max_abs
              << " flags=" << snap.anomaly_flags
              << "]\n";
}