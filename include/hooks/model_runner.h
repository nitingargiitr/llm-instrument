#pragma once

#include "hooks/hook_manager.h"
#include <string>

// Run real llama.cpp inference and capture layer snapshots via cb_eval.
void run_model(HookManager& hooks,
               const std::string& model_path,
               const std::string& prompt,
               bool& running);

// Simulated model loop (development / fallback when no model file).
void run_simulate(HookManager& hooks, bool& running,
                  const std::string& prompt = "Hello");
