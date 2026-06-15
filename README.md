# LLM-Instrument — Real-Time Transformer Telemetry TUI

*Non-invasive C++ hooking for local GGUF models — live activations, attention maps, and layer latency in the terminal.*

**LLM-Instrument** is aimed at developers who want to **see inside a local GGUF model while it runs**: which layer is hot, what shape left the MLP, what the attention map looks like for the current context, and whether anything looks anomalous—all from the terminal with vim-style keys. It hooks the forward pass via [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `cb_eval` callback without modifying model source.

---

## Why this project?

Many existing tools overlap partially with what LLM-Instrument does, but few combine **all** of the following in one **offline, C++ native** package:

| Approach | Typical limitation |
|----------|-------------------|
| Web UIs / hosted dashboards (Weights & Biases, TensorBoard, etc.) | Cloud or Python stack; not tied to a local GGUF inference run |
| Static graph viewers (Netron, ONNX tools) | Architecture only—no live activations or per-token attention |
| Python `forward` hooks in PyTorch/Hugging Face | Requires the training framework and often model code changes |
| llama.cpp server logs / bench tools | Throughput-focused; no per-layer activation stats or attention heatmaps |

---

## Features

### Non-invasive hooking
- Uses llama.cpp's `cb_eval` callback to observe tensors during `llama_decode`
- No patches to model weights or llama.cpp inference logic beyond the registered callback
- Hooks: embeddings, attention output, MLP output, layer norms, and **`kq_soft_max`** (real softmax attention weights)

### Real-time capture (prefill + generation)
- **Prefill**: full prompt pass with `decode_step = 0`
- **Generation**: up to 48 tokens at ~450 ms/token so the live stream is readable in the TUI
- Ring buffer (256 slots) decouples inference from UI refresh

### Five-panel interactive TUI ([FTXUI](https://github.com/ArthurSonzogni/FTXUI))

| Panel | Contents |
|-------|----------|
| **1. Model topology** | Tree of layers; set **capture target** with Space |
| **2. Live stream** | Timestamped events for all layers; capture target highlighted |
| **3. Attention weights** | ASCII softmax heatmap with BPE token labels; pan & fullscreen |
| **4. Runtime metrics** | Shape (activation), dtype, latency (op time), mean, max; **Sparsity (activations)** and **Sparsity (attn weights)** on attention layers |
| **5. Anomaly ledger** | High activation, device fallback, sparsity shift flags |

### Metrics & anomalies
- **Latency**: GPU/CPU op time via `ggml_time_us` (not host readback overhead)
- **Activations**: mean, max abs; **Sparsity (activations)** — sampled up to 4096 elements from the layer output tensor
- **Attention**: merged N×N prefill grid + per-token decode rows (cap 64×64)
- **Sparsity (attn weights)**: fraction of near-zero cells in the softmax map (shown alongside activation sparsity on attention layers)
- **Shape (activation)**: tensor dimensions of the hooked output (e.g. `attn_out`), separate from the attention map size
- **Anomalies**: spike detection, CPU fallback vs Metal/CUDA, sparsity drift

### Simulate mode
- Run the full TUI **without a GGUF file** for CI, grading, or quick UI checks

---

## Requirements

- **CMake** ≥ 3.16  
- **C++17** compiler (Clang, GCC, or MSVC)  
- **Ninja** or Make  
- **macOS**: Metal (default GPU backend in llama.cpp)  
- **Linux**: CPU; CUDA if built with GPU support in llama.cpp  
- **Terminal**: ≥ 120×40 characters recommended for the full layout  

Vendored dependencies (included under `third_party/`):
- [llama.cpp](https://github.com/ggml-org/llama.cpp) + GGML  
- [FTXUI](https://github.com/ArthurSonzogni/FTXUI)

---

## Build

```bash
git clone https://github.com/nitingargiitr/llm-instrument.git
cd llm-instrument

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The executable is `build/src/llm_instrument`.

### Run unit tests

```bash
./build/tests/test_ring_buffer
./build/tests/test_hook_manager
./build/tests/test_metrics
```

All three should print `All tests passed` (or equivalent).

---

## Run

With **no arguments**, the binary defaults to **simulate mode** with prompt `"Hello"`:

```bash
./build/src/llm_instrument
```

### Option A — Simulate (no model file)

Fastest way to verify the TUI and hook pipeline:

```bash
./build/src/llm_instrument --simulate --prompt "Hello world"
```

### Option B — Real GGUF model

Download a small instruct model, e.g. [SmolLM2-360M-Instruct Q4_K_M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF), into `models/`:

```bash
mkdir -p models
# place SmolLM2-360M-Instruct-Q4_K_M.gguf in models/
```

Run:

```bash
./build/src/llm_instrument \
  --model models/SmolLM2-360M-Instruct-Q4_K_M.gguf \
  --prompt "Explain how self-attention works in transformers. Each token looks at earlier tokens in the sequence to decide what to focus on."
```

**First launch on macOS** may take ~30 seconds while Metal shaders compile; wait until the live stream populates.

Press **`q`** to quit.

### CLI reference

```
Usage:
  llm_instrument                       Simulate mode, prompt "Hello" (default)
  llm_instrument [--simulate]          Run with simulated model data
  llm_instrument --model <path.gguf>   Run with real llama.cpp model
  llm_instrument --prompt "text"       Prompt text (simulate and model)
  llm_instrument --help | -h           Print usage and exit

Environment:
  LLM_INSTRUMENT_VERBOSE=1             Show model-runner stderr during TUI
                                       (default: stderr silenced while UI runs)
```

---

## TUI keybindings

| Key | Context | Action |
|-----|---------|--------|
| `Tab` | Global | Cycle focused panel (1→5) |
| `j` / `k` | Topology | Move layer cursor |
| `Space` | Topology | Set **capture target** (panels 3–5 show this layer; stream highlights it) |
| `j` / `k` | Stream / Anomalies | Scroll |
| `h` `j` `k` `l` | Matrix | Pan attention window |
| `F` | Matrix | Toggle fullscreen |
| `+` / `-` | Matrix | Contrast threshold |
| `q` | Global | Quit |

**Tip:** select an **`attn`** layer (e.g. `layers.0.attn`), not `attn_norm`, then press **Space** to see the attention heatmap in panel 3.

---

## Architecture (brief)

```
llama_decode()
    └── cb_eval (model_runner.cpp)
            ├── extract kq_soft_max → HookManager::set_attn_weights
            └── capture_layer (activations, latency, shapes)
                    └── RingBuffer
                            └── TUI refresh thread → FTXUI panels
```

- **Thread-safe**: mutex between refresh thread and renderer  
- **Capture target**: metrics, matrix, and anomaly ledger use the layer chosen in topology; stream highlights it

---

## Assumptions & limitations

These are intentional trade-offs for a lightweight demo tool:

| Topic | Assumption |
|-------|------------|
| **Models** | GGUF via llama.cpp; chat template applied when the model provides one |
| **Attention** | Head **0** only; matrix capped at **64×64** (recent context window) |
| **Token labels** | Last **16** BPE pieces stored; older positions use `tN` fallbacks |
| **Activation stats** | Subsampled to **4096** elements for mean / max / sparsity |
| **Generation** | Bounded (**48** tokens max, **16** min before EOS); then UI holds for inspection |
| **Stream UI** | Last **500** events shown; ring buffer may drop under heavy load |
| **Devices** | Metal on Apple Silicon; CUDA label for NVIDIA; CPU fallback detected |
| **Flash attention** | Disabled so `kq_soft_max` tensors are materialized and hookable |

---

## Project layout

```
include/          Public headers (snapshot, ring buffer, hooks, TUI)
src/
  core/           Ring buffer, metrics, snapshot printing
  hooks/          HookManager + llama.cpp model runner / cb_eval
  tui/            FTXUI application
tests/            Unit tests
third_party/      llama.cpp, FTXUI (vendored)
models/           Place .gguf files here (gitignored)
```

---

## License

Third-party code retains its own licenses (llama.cpp, FTXUI). See respective directories under `third_party/`.
