#include "hooks/model_runner.h"
#include "llama.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <array>

namespace {

static std::string display_model_name(const llama_model* model,
                                      const std::string& model_path) {
    char meta[64]{};
    if (llama_model_meta_val_str(model, "general.name", meta, sizeof(meta)) > 0 &&
        meta[0] != '\0') {
        return meta;
    }

    const size_t slash = model_path.find_last_of("/\\");
    std::string base =
        slash == std::string::npos ? model_path : model_path.substr(slash + 1);
    constexpr const char* kGguf = ".gguf";
    if (base.size() > std::strlen(kGguf) &&
        base.compare(base.size() - std::strlen(kGguf), std::strlen(kGguf),
                     kGguf) == 0) {
        base.resize(base.size() - std::strlen(kGguf));
    }
    if (!base.empty()) {
        return base;
    }

    char desc[64]{};
    llama_model_desc(model, desc, sizeof(desc));
    return desc[0] != '\0' ? std::string(desc) : "model";
}

static float ggml_value_at(const uint8_t* data, ggml_type type,
                           const size_t* nb,
                           int64_t i0, int64_t i1,
                           int64_t i2, int64_t i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    switch (type) {
        case GGML_TYPE_F32:
            return *(const float*)&data[i];
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t*)&data[i]);
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*(const ggml_bf16_t*)&data[i]);
        default:
            return 0.0f;
    }
}

static int tensor_to_floats(ggml_tensor* t, std::vector<float>& out, int max_count) {
    std::vector<uint8_t> host;
    const uint8_t* data = nullptr;

    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = static_cast<const uint8_t*>(t->data);
    } else {
        host.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, host.data(), 0, host.size());
        data = host.data();
    }

    const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    const size_t  nb[4] = {t->nb[0], t->nb[1], t->nb[2], t->nb[3]};
    const int64_t total = ne[0] * ne[1] * ne[2] * ne[3];
    if (total <= 0) return 0;

    const int stride = total > max_count ? static_cast<int>(total / max_count) : 1;
    out.clear();
    out.reserve(max_count);

    int64_t idx = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (idx % stride == 0) {
                        out.push_back(ggml_value_at(data, t->type, nb, i0, i1, i2, i3));
                        if (static_cast<int>(out.size()) >= max_count) {
                            return static_cast<int>(out.size());
                        }
                    }
                    idx++;
                }
            }
        }
    }
    return static_cast<int>(out.size());
}

struct LlamaInstrumentData {
    HookManager* hooks = nullptr;
    int64_t      op_start_us = 0;
    int          n_layer = 0;
    int          attn_head = 0;
    int          n_kv_active = 0;
    std::array<bool, 128> attn_diag_logged{};

    void reset_attn_diag() { attn_diag_logged.fill(false); }
};

static ComputeDevice detect_device(ggml_tensor* t) {
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return ComputeDevice::CPU;
    }
    const char* name = ggml_backend_buffer_name(t->buffer);
    if (!name) return ComputeDevice::CUDA;
    if (std::strstr(name, "Metal") || std::strstr(name, "MTL")) {
        return ComputeDevice::Metal;
    }
    if (std::strstr(name, "CUDA") || std::strstr(name, "ROCm") ||
        std::strstr(name, "SYCL") || std::strstr(name, "Vulkan")) {
        return ComputeDevice::CUDA;
    }
    return ComputeDevice::CUDA;
}

static const uint8_t* tensor_host_data(ggml_tensor* t, std::vector<uint8_t>& host) {
    if (ggml_backend_buffer_is_host(t->buffer)) {
        return static_cast<const uint8_t*>(t->data);
    }
    host.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, host.data(), 0, host.size());
    return host.data();
}

static bool extract_attn_weights(ggml_tensor* t, int head_idx, int n_kv_active,
                                 float out[][LayerSnapshot::ATTN_CAP],
                                 int& out_rows, int& out_cols) {
    std::vector<uint8_t> host;
    const uint8_t* data = tensor_host_data(t, host);
    if (!data) return false;

    const int64_t n_kv_padded = t->ne[0];
    const int64_t n_q         = t->ne[1];
    const int64_t n_heads     = t->ne[2];
    if (n_kv_padded <= 0 || n_q <= 0 || n_heads <= 0) return false;

    const int n_kv = std::min(n_kv_active, static_cast<int>(n_kv_padded));
    if (n_kv <= 0) return false;

    const int head = std::max(0, std::min(head_idx, static_cast<int>(n_heads) - 1));
    out_rows = static_cast<int>(std::min<int64_t>(LayerSnapshot::ATTN_CAP, n_q));
    out_cols = static_cast<int>(std::min<int64_t>(LayerSnapshot::ATTN_CAP, n_kv));

    const int q_start  = static_cast<int>(n_q) - out_rows;
    const int kv_start = n_kv - out_cols;
    const size_t nb[4] = {t->nb[0], t->nb[1], t->nb[2], t->nb[3]};

    for (int r = 0; r < out_rows; r++) {
        for (int c = 0; c < out_cols; c++) {
            const int64_t q  = q_start + r;
            const int64_t kv = kv_start + c;
            out[r][c] = ggml_value_at(data, t->type, nb, kv, q, head, 0);
        }
    }
    return true;
}

static bool is_interested_tensor(const char* name) {
    return std::strcmp(name, "inp_embd") == 0 ||
           std::strcmp(name, "result_norm") == 0 ||
           std::strstr(name, "kq_soft_max-") != nullptr ||
           std::strstr(name, "attn_out-") != nullptr ||
           std::strstr(name, "ffn_out-") != nullptr ||
           std::strstr(name, "attn_norm-") != nullptr ||
           std::strstr(name, "ffn_norm-") != nullptr;
}

static bool is_attn_weights_tensor(const char* name) {
    return std::strncmp(name, "kq_soft_max", 11) == 0;
}

static bool parse_tensor(const char* name, int n_layer,
                         int& layer_index, LayerType& type,
                         char* layer_name, size_t layer_len) {
    if (std::strcmp(name, "inp_embd") == 0) {
        layer_index = 0;
        type = LayerType::Embedding;
        std::snprintf(layer_name, layer_len, "embed_tokens");
        return true;
    }
    if (std::strcmp(name, "result_norm") == 0) {
        layer_index = n_layer > 0 ? n_layer : 0;
        type = LayerType::LayerNorm;
        std::snprintf(layer_name, layer_len, "result_norm");
        return true;
    }

    const char* dash = std::strrchr(name, '-');
    if (!dash) return false;

    layer_index = std::atoi(dash + 1);
    if (std::strncmp(name, "attn_out", 8) == 0) {
        type = LayerType::Attention;
        std::snprintf(layer_name, layer_len, "layers.%d.attn", layer_index);
        return true;
    }
    if (std::strncmp(name, "ffn_out", 7) == 0) {
        type = LayerType::MLP;
        std::snprintf(layer_name, layer_len, "layers.%d.mlp", layer_index);
        return true;
    }
    if (std::strncmp(name, "attn_norm", 9) == 0) {
        type = LayerType::LayerNorm;
        std::snprintf(layer_name, layer_len, "layers.%d.attn_norm",
                     layer_index);
        return true;
    }
    if (std::strncmp(name, "ffn_norm", 8) == 0) {
        type = LayerType::LayerNorm;
        std::snprintf(layer_name, layer_len, "layers.%d.ffn_norm",
                     layer_index);
        return true;
    }
    return false;
}

static bool llama_eval_callback(ggml_tensor* t, bool ask, void* user_data) {
    auto* data = static_cast<LlamaInstrumentData*>(user_data);
    if (!data || !data->hooks) return true;

    if (ask) {
        const bool interested = is_interested_tensor(t->name);
        if (interested) {
            data->op_start_us = ggml_time_us();
        }
        return interested;
    }

    if (is_attn_weights_tensor(t->name)) {
        const char* dash = std::strrchr(t->name, '-');
        if (!dash) return true;
        const int layer_index = std::atoi(dash + 1);

        float weights[LayerSnapshot::ATTN_CAP][LayerSnapshot::ATTN_CAP]{};
        int rows = 0;
        int cols = 0;
        if (extract_attn_weights(t, data->attn_head, data->n_kv_active,
                                 weights, rows, cols)) {
            int wshape[4] = {
                static_cast<int>(t->ne[0]),
                static_cast<int>(t->ne[1]),
                static_cast<int>(t->ne[2]),
                static_cast<int>(t->ne[3]),
            };
            int wndim = 1;
            for (int d = 3; d >= 0; d--) {
                if (t->ne[d] > 0) { wndim = d + 1; break; }
            }
            if (layer_index >= 0 && layer_index < (int)data->attn_diag_logged.size() &&
                !data->attn_diag_logged[layer_index]) {
                data->attn_diag_logged[layer_index] = true;
                std::fprintf(stderr,
                    "[AttnDiag] Tensor: %s\n"
                    "           Original shape: [%d,%d,%d,%d]\n"
                    "           Extracted rows: %d  cols: %d  head: %d\n"
                    "           n_kv_active: %d\n",
                    t->name,
                    wshape[0], wshape[1], wshape[2], wshape[3],
                    rows, cols, data->attn_head,
                    data->n_kv_active);
            }
            data->hooks->set_attn_weights(layer_index, data->attn_head,
                                          weights, rows, cols,
                                          data->n_kv_active,
                                          wshape, wndim);
        }
        return true;
    }

    int layer_index = 0;
    LayerType type = LayerType::Unknown;
    char layer_name[64]{};
    if (!parse_tensor(t->name, data->n_layer, layer_index, type,
                      layer_name, sizeof(layer_name))) {
        return true;
    }

    const float latency_ms =
        static_cast<float>(ggml_time_us() - data->op_start_us) / 1000.0f;

    std::vector<float> samples;
    const int count = tensor_to_floats(t, samples, 4096);
    if (count <= 0) return true;

    int shape[4] = {
        static_cast<int>(t->ne[0]),
        static_cast<int>(t->ne[1]),
        static_cast<int>(t->ne[2]),
        static_cast<int>(t->ne[3]),
    };
    int ndim = 1;
    for (int d = 3; d >= 0; d--) {
        if (t->ne[d] > 0) {
            ndim = d + 1;
            break;
        }
    }

    const ComputeDevice device = detect_device(t);

    data->hooks->capture_layer(
        layer_index, type, device,
        samples.data(), count, latency_ms,
        layer_name, shape, ndim, ggml_type_name(t->type));

    return true;
}

}  // namespace

static void sanitize_token_piece(char* buf, size_t len) {
    for (size_t i = 0; i < len && buf[i] != '\0'; i++) {
        if (std::isspace(static_cast<unsigned char>(buf[i]))) {
            buf[i] = '_';
        }
    }
}

static void set_token_labels_from_prompt(HookManager& hooks,
                                         const std::string& prompt) {
    char labels[LayerSnapshot::TOKEN_LABEL_CAP][16];
    int  n = 0;
    std::string word;
    for (size_t i = 0; i <= prompt.size() && n < LayerSnapshot::TOKEN_LABEL_CAP;
         i++) {
        if (i == prompt.size() ||
            std::isspace(static_cast<unsigned char>(prompt[i]))) {
            if (!word.empty()) {
                std::snprintf(labels[n], sizeof(labels[n]), "%s", word.c_str());
                n++;
                word.clear();
            }
        } else {
            word += prompt[i];
        }
    }
    if (n == 0) {
        std::snprintf(labels[0], sizeof(labels[0]), "Hello");
        n = 1;
    }
    hooks.set_token_labels(labels, n);
}

static void sync_token_labels(HookManager& hooks,
                              const llama_vocab* vocab,
                              const std::vector<llama_token>& tokens) {
    const int start = std::max(
        0, static_cast<int>(tokens.size()) - LayerSnapshot::TOKEN_LABEL_CAP);
    char labels[LayerSnapshot::TOKEN_LABEL_CAP][16];
    int  n = 0;
    for (int i = start; i < static_cast<int>(tokens.size()) &&
                       n < LayerSnapshot::TOKEN_LABEL_CAP;
         i++) {
        char buf[16];
        const int32_t len = llama_token_to_piece(
            vocab, tokens[i], buf, static_cast<int32_t>(sizeof(buf) - 1), 0,
            true);
        if (len <= 0) {
            std::snprintf(labels[n], sizeof(labels[n]), "t%d", i);
        } else {
            buf[len] = '\0';
            sanitize_token_piece(buf, static_cast<size_t>(len));
            std::snprintf(labels[n], sizeof(labels[n]), "%s", buf);
        }
        n++;
    }
    hooks.set_token_labels(labels, n);
}

static std::string format_chat_prompt(const llama_model* model,
                                      const std::string& user_prompt) {
    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl || tmpl[0] == '\0') {
        return user_prompt;
    }
    const llama_chat_message msg = {"user", user_prompt.c_str()};
    std::vector<char> buf(4096);
    const int32_t len = llama_chat_apply_template(
        tmpl, &msg, 1, true, buf.data(),
        static_cast<int32_t>(buf.size()));
    if (len < 0 || len >= static_cast<int32_t>(buf.size())) {
        return user_prompt;
    }
    return std::string(buf.data(), static_cast<size_t>(len));
}

static void set_simulated_attn_weights(HookManager& hooks, int layer) {
    float weights[LayerSnapshot::ATTN_CAP][LayerSnapshot::ATTN_CAP]{};
    constexpr int rows = 8;
    constexpr int cols = 8;
    for (int r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            const float v = (r == c) ? 0.35f : 0.05f + 0.02f * ((r + c + layer) % 5);
            weights[r][c] = v;
            row_sum += v;
        }
        for (int c = 0; c < cols; c++) {
            weights[r][c] /= row_sum;
        }
    }
    int wshape[4] = {cols, rows, 1, 1};
    hooks.set_attn_weights(layer, 0, weights, rows, cols, rows, wshape, 4);
}

void run_simulate(HookManager& hooks, bool& running,
                  const std::string& prompt) {
    float fake_data[128] = {};
    for (int i = 0; i < 128; i++) {
        fake_data[i] = (i % 7) * 0.5f - 1.5f;
    }

    hooks.set_model_name("simulated-model");
    set_token_labels_from_prompt(hooks, prompt);

    while (running) {
        for (int layer = 0; layer < 32 && running; layer++) {
            LayerType type;
            char layer_name[64];
            switch (layer % 4) {
                case 0:
                    type = LayerType::Embedding;
                    std::snprintf(layer_name, sizeof(layer_name),
                                  "embed_tokens");
                    break;
                case 1:
                    type = LayerType::Attention;
                    std::snprintf(layer_name, sizeof(layer_name),
                                  "layers.%d.attn", layer);
                    break;
                case 2:
                    type = LayerType::MLP;
                    std::snprintf(layer_name, sizeof(layer_name),
                                  "layers.%d.mlp", layer);
                    break;
                default:
                    type = LayerType::LayerNorm;
                    std::snprintf(layer_name, sizeof(layer_name),
                                  "layers.%d.ffn_norm", layer);
                    break;
            }

            if (type == LayerType::Attention) {
                set_simulated_attn_weights(hooks, layer);
            }

            ComputeDevice dev = (layer % 4 == 3)
                ? ComputeDevice::CPU
                : ComputeDevice::Metal;

            int shape[4] = {1, 32, 4096, 0};
            hooks.capture_layer(layer, type, dev, fake_data, 128,
                0.8f + (layer % 4) * 0.3f,
                layer_name, shape, 3, "float16");

            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }
}

void run_model(HookManager& hooks,
               const std::string& model_path,
               const std::string& prompt,
               bool& running) {
    ggml_backend_load_all();
    llama_backend_init();

    const llama_vocab* vocab = nullptr;
    std::vector<llama_token> prompt_tokens;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* smpl = nullptr;

    auto load_model = [&](int n_gpu_layers) -> bool {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;
        model = llama_model_load_from_file(model_path.c_str(), model_params);
        return model != nullptr;
    };

    constexpr uint32_t k_predict = 256;
    auto create_context = [&](LlamaInstrumentData& cb_data) -> bool {
        llama_context_params ctx_params = llama_context_default_params();
        const uint32_t n_prompt_u =
            static_cast<uint32_t>(prompt_tokens.size());
        ctx_params.n_ctx = n_prompt_u + k_predict;
        ctx_params.n_batch = n_prompt_u > 512u ? n_prompt_u : 512u;
        ctx_params.n_threads = 4;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_params.cb_eval = llama_eval_callback;
        ctx_params.cb_eval_user_data = &cb_data;
        ctx = llama_init_from_model(model, ctx_params);
        return ctx != nullptr;
    };

    if (!load_model(-1)) {
        std::fprintf(stderr, "[ModelRunner] failed to load model: %s\n",
                     model_path.c_str());
        std::fprintf(stderr, "[ModelRunner] check the path exists and is a valid .gguf file\n");
        llama_backend_free();
        return;
    }

    std::fprintf(stderr, "[ModelRunner] model loaded, tokenizing prompt...\n");

    const std::string model_label = display_model_name(model, model_path);
    hooks.set_model_name(model_label.c_str());

    vocab = llama_model_get_vocab(model);
    const std::string formatted_prompt = format_chat_prompt(model, prompt);
    if (formatted_prompt != prompt) {
        std::fprintf(stderr, "[ModelRunner] applied chat template\n");
    }

    const int n_prompt = -llama_tokenize(
        vocab, formatted_prompt.c_str(),
        static_cast<int32_t>(formatted_prompt.size()),
        nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "[ModelRunner] failed to tokenize prompt\n");
        llama_model_free(model);
        llama_backend_free();
        return;
    }

    prompt_tokens.resize(n_prompt);
    if (llama_tokenize(vocab, formatted_prompt.c_str(),
            static_cast<int32_t>(formatted_prompt.size()),
            prompt_tokens.data(), prompt_tokens.size(),
            true, true) < 0) {
        std::fprintf(stderr, "[ModelRunner] failed to tokenize prompt\n");
        llama_model_free(model);
        llama_backend_free();
        return;
    }

    LlamaInstrumentData cb_data;
    cb_data.hooks = &hooks;
    cb_data.n_layer = llama_model_n_layer(model);

    if (!create_context(cb_data)) {
        std::fprintf(stderr,
            "[ModelRunner] GPU init failed, retrying CPU-only...\n");
        llama_model_free(model);
        model = nullptr;
        if (!load_model(0) || !create_context(cb_data)) {
            std::fprintf(stderr, "[ModelRunner] failed to create context\n");
            if (model) llama_model_free(model);
            llama_backend_free();
            return;
        }
        std::fprintf(stderr, "[ModelRunner] running on CPU\n");
    }

    auto sparams = llama_sampler_chain_default_params();
    auto make_sampler = [&](bool suppress_eos) -> llama_sampler* {
        auto* chain = llama_sampler_chain_init(sparams);
        if (suppress_eos) {
            const llama_token eos = llama_vocab_eos(vocab);
            llama_logit_bias bias{eos, -1e10f};
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            llama_sampler_chain_add(
                chain, llama_sampler_init_logit_bias(n_vocab, 1, &bias));
        }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(42));
        return chain;
    };

    smpl = make_sampler(true);

    std::vector<llama_token> context_tokens = prompt_tokens;
    sync_token_labels(hooks, vocab, context_tokens);

    const int32_t n_prompt_tokens =
        static_cast<int32_t>(prompt_tokens.size());
    llama_batch batch = llama_batch_get_one(
        prompt_tokens.data(), n_prompt_tokens);

    hooks.reset_attn_caches();
    hooks.set_decode_step(0);
    cb_data.n_kv_active = n_prompt_tokens;
    cb_data.reset_attn_diag();
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "[ModelRunner] prompt decode failed\n");
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return;
    }

    std::fprintf(stderr,
        "[ModelRunner] prefill complete — generating tokens (live stream)\n");

    constexpr int k_max_gen_tokens = 48;
    constexpr int k_min_gen_tokens = 16;
    constexpr int k_gen_delay_ms   = 450;
    bool          eos_allowed      = false;

    for (int gen = 1; running && gen <= k_max_gen_tokens; gen++) {
        if (!eos_allowed && gen >= k_min_gen_tokens) {
            llama_sampler_free(smpl);
            smpl = make_sampler(false);
            eos_allowed = true;
            std::fprintf(stderr,
                "[ModelRunner] minimum %d tokens reached — EOS allowed\n",
                k_min_gen_tokens);
        }

        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            std::fprintf(stderr,
                "[ModelRunner] EOS at generated token %d — stream pausing\n",
                gen);
            break;
        }

        context_tokens.push_back(tok);
        sync_token_labels(hooks, vocab, context_tokens);

        hooks.set_decode_step(gen);
        cb_data.n_kv_active = static_cast<int>(context_tokens.size());

        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr,
                "[ModelRunner] token decode failed at step %d\n", gen);
            break;
        }
        llama_sampler_accept(smpl, tok);

        std::fprintf(stderr, "[ModelRunner] generated token %d / %d\n",
                     gen, k_max_gen_tokens);

        std::this_thread::sleep_for(
            std::chrono::milliseconds(k_gen_delay_ms));
    }

    std::fprintf(stderr,
        "[ModelRunner] generation done (%d ms/token) — inspect UI, q to quit\n",
        k_gen_delay_ms);

    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
