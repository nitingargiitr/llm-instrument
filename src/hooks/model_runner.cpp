#include "hooks/model_runner.h"
#include "llama.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

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
};

static bool is_interested_tensor(const char* name) {
    return std::strcmp(name, "inp_embd") == 0 ||
           std::strcmp(name, "result_norm") == 0 ||
           std::strstr(name, "attn_out-") != nullptr ||
           std::strstr(name, "ffn_out-") != nullptr ||
           std::strstr(name, "attn_norm-") != nullptr ||
           std::strstr(name, "ffn_norm-") != nullptr;
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

    int layer_index = 0;
    LayerType type = LayerType::Unknown;
    char layer_name[64]{};
    if (!parse_tensor(t->name, data->n_layer, layer_index, type,
                      layer_name, sizeof(layer_name))) {
        return true;
    }

    std::vector<float> samples;
    const int count = tensor_to_floats(t, samples, 4096);
    if (count <= 0) return true;

    const float latency_ms =
        static_cast<float>(ggml_time_us() - data->op_start_us) / 1000.0f;

    int shape[4] = {
        static_cast<int>(t->ne[0]),
        static_cast<int>(t->ne[1]),
        static_cast<int>(t->ne[2]),
        static_cast<int>(t->ne[3]),
    };
    int ndim = 0;
    for (int d = 0; d < 4; d++) {
        if (t->ne[d] > 1) ndim = d + 1;
    }
    if (ndim == 0) ndim = 1;

    const ComputeDevice device = ggml_backend_buffer_is_host(t->buffer)
        ? ComputeDevice::CPU
        : ComputeDevice::CUDA;

    data->hooks->capture_layer(
        layer_index, type, device,
        samples.data(), count, latency_ms,
        layer_name, shape, ndim, ggml_type_name(t->type));

    return true;
}

}  // namespace

void run_simulate(HookManager& hooks, bool& running) {
    float fake_data[128] = {};
    for (int i = 0; i < 128; i++) {
        fake_data[i] = (i % 7) * 0.5f - 1.5f;
    }

    hooks.set_model_name("simulated-model");

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

            ComputeDevice dev = (layer % 4 == 3)
                ? ComputeDevice::CPU
                : ComputeDevice::CUDA;

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

    auto create_context = [&](LlamaInstrumentData& cb_data) -> bool {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = static_cast<uint32_t>(prompt_tokens.size() + 256);
        ctx_params.n_batch = static_cast<uint32_t>(prompt_tokens.size());
        ctx_params.n_threads = 4;
        ctx_params.cb_eval = llama_eval_callback;
        ctx_params.cb_eval_user_data = &cb_data;
        ctx = llama_init_from_model(model, ctx_params);
        return ctx != nullptr;
    };

    if (!load_model(-1)) {
        std::fprintf(stderr, "[ModelRunner] failed to load model: %s\n",
                     model_path.c_str());
        llama_backend_free();
        return;
    }

    char model_desc[64]{};
    llama_model_desc(model, model_desc, sizeof(model_desc));
    hooks.set_model_name(model_desc);

    vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(
        vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
        nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "[ModelRunner] failed to tokenize prompt\n");
        llama_model_free(model);
        llama_backend_free();
        return;
    }

    prompt_tokens.resize(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
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
    smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(
        prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));

    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "[ModelRunner] prompt decode failed\n");
    }

    llama_token next = llama_sampler_sample(smpl, ctx, -1);

    while (running) {
        if (llama_vocab_is_eog(vocab, next)) {
            next = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "[ModelRunner] decode failed\n");
            break;
        }

        next = llama_sampler_sample(smpl, ctx, -1);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
