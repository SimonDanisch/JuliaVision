// Same-checkpoint Vulkan numerical reference and synchronized-greedy timing.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

struct trace_data {
    bool enabled = false;
    std::filesystem::path directory;
};

static bool trace_callback(ggml_tensor * tensor, bool ask, void * pointer) {
    auto & data = *static_cast<trace_data *>(pointer);
    const std::string name = ggml_get_name(tensor);
    const bool capture = data.enabled && tensor->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(tensor) && ggml_nbytes(tensor) <= 1024*1024 &&
        (name.find("l_out-")==0 || name.find("attn_norm-")==0 ||
         name.find("linear_attn_out-")==0 || name.find("attn_post_norm-")==0 ||
         name.find("ffn_out-")==0 || name=="model.input_embed");
    if (ask) return capture;
    if (!capture) return true;
    std::vector<float> values(ggml_nbytes(tensor)/sizeof(float));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    FILE * file = std::fopen((data.directory/(name+".f32")).c_str(),"wbx");
    if (!file) throw std::runtime_error("Trace receipt exists");
    std::fwrite(values.data(),sizeof(float),values.size(),file);
    std::fclose(file);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) throw std::runtime_error("checkpoint output-directory Vulkan-device-substring [trace] required");
    const std::filesystem::path output(argv[2]);
    if (std::filesystem::exists(output)) throw std::runtime_error("Preserve first reference attempt");
    const bool trace = argc==5 && std::string(argv[4])=="trace";
    trace_data trace_state;
    trace_state.directory=output/"stages";
    if (trace) std::filesystem::create_directories(trace_state.directory);
    auto vulkan = ggml_backend_reg_by_name("Vulkan");
    if (!vulkan) throw std::runtime_error("Vulkan backend unavailable; no CPU fallback");
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(vulkan); ++i) {
        auto candidate = ggml_backend_reg_dev_get(vulkan, i);
        if (std::string(ggml_backend_dev_description(candidate)).find(argv[3]) != std::string::npos) gpu = candidate;
    }
    if (!gpu) throw std::runtime_error("Requested Radeon unavailable; no fallback");
    ggml_backend_dev_t devices[] = {gpu, nullptr};
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.devices = devices;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    const auto load_start = std::chrono::steady_clock::now();
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) throw std::runtime_error("Model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 32;
    cp.n_ubatch = 32;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.offload_kqv = true;
    cp.op_offload = true;
    if (trace) { cp.cb_eval=trace_callback;cp.cb_eval_user_data=&trace_state; }
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("Context initialization failed");
    llama_synchronize(ctx);
    const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-load_start).count();
    const auto vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const std::string prompt = "Julia arrays use column-major storage.";
    std::vector<llama_token> tokens(128);
    int count = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
    if (count <= 0) throw std::runtime_error("Tokenization failed");
    tokens.resize(std::min(count, 12));
    if (trace) tokens.resize(1);
    std::filesystem::create_directories(output);
    FILE * data = std::fopen((output / "logits.f32").c_str(), "wbx");
    if (!data) throw std::runtime_error("Cannot create reference logits");
    std::vector<int> best;
    for (size_t position = 0; position < tokens.size(); ++position) {
        trace_state.enabled=trace;
        auto batch = llama_batch_get_one(&tokens[position], 1);
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("Reference decode failed");
        const auto logits = llama_get_logits_ith(ctx, -1);
        for (int i = 0; i < n_vocab; ++i) if (!std::isfinite(logits[i])) throw std::runtime_error("Nonfinite logits");
        if (std::fwrite(logits, sizeof(float), n_vocab, data) != size_t(n_vocab)) throw std::runtime_error("Reference write failed");
        best.push_back(std::max_element(logits, logits + n_vocab) - logits);
        std::fprintf(stderr, "Vulkan reference step%zu/%zu next=%d\n", position+1, tokens.size(), best.back());
    }
    std::fclose(data);
    if (trace) {
        llama_synchronize(ctx);
        FILE * receipt=std::fopen((output/"trace.json").c_str(),"wx");
        std::fprintf(receipt,"{\"complete\":true,\"device\":\"Vulkan Radeon\",\"token\":%d,\"normal_final_sync\":true,\"scope\":\"First-token stage trace; not timing\"}\n",tokens.front());
        std::fclose(receipt);
        llama_free(ctx);llama_model_free(model);llama_backend_free();
        return 0;
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    const auto prefill_start = std::chrono::steady_clock::now();
    for (auto prompt_token : tokens) {
        auto batch = llama_batch_get_one(&prompt_token, 1);
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("Timed Vulkan prefill failed");
    }
    llama_synchronize(ctx);
    const double prefill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-prefill_start).count();
    // Label this separately from device-feedback decode: this harness reads
    // logits and selects greedily on the host, as a synchronized baseline.
    std::vector<int> generated;
    const auto prompt_logits = llama_get_logits_ith(ctx, -1);
    llama_token token = std::max_element(prompt_logits, prompt_logits+n_vocab)-prompt_logits;
    const auto decode_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 64; ++i) {
        auto batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("Timed Vulkan decode failed");
        const auto logits = llama_get_logits_ith(ctx, -1);
        token = std::max_element(logits, logits+n_vocab)-logits;
        generated.push_back(token);
    }
    llama_synchronize(ctx);
    const double decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-decode_start).count();
    FILE * report = std::fopen((output / "reference.json").c_str(), "wx");
    if (!report) throw std::runtime_error("Cannot create reference receipt");
    std::fprintf(report, "{\"complete\":true,\"device\":\"Vulkan Radeon\",\"normal_final_sync\":true,\"scope\":\"Same-GGUF Vulkan reference, not task quality\",\"load_seconds\":%.9f,\"prefill_seconds\":%.9f,\"prefill_mode\":\"warm sequential one-token batches; final sync\",\"decode_seconds\":%.9f,\"decode_tokens\":64,\"decode_tokens_per_second\":%.9f,\"timing_mode\":\"synchronized greedy; host logits readback and argmax included\",\"kv_type\":\"Float32\",\"capacity\":256,\"vocab\":%d,\"tokens\":[", load_seconds, prefill_seconds, decode_seconds, 64.0/decode_seconds, n_vocab);
    for (size_t i=0;i<tokens.size();++i) std::fprintf(report, "%s%d", i?",":"", tokens[i]);
    std::fprintf(report, "],\"next_tokens\":[");
    for (size_t i=0;i<best.size();++i) std::fprintf(report, "%s%d", i?",":"", best[i]);
    std::fprintf(report, "],\"generated_tokens\":[");
    for (size_t i=0;i<generated.size();++i) std::fprintf(report, "%s%d", i?",":"", generated[i]);
    std::fprintf(report, "]}\n");
    std::fclose(report);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
