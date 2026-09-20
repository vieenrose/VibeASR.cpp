/**
 * VibeASR.cpp - Streaming ASR inference for VibeVoice-ASR-Streaming checkpoints.
 *
 * Mirrors upstream `VibeVoiceASRForConditionalGeneration.streaming_generate`
 * (default split_then_encode, greedy):
 *   prefill plain-text prompt (no chat template)
 *   per 2.933s chunk + 0.533s lookahead window (83200 samples, hop 70400):
 *     [speech_start] + 26 VAE frames + [speech_end] (KV keeps growing)
 *     greedy-decode until <|text_chunk_end|> (151665) or EOS (151643), cap N/chunk
 *     feed <|text_chunk_end|> itself so the cache always ends on it
 *     print chunk text as soon as it is emitted
 *
 * Usage:
 *   ./asr_streaming --vae-model <vae.gguf> --lm-model <lm-q4.gguf> --audio <wav>
 *       [-t 4] [--max-tokens 256] [--context "hotwords"] [--no-mmap]
 */

#include "vae.h"
#include "llama.h"

#include "../utils/audio_io.h"
#include "time_compat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Canonical token IDs from the Python tokenizer (Qwen2.5 + VibeVoice).
// GGUF vocab was converted from the same tokenizer, so positions match.
static const int TOK_EOS = 151643;            // <|endoftext|>
static const int TOK_SPEECH_START = 151646;   // <|speech_start|>
static const int TOK_SPEECH_END = 151647;     // <|speech_end|>
static const int TOK_TEXT_CHUNK_END = 151665; // <|text_chunk_end|>

// Streaming constants from preprocessor_config.json
static const int SAMPLE_RATE = 24000;
static const int WINDOW_SAMPLES = 83200;  // 26 frames * 3200
static const int HOP_SAMPLES = 70400;     // 22 frames * 3200
static const int FRAMES_PER_WINDOW = 26;

struct stream_params {
    std::string vae_model_path;
    std::string lm_model_path;
    std::string audio_path;
    std::string context_info;
    int n_threads = 4;
    // Measured 46.5 KV positions per window, counted directly from a 155 s LATENCY_TRACE run (28 fed rows +
    // ~18.5 emitted tokens; Exp849's -c bracketing said 49.5 +- 3, consistent). So 4096 covers ~258 s = 4.3 min
    // of continuous audio, NOT 15 min - that figure belongs to 16384 (~16 min, KV 450MB vs 112MB here:
    // 28 layers x 2 kv heads x 128 head dim x K+V x 2 bytes = 28.0 KB per position).
    // On exhaustion the window loop exits 1 with "decode failed" or "frames failed" and prints NO final summary.
    // Raise -c for longer sessions. An 8-bit KV cache would give 1.88x positions per byte but is NOT exposed in
    // this build: the type is hardcoded (3rdparty/llama.cpp/src/llama.cpp: type_k/type_v = GGML_TYPE_F16) and there
    // is no --kv-type flag (passing it exits 1 "Unknown arg"). Enabling it needs a vendored change + an accuracy gate.
    int n_ctx = 4096;
    int n_batch = 512;
    int max_tokens_per_chunk = 256;
    int vae_pieces = 1;   // window split count; must divide 26 (frames).
                          // DEFAULT IS THE SHIPPED TIER (Exp880): a bare invocation with no
                          // --vae-pieces must reproduce the gated configuration, not p2 (which
                          // is +1.2 % slower and heavier - the Exp859 class: a default that
                          // silently measures a different system than the docs describe).
    bool xwin = false;    // cross-window carry: VAE cache persists across hops
                          // (full-context features, no overlap recompute)
    int xwin_reset = 8;   // reset carry every N hops (0 = never). Bounds KV/context
                          // drift that otherwise degrades long files (12% WER at 69s
                          // with unbounded carry vs 3.7% cold; shorts unaffected).
    bool use_mmap = true;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --vae-model <vae.gguf> --lm-model <lm.gguf> --audio <wav> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --vae-model <path>     VAE encoder GGUF (required)\n");
    fprintf(stderr, "  --lm-model <path>      Streaming LM GGUF, e.g. Q4_K_M (required)\n");
    fprintf(stderr, "  --audio <path>         Input WAV file (required, resampled to 24k mono)\n");
    fprintf(stderr, "  -t <n>                 Threads (default: 4)\n");
    fprintf(stderr, "  -c <n>                 Context size (default: 4096)\n");
    fprintf(stderr, "  --max-tokens <n>       Max new tokens per chunk (default: 256)\n");
    fprintf(stderr, "  --vae-pieces <n>       Window split count, must divide 26 (default: 1)\n");
    fprintf(stderr, "                         1 = full-window encode (the shipped tier); 13/26 = cached pieces\n");
    fprintf(stderr, "  --xwin                   Cross-window VAE carry (full-context\n");
    fprintf(stderr, "                         features, skips overlap recompute)\n");
    fprintf(stderr, "  --xwin-reset <n>        Reset carry every N hops, 0 = never\n");
    fprintf(stderr, "                         (default: 8; bounds long-file drift)\n");
    fprintf(stderr, "  --context <text>       Hotwords, e.g. \"VibeVoice,diarization\"\n");
    fprintf(stderr, "  --no-mmap              Do not mmap LM weights\n");
}

static bool parse_args(int argc, char ** argv, stream_params & p) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--vae-model" && i + 1 < argc) p.vae_model_path = argv[++i];
        else if (a == "--lm-model" && i + 1 < argc) p.lm_model_path = argv[++i];
        else if (a == "--audio" && i + 1 < argc) p.audio_path = argv[++i];
        else if (a == "--context" && i + 1 < argc) p.context_info = argv[++i];
        else if ((a == "-t" || a == "--threads") && i + 1 < argc) p.n_threads = std::stoi(argv[++i]);
        else if (a == "-c" && i + 1 < argc) p.n_ctx = std::stoi(argv[++i]);
        else if (a == "--max-tokens" && i + 1 < argc) p.max_tokens_per_chunk = std::stoi(argv[++i]);
        else if (a == "--vae-pieces" && i + 1 < argc) p.vae_pieces = std::stoi(argv[++i]);
        else if (a == "--xwin") p.xwin = true;
        else if (a == "--xwin-reset" && i + 1 < argc) p.xwin_reset = std::stoi(argv[++i]);
        else if (a == "--no-mmap") p.use_mmap = false;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown arg: %s\n", a.c_str()); return false; }
    }
    if (p.vae_model_path.empty() || p.lm_model_path.empty() || p.audio_path.empty()) {
        fprintf(stderr, "Missing required args\n\n"); print_usage(argv[0]); return false;
    }
    if (26 % p.vae_pieces != 0) {
        fprintf(stderr, "--vae-pieces must divide 26 (1, 2, 13, 26)\n"); return false;
    }
    return true;
}

// Phase timers (LM prefill vs decode split). Zero per run in main.
static double g_prefill_ms = 0.0;
static double g_decode_ms = 0.0;
static double g_ac_ms = 0.0;
static double g_sem_ms = 0.0;

// Census phase hook, defined in ggml.c (Exp816). Declared here rather than in a header because it is a
// measurement instrument, not API: it lets the MAC census attribute blocked-int8 MACs to the SAME spans as
// the phase timers below, so a per-phase rate is measured rather than derived from a parameter count.
extern "C" void ggml_mm_set_phase(int);

static double now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static std::vector<llama_token> tokenize_text(const llama_model * model, const std::string & text) {
    int n = llama_tokenize(model, text.c_str(), (int)text.size(), nullptr, 0, false, false);
    if (n < 0) n = -n;
    std::vector<llama_token> out(n);
    int r = llama_tokenize(model, text.c_str(), (int)text.size(), out.data(), n, false, false);
    if (r < 0) return {};
    out.resize(r);
    return out;
}

static std::string detokenize(const llama_model * model, const std::vector<llama_token> & toks) {
    std::string s;
    char buf[256];
    for (auto t : toks) {
        int n = llama_token_to_piece(model, t, buf, sizeof(buf), 0, false);
        if (n > 0) s.append(buf, n);
    }
    return s;
}

// Decode one token-ID at running position; returns <0 on error, else new pos.
static int feed_token(llama_context * ctx, llama_token tok, int pos) {
    llama_batch b = llama_batch_get_one(&tok, 1, pos, 0);
    if (llama_decode(ctx, b) != 0) return -1;
    return pos + 1;
}

// Decode n_embd-dim embedding rows at running position; returns <0 on error.
static int feed_embeds(llama_context * ctx, const float * emb, int n_rows, int n_embd, int pos, int n_batch) {
    int done = 0;
    while (done < n_rows) {
        int bl = std::min(n_batch, n_rows - done);
        llama_batch b = llama_batch_init(bl, n_embd, 1);
        b.n_tokens = bl;
        b.token = nullptr;
        float * save = b.embd;
        b.embd = const_cast<float *>(emb) + done * n_embd;
        for (int i = 0; i < bl; i++) {
            b.pos[i] = pos + i;
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0;
        }
        // Request logits on the last row of the last batch so sampling can follow.
        b.logits[bl - 1] = (done + bl == n_rows) ? 1 : 0;
        int rc = llama_decode(ctx, b);
        b.embd = save;
        llama_batch_free(b);
        if (rc != 0) return -1;
        done += bl;
        pos += bl;
    }
    return pos;
}

// Emit one chunk (26 frames at [frames, frames+26)) through the LM. Mirrors the
// legacy per-window loop exactly (speech markers, greedy-to-chunk-end, tce feed).
// Returns 0 on success, -1 on error. Updates pos_ms/lm_ms/total_tokens/full_text.
static int emit_chunk(llama_context * lctx, llama_sampler * smpl, llama_model * model,
                      const float * frames, int n_embd, int n_batch, int max_tokens,
                      const char ** strip_list, int n_strip,
                      int & pos, double & lm_ms, int & total_tokens, std::string & full_text,
                      int show_idx, int show_total) {
    double t0 = now_ms();
    ggml_mm_set_phase(2);   // lm_prefill - same span as the g_prefill_ms timer below (Exp816 phase census)
    llama_token t_start = TOK_SPEECH_START, t_end = TOK_SPEECH_END, t_tce = TOK_TEXT_CHUNK_END;
    if ((pos = feed_token(lctx, t_start, pos)) < 0) return -1;
    if ((pos = feed_embeds(lctx, frames, FRAMES_PER_WINDOW, n_embd, pos, n_batch)) < 0) return -1;
    if ((pos = feed_token(lctx, t_end, pos)) < 0) return -1;
    g_prefill_ms += now_ms() - t0;
    ggml_mm_set_phase(3);   // lm_decode
    llama_token tok = llama_sampler_sample(smpl, lctx, -1);
    llama_sampler_accept(smpl, tok);
    double tdec = now_ms();
    std::vector<llama_token> chunk_ids;
    for (int i = 0; i < max_tokens; i++) {
        if (tok == TOK_TEXT_CHUNK_END || tok == TOK_EOS) break;
        chunk_ids.push_back(tok);
        if ((pos = feed_token(lctx, tok, pos)) < 0) return -1;
        tok = llama_sampler_sample(smpl, lctx, -1);
        llama_sampler_accept(smpl, tok);
    }
    if ((pos = feed_token(lctx, t_tce, pos)) < 0) return -1;
    g_decode_ms += now_ms() - tdec;
    ggml_mm_set_phase(0);   // outside any measured phase (sampling/detokenize still happen, unattributed)
    lm_ms += now_ms() - t0;
    std::string text = detokenize(model, chunk_ids);
    for (int s = 0; s < n_strip; s++) {
        size_t p;
        while ((p = text.find(strip_list[s])) != std::string::npos) text.erase(p, strlen(strip_list[s]));
    }
    full_text += text;
    total_tokens += (int)chunk_ids.size();
    printf("[%d/%d] %s\n", show_idx, show_total, text.c_str());
    fflush(stdout);
    return 0;
}

// Encode nsamp samples (3200-multiple) into summed 1536-dim frames via the
// carried cache. Returns frames written, or -1 on error.
static int encode_frames(vae_context_t * vae_ctx, vae_cache_t * vcache,
                         const float * samples, int nsamp,
                         float * afe, float * sfe, int acoustic_dim, int semantic_dim,
                         int want_frames, double & vae_ms) {
    static const int SUB_SAMPLES = 6400;  // 2 frames; divides window/hop/tail
    if (nsamp % SUB_SAMPLES != 0) return -1;
    double t0 = now_ms();
    ggml_mm_set_phase(1);   // vae - both encoder chains run under this phase
    int got = 0;
    for (int off = 0; off < nsamp; off += SUB_SAMPLES) {
        float * af = afe + (got * acoustic_dim);
        float * sf = sfe + (got * semantic_dim);
        double ta = now_ms();
        int na = vae_encode_acoustic_cached(vae_ctx, vcache, samples + off, SUB_SAMPLES, af);
        g_ac_ms += now_ms() - ta;
        double ts = now_ms();
        int ns = vae_encode_semantic_cached(vae_ctx, vcache, samples + off, SUB_SAMPLES, sf);
        g_sem_ms += now_ms() - ts;
        if (na != 2 || ns != 2) return -1;
        got += 2;
    }
    vae_ms += now_ms() - t0;
    ggml_mm_set_phase(0);
    return (got == want_frames) ? got : -1;
}


// Exp669: fault tracer for diagnosing the dw-conv1d crash (Exp666-668). Off unless
// VAE_FAULT_TRACE=1, so the shipped binary's crash behaviour is untouched. Prints the faulting
// address plus the PC as an offset inside its shared object, which maps to a function with
// llvm-objdump even for static (unexported) kernels: addr = pc - dli_fbase.
#include <signal.h>
#include <dlfcn.h>
#include <ucontext.h>
#include <unistd.h>

static void vibe_fault_handler(int sig, siginfo_t * si, void * uc) {
    ucontext_t * c = (ucontext_t *) uc;
    void * pc = (void *) c->uc_mcontext.pc;
    Dl_info info;
    if (dladdr(pc, &info) && info.dli_fbase) {
        fprintf(stderr, "[FAULT] sig=%d addr=%p pc=%p fbase=%p off=0x%lx sym=%s%+ld\n",
                sig, si->si_addr, pc, info.dli_fbase,
                (unsigned long) ((char *) pc - (char *) info.dli_fbase),
                info.dli_sname ? info.dli_sname : "?",
                info.dli_sname ? (long) ((char *) pc - (char *) info.dli_saddr) : 0L);
    } else {
        fprintf(stderr, "[FAULT] sig=%d addr=%p pc=%p (no dladdr)\n", sig, si->si_addr, pc);
    }
    fflush(stderr);
    _exit(139);
}

static void vibe_fault_trace_install(void) {
    if (getenv("VAE_FAULT_TRACE") == nullptr) return;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = vibe_fault_handler;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    for (int sig : {SIGSEGV, SIGBUS}) sigaction(sig, &sa, nullptr);
    fprintf(stderr, "[FAULT] tracer installed\n");
}


int main(int argc, char ** argv) {
    vibe_fault_trace_install();
    stream_params params;
    if (!parse_args(argc, argv, params)) return 1;

    fprintf(stderr, "========================================\n");
    fprintf(stderr, " VibeASR.cpp - Streaming ASR (1.5B)\n");
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "  VAE: %s\n  LM:  %s\n  Audio: %s\n  Threads: %d\n\n",
            params.vae_model_path.c_str(), params.lm_model_path.c_str(),
            params.audio_path.c_str(), params.n_threads);

    double total_start = now_ms();

    // ---- audio ----
    audio_io::AudioData audio;
    if (!audio_io::load_audio(params.audio_path, SAMPLE_RATE, true, audio)) {
        fprintf(stderr, "Failed to load audio\n"); return 1;
    }
    fprintf(stderr, "Audio: %.2fs, %zu samples\n", audio.duration_sec, audio.samples.size());

    // ---- VAE ----
    struct vae_model_params vmp = vae_model_default_params();
    vmp.n_threads = params.n_threads;
    vae_model_t * vae_model = vae_load_model_from_file(params.vae_model_path.c_str(), vmp);
    if (!vae_model) { fprintf(stderr, "VAE load failed\n"); return 1; }
    struct vae_context_params vcp = vae_context_default_params();
    vcp.n_threads = params.n_threads;
    vae_context_t * vae_ctx = vae_new_context_with_model(vae_model, vcp);
    if (!vae_ctx) { fprintf(stderr, "VAE ctx failed\n"); return 1; }
    int acoustic_dim = vae_model_acoustic_dim(vae_model);
    int semantic_dim = vae_model_semantic_dim(vae_model);
    fprintf(stderr, "VAE dims: acoustic=%d semantic=%d\n", acoustic_dim, semantic_dim);

    // ---- LM ----
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = params.use_mmap;
    llama_model * model = llama_load_model_from_file(params.lm_model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "LM load failed\n"); return 1; }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = params.n_ctx;
    cparams.n_batch = params.n_batch;
    cparams.n_threads = params.n_threads;
    cparams.n_threads_batch = params.n_threads;
    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (!lctx) { fprintf(stderr, "LM ctx failed\n"); return 1; }
    int n_embd = llama_n_embd(model);
    fprintf(stderr, "LM n_embd=%d n_vocab=%d\n", n_embd, llama_n_vocab(model));

    double load_done = now_ms();

    // ---- prompt (plain text, no chat template — exactly like upstream) ----
    std::string prompt_text =
        "You are a helpful assistant that transcribes audio input into text output. "
        "Please transcribe the following audios streamingly with these keys: speaker, content";
    if (!params.context_info.empty()) prompt_text += " and extra info: " + params.context_info;
    prompt_text += "\n";
    std::vector<llama_token> prompt_ids = tokenize_text(model, prompt_text);
    fprintf(stderr, "Prompt: %zu tokens\n", prompt_ids.size());

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // ---- prefill prompt ----
    llama_kv_cache_clear(lctx);
    int pos = 0;
    double tpre = now_ms();
    ggml_mm_set_phase(2);   // lm_prefill (the one-time prompt pass; ne11=31 in the census)
    {
        int done = 0, n = (int)prompt_ids.size();
        while (done < n) {
            int bl = std::min(params.n_batch, n - done);
            llama_batch b = llama_batch_init(bl, 0, 1);
            b.n_tokens = bl;
            for (int i = 0; i < bl; i++) {
                b.token[i] = prompt_ids[done + i];
                b.pos[i] = pos + i;
                b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
                b.logits[i] = (done + i == n - 1) ? 1 : 0;
            }
            if (llama_decode(lctx, b) != 0) { fprintf(stderr, "prompt prefill failed\n"); return 1; }
            llama_batch_free(b);
            done += bl; pos += bl;
        }
    }
    g_prefill_ms += now_ms() - tpre;
    ggml_mm_set_phase(0);

    // ---- windows ----
    int n_samples = (int)audio.samples.size();
    int n_windows = (n_samples + HOP_SAMPLES - 1) / HOP_SAMPLES;
    fprintf(stderr, "Windows: %d (window=%d hop=%d)\n\n", n_windows, WINDOW_SAMPLES, HOP_SAMPLES);

    std::vector<float> window(WINDOW_SAMPLES);
    std::vector<float> afe(FRAMES_PER_WINDOW * acoustic_dim);
    std::vector<float> sfe(FRAMES_PER_WINDOW * semantic_dim);
    std::vector<float> speech_emb(FRAMES_PER_WINDOW * n_embd);
    // Exp834: one prefill batch per window instead of three decodes. This llama cannot mix token-id rows with
    // embedding rows in a batch, so the two boundary tokens are pre-embedded (llama_token_embd_row) and placed
    // around the audio frames: [speech_start, frame x win_frames, speech_end]. Attention and positions are
    // unchanged; the win_frames+2 rows just share one weight pass instead of three.
    // Exp835: the batch moves the two boundary rows from the gemv path into the gemm path. That is equal in
    // arithmetic but NOT bit-identical, so it is enabled only where byte-identity was actually verified: the
    // shipping windowed config. Measured: shipped tier 39 tokens, transcript byte-identical; the deferred
    // (RAM-lean) path flips one token (38 vs 39), so it keeps the three-decode path until that is understood.
    const bool bound_batch = getenv("BOUND_BATCH_OFF") == nullptr;   // config test is at the feed site (defer_now)
    std::vector<float> chunk_emb(bound_batch ? (size_t)(FRAMES_PER_WINDOW + 2) * n_embd : 0);
    bool bound_ready = false;
    // The cached (and, by default, deferred) path is used for every piece count
    // including 1, where the single piece is the whole window: the cache is cold
    // at the window start and there are no interior boundaries, i.e. exactly the
    // semantics of the legacy full-window encode, but the deep stages still get
    // their window-level pass.
    vae_cache_t * vcache = (params.vae_pieces >= 1) ? vae_cache_new() : nullptr;
    {   // Exp821: the depthwise kernel can absorb the conv's causal left pad instead of the graph materialising
        // [hist | x] (-3.2% RTF, -183 MB RSS at the shipped tier, transcript byte-identical over 6 runs). It is
        // output-neutral only when no later build in the window reads the history this build rolls: one piece
        // per window AND no deferred late pass. At p13+defer-ON it changed the lean transcript (38 vs 39
        // tokens), so the gate stays closed there - see vae_stream_cache::single_piece_window.
        const char * denv = getenv("VAE_DEFER_LATE");
        const bool defer_now = (denv != nullptr) && (atoi(denv) > 0) && (getenv("VAE_SEQ_ENCODERS") == nullptr);
        // Exp825: --xwin is excluded too. The carry mode never resets the cache per window, so a site's cold
        // build (window 1, where the fast path fires) rolls a history that later CARRY builds read - and the
        // host-side roll is not what the materialized [hist | x] tensor would have left there. Measured: with
        // the fast path on, --xwin emits 37 tokens vs 39 with VAE_DW_LPAD_OFF=1, i.e. this option's documented
        // characterization (Exp648/714 attribution & WER) was silently measuring a different system since
        // v4.5. Excluding carry restores the pre-Exp821 output exactly (hash 5f08cd0af04f).
        vae_cache_set_whole_window(vcache,
            (params.vae_pieces == 1 && !defer_now && !params.xwin) ? 1 : 0);
    }
    // Exp829: the final-window flush is the DEFAULT. Parity-gated on the phone 40-utt set:
    // WER 4.38 % vs 4.51 %, 0 discordant tokens of 731 (McNemar p=1.0), one FEWER insertion, and 6/40
    // transcripts differing only in punctuation/marginal class. Speed: protocol 2.1866 -> 1.9295 (-11.8 %),
    // gate mean 2.4456 -> 1.9811 (-19.0 %), 17 s -4.0 %, 69 s -1.5 % (the win is length-weighted, as priced
    // in Exp828). Hatches: FLUSH_TAIL_OFF=1 reverts to the padded protocol exactly; FLUSH_TAIL=0 is an alias.
    // The deferred late path is excluded - it assembles a window-sized boundary buffer (tpiece * pieces), so a
    // short window needs its own shape handling; the lean tier therefore keeps the padded behaviour.
    const char * fl_env = getenv("FLUSH_TAIL");
    const bool flush_tail = getenv("FLUSH_TAIL_OFF") == nullptr && (fl_env == nullptr || atoi(fl_env) > 0);
    fprintf(stderr, "note: final-window flush %s\n", flush_tail ? "ACTIVE (last window encodes only real frames)"
                                                                : "OFF (fixed 26-frame tail padding, pre-Exp829 protocol)");
    const int piece_samples = WINDOW_SAMPLES / params.vae_pieces;
    const int piece_frames = FRAMES_PER_WINDOW / params.vae_pieces;
    fprintf(stderr, "VAE pieces: %d x %d samples (%d frames each)%s\n\n",
            params.vae_pieces, piece_samples, piece_frames,
            vcache ? " [streaming cache]" : " [legacy full-window]");

    std::string full_text;
    double gen_start = now_ms();
    double vae_ms = 0, lm_ms = 0;
    int total_tokens = 0;
    g_prefill_ms = 0.0;
    g_decode_ms = 0.0;
    g_ac_ms = 0.0;
    g_sem_ms = 0.0;

    const char * strip_list[] = {"<|text_chunk_end|>", "<|object_ref_start|>", "<|object_ref_end>",
                                 "<|box_start|>", "<|speech_start|>", "<|speech_end|>", "<|speech_pad|>"};

    // ---- cross-window carry mode (full-context features) ----
    // Hops tile the audio contiguously (83200, then 70400s, last zero-padded);
    // the VAE cache is reset once, so every hop after the first sees full left
    // context (like upstream encode_then_split, but bounded memory). Chunk h is
    // complete at step h (hop h covers through frame 22h+25), so emission is
    // immediate, exactly like the legacy loop.
    if (params.xwin && n_samples > WINDOW_SAMPLES) {
        int n_hops = 1 + (n_samples - WINDOW_SAMPLES + HOP_SAMPLES - 1) / HOP_SAMPLES;
        // frames after all hops: 26 + 22*(n_hops-1); chunks while 22k+25 < F
        int total_frames = FRAMES_PER_WINDOW + (n_hops - 1) * (HOP_SAMPLES / 3200);
        int n_chunks = 0;
        while (22 * n_chunks + 25 < total_frames) n_chunks++;
        fprintf(stderr, "XWIN: %d hops, %d frames, %d chunks (carry, immediate emission)\n\n",
                n_hops, total_frames, n_chunks);
        vae_cache_t * xvc = vcache;
        bool own_cache = false;
        if (xvc == nullptr) { xvc = vae_cache_new(); own_cache = true; }
        vae_cache_reset(xvc);
        std::vector<float> wafe(FRAMES_PER_WINDOW * acoustic_dim);
        std::vector<float> wsfe(FRAMES_PER_WINDOW * semantic_dim);
        std::vector<float> frames;
        frames.reserve((size_t)total_frames * n_embd);
        std::vector<float> hop(std::max(WINDOW_SAMPLES, HOP_SAMPLES), 0.0f);
        for (int h = 0; h < n_hops; h++) {
            // Bounded carry: periodic cold reset stops context drift from
            // compounding on long files (unbounded carry degrades 69s to 12%
            // WER; cold windows hold 3.7%). Short files never hit the reset.
            if (params.xwin_reset > 0 && h > 0 && h % params.xwin_reset == 0) {
                vae_cache_reset(xvc);
            }
            int start = (h == 0) ? 0 : WINDOW_SAMPLES + (h - 1) * HOP_SAMPLES;
            int want = (h == 0) ? WINDOW_SAMPLES : HOP_SAMPLES;
            int avail = std::min(want, n_samples - start);
            if (avail <= 0) break;
            memcpy(hop.data(), audio.samples.data() + start, avail * sizeof(float));
            if (avail < want) memset(hop.data() + avail, 0, (want - avail) * sizeof(float));
            // Partial (zero-padded) final hop: reset to cold first. Carried speech
            // state makes padding look like continued speech and the LM
            // hallucinates repetitions on the last chunk; cold padding decodes
            // as silence (matches legacy). Only the final hop can be partial.
            if (avail < want) vae_cache_reset(xvc);
            int gotf = (want == WINDOW_SAMPLES) ? FRAMES_PER_WINDOW : want / 3200;
            if (encode_frames(vae_ctx, xvc, hop.data(), want, wafe.data(), wsfe.data(),
                              acoustic_dim, semantic_dim, gotf, vae_ms) < 0) {
                fprintf(stderr, "xwin hop %d: encode failed\n", h);
                return 1;
            }
            for (int f = 0; f < gotf; f++)
                for (int d = 0; d < n_embd; d++)
                    frames.push_back(wafe[f * acoustic_dim + d] + wsfe[f * semantic_dim + d]);
            if (h >= n_chunks) {
                fprintf(stderr, "xwin: more hops than chunks, stopping\n");
                break;
            }
            if (emit_chunk(lctx, smpl, model, frames.data() + (size_t)h * 22 * n_embd,
                           n_embd, params.n_batch, params.max_tokens_per_chunk,
                           strip_list, 7, pos, lm_ms, total_tokens, full_text,
                           h + 1, n_chunks) < 0) return 1;
        }
        if (own_cache) vae_cache_free(xvc);
        n_windows = n_chunks;  // for the summary line below
    } else {
    // Exp949: the context-exhaustion path used to be a bare "frames failed" + return 1 with NO summary, so a
    // recording longer than the context failed hard and the partial transcript was lost to the caller (the
    // per-window text did reach stdout, but nothing said why the run stopped). Measured at Exp948: the default
    // -c 4096 covers ~258 s of DENSE audio (chat276 = 276 s died at window 85 of 95; the same clip completes at
    // -c 8192). This path is unreachable for any input that fits, so normal outputs are unchanged - verified by
    // the protocol transcript hash and the 40-utt gate. The original message is kept verbatim because the
    // fault board greps for it. NB the lambda must live INSIDE a braced block: inserted directly after `} else`
    // it became the else-branch's substatement, went out of scope immediately, and the build failed (Exp949b).
    auto context_note = [&](int at_window) {
        fprintf(stderr, "context exhausted at window %d/%d: n_ctx=%d is full, so %d window(s) of the audio "
                        "were NOT transcribed\n", at_window, n_windows, params.n_ctx, n_windows - at_window);
        fprintf(stderr, "  raise -c for longer sessions (-c 4096 ~= 258 s of dense audio, -c 8192 ~= 8.6 min, "
                        "+112 MB of KV)\n");
        fprintf(stderr, "  partial transcript (%d tokens so far):\n%s\n", total_tokens, full_text.c_str());
    };
    for (int w = 0; w < n_windows; w++) {
        // LATENCY_TRACE=1: per-window leaves of the latency tree (ms, cumulative
        // counters differenced). Zero cost when unset; measurement only.
        const double vae_prev = vae_ms, lm_prev = lm_ms;
        const double pre_prev = g_prefill_ms, dec_prev = g_decode_ms;
        int start = w * HOP_SAMPLES;
        int avail = std::min(WINDOW_SAMPLES, n_samples - start);
        if (avail <= 0) break;
        // Exp829 FLUSH_TAIL: the final window of a clip is mostly tail padding under the fixed-window
        // protocol (measured: 0.653 of it on the 10 s protocol clip), and that padding costs BOTH VAE time
        // (3.45 s per 26-frame window) and LM prefill rows (1.07 s/window). Flushing encodes only the real
        // samples, rounded up to the 6400-sample (2-frame) granularity the cached encoder asserts.
        // Off by default: this changes what the model sees in the final window, so it is gated on the
        // 40-utt WER parity test before it can ever become the default.
        int want = WINDOW_SAMPLES;
        if (flush_tail && avail < WINDOW_SAMPLES) {
            want = ((avail + 6399) / 6400) * 6400;
            if (want > WINDOW_SAMPLES) want = WINDOW_SAMPLES;
        }
        memcpy(window.data(), audio.samples.data() + start, avail * sizeof(float));
        if (avail < want) memset(window.data() + avail, 0, (want - avail) * sizeof(float));
        // A flushed final window has fewer frames than the protocol grid; the non-deferred branch overwrites
        // this from the pieces it actually encoded, the deferred branch takes it from the window length.
        int win_frames = flush_tail ? (want / 3200) : FRAMES_PER_WINDOW;

        double t0 = now_ms();
        ggml_mm_set_phase(1);   // vae: the shipped loop's own span (Exp816). NOTE the first attempt hooked
                                // emit_chunk/piece-wise helpers, which the shipped path does not call, and the
                                // census then printed everything as phase=idle - the output-change rule again.
        if (vcache) {
            // Piece-wise encode with carried conv state (reset per window =
            // upstream cold-window parity). Pieces are 3200-multiples, so every
            // strided layer grid stays aligned and outputs tile exactly.
            vae_cache_reset(vcache);
            // Deferred late stages are the DEFAULT: the early stages run
            // piece-wise as usual (their activations are what the RAM budget
            // allows) but write their boundary tensors into a window-sized
            // buffer; the deep stages then run ONCE for the whole window, which
            // turns the deep GEMVs into window-level GEMMs (measured on device:
            // VAE 26.0 -> 24.2 s, RTF -5%, RSS unchanged, transcript differs by
            // one proper-noun token). VAE_DEFER_LATE=1 restores the deferred
            // path (needed by fine granularity: at p13/p26 the deep layers are
            // GEMV-shaped per piece, so window-batching wins ~5% - Exp708); the
            // lean tier recipe sets it explicitly. VAE_LATE_SPLIT overrides the
            // stage split (default 6).
            const char * defer_env = getenv("VAE_DEFER_LATE");
            const bool defer_late =
                ((defer_env != nullptr) && (atoi(defer_env) > 0)) &&
                // RAM-lean mode keeps its single shared arena (both encoders on
                // slot 0, sequential), which the deferred path does not use, so
                // VAE_SEQ_ENCODERS=1 falls back to the piece-wise encode.
                (getenv("VAE_SEQ_ENCODERS") == nullptr);
            if (defer_late) {
                // Deferred late stages (VAE_DEFER_LATE=1): the early stages run
                // piece-wise as usual (their activations are what the RAM budget
                // allows) but write their boundary tensors into a window-sized
                // buffer; the deep stages then run ONCE for the whole window,
                // which turns the deep GEMVs into L=26 GEMMs and reads the deep
                // weights once per window instead of once per piece. Equivalent
                // by the cache invariant (piece-wise early == full window).
                const size_t scratch_n = (size_t)piece_samples * 128 + 4096;
                std::vector<float> asct(scratch_n), ssct(scratch_n);
                std::vector<float> abnd, sbnd;
                int64_t ashape[4] = {0,0,0,0}, sshape[4] = {0,0,0,0};
                int64_t tpiece = 0, cch_a = 0, cch_s = 0;
                // Exp830: the flush works here too. What made a short window impossible in this branch was
                // the boundary SPACING: buffers were laid out with per-channel stride tpiece * pieces, but
                // vae_encode_late_impl takes the stride from n_time_total and uses bshape[0] only to validate,
                // so the right quantity is the number of frames PRESENT in this window - want/3200, which for
                // a full window equals tpiece * pieces exactly. Nothing changes off the final window.
                // NOTE the boundary is at the LATE-SPLIT stage, so its frame count per piece (ashape[0]) is
                // downsampled relative to the audio grid - frames present cannot be derived from sample counts.
                // Buffers are therefore allocated and written with the full-window spacing and packed down to
                // the actual length afterwards (a few KB per channel), which keeps the full-window path
                // byte-identical and makes a short final window self-consistent.
                int64_t T_full = 0;      // boundary frames in a FULL window = tpiece * pieces (set at piece 0)
                int64_t dfr = 0;                            // boundary frames written so far
                int rem_defer = want;
                for (int p = 0; p < params.vae_pieces && rem_defer > 0; p++) {
                    const float * piece = window.data() + p * piece_samples;
                    const int nsamp = std::min(piece_samples, rem_defer);
                    float ac_ms = 0.0f, sem_ms = 0.0f;
                    if (vae_encode_early_parallel_cached(vae_ctx, vcache, piece, nsamp,
                                                         asct.data(), ashape, ssct.data(), sshape,
                                                         &ac_ms, &sem_ms) < 0) {
                        fprintf(stderr, "window %d piece %d: early encode failed\n", w, p);
                        return 1;
                    }
                    g_ac_ms += ac_ms; g_sem_ms += sem_ms;
                    if (p == 0) {
                        tpiece = ashape[0]; cch_a = ashape[1]; cch_s = sshape[1];
                        // Boundary layout is [T, C] (time fastest; the stage-level
                        // cont(permute(1,0,2,3)) guarantees ne0 = time). Only the
                        // per-encoder shape agreement and the scratch fit are
                        // checked here; C can be smaller than T for deep splits.
                        if (tpiece <= 0 || cch_a <= 0 || cch_s <= 0 ||
                            ashape[0] * ashape[1] > (int64_t)scratch_n ||
                            sshape[0] != ashape[0] ||
                            sshape[0] * sshape[1] > (int64_t)scratch_n) {
                            fprintf(stderr, "window %d: unexpected boundary shape a=[%lld,%lld] s=[%lld,%lld]\n",
                                    w, (long long)ashape[0], (long long)ashape[1],
                                    (long long)sshape[0], (long long)sshape[1]);
                            return 1;
                        }
                        T_full = tpiece * params.vae_pieces;   // 16 boundary frames per 2-frame piece at p13
                        abnd.assign((size_t)T_full * cch_a, 0.0f);
                        sbnd.assign((size_t)T_full * cch_s, 0.0f);
                    } else if (ashape[1] != cch_a || sshape[1] != cch_s) {   // frame count may shrink on a short final piece
                        fprintf(stderr, "window %d piece %d: boundary shape drift\n", w, p);
                        return 1;
                    }
                    const int64_t tf = ashape[0];          // boundary frames this piece produced
                    if (tf <= 0 || dfr + tf > T_full) {
                        fprintf(stderr, "window %d piece %d: boundary frames %lld overflow %lld\n",
                                w, p, (long long)tf, (long long)T_full);
                        return 1;
                    }
                    // Strided tile: piece p's channel c row (tf contiguous floats at c*tf) lands at
                    // offset c*T_win + dfr - the sequence is packed, so no hole is left by a short piece.
                    for (int64_t c = 0; c < cch_a; c++)
                        memcpy(abnd.data() + (size_t)c * T_full + (size_t)dfr,
                               asct.data() + (size_t)c * tf, (size_t)tf * sizeof(float));
                    for (int64_t c = 0; c < cch_s; c++)
                        memcpy(sbnd.data() + (size_t)c * T_full + (size_t)dfr,
                               ssct.data() + (size_t)c * tf, (size_t)tf * sizeof(float));
                    dfr += tf;
                    rem_defer -= nsamp;
                }
                if (dfr <= 0 || T_full <= 0) { fprintf(stderr, "window %d: no boundary frames\n", w); return 1; }
                if (dfr < T_full) {          // flushed final window: re-space each channel to length dfr
                    for (int64_t c = 1; c < cch_a; c++)
                        memmove(abnd.data() + (size_t)c * dfr, abnd.data() + (size_t)c * T_full,
                                (size_t)dfr * sizeof(float));
                    for (int64_t c = 1; c < cch_s; c++)
                        memmove(sbnd.data() + (size_t)c * dfr, sbnd.data() + (size_t)c * T_full,
                                (size_t)dfr * sizeof(float));
                }
                float ac_ms = 0.0f, sem_ms = 0.0f;
                int nfr = vae_encode_late_parallel(vae_ctx, abnd.data(), ashape,
                                                   sbnd.data(), sshape, dfr,
                                                   afe.data(), sfe.data(), &ac_ms, &sem_ms);
                g_ac_ms += ac_ms; g_sem_ms += sem_ms;
                // nfr is the number of OUTPUT (audio) frames the late pass produced, not boundary frames:
                // for a flushed window that is want/3200, for a full window FRAMES_PER_WINDOW.
                if (nfr != win_frames) {
                    fprintf(stderr, "window %d: late stages returned %d frames (want %d)\n",
                            w, nfr, win_frames);
                    return 1;
                }
                win_frames = nfr;
            } else {
            int rem_samples = want;
            int got_frames = 0;
            for (int p = 0; p < params.vae_pieces && rem_samples > 0; p++) {
                const float * piece = window.data() + p * piece_samples;
                float * af = afe.data() + p * piece_frames * acoustic_dim;
                float * sf = sfe.data() + p * piece_frames * semantic_dim;
                const int nsamp = std::min(piece_samples, rem_samples);   // short final piece when flushing
                af = afe.data() + (size_t)got_frames * acoustic_dim;      // frames pack contiguously
                sf = sfe.data() + (size_t)got_frames * semantic_dim;
                int na, ns;
                // Concurrent encoders are the DEFAULT (they are independent and
                // the per-encoder math is split-invariant: transcripts are
                // byte-identical to the sequential path). VAE_SEQ_ENCODERS=1
                // restores the sequential pair for RAM-critical runs.
                if (getenv("VAE_SEQ_ENCODERS") == nullptr) {
                    float ac_ms = 0.0f, sem_ms = 0.0f;
                    na = ns = vae_encode_parallel_cached(vae_ctx, vcache, piece, nsamp,
                                                         af, sf, &ac_ms, &sem_ms);
                    g_ac_ms += ac_ms; g_sem_ms += sem_ms;
                } else {
                    double ta2 = now_ms();
                    na = vae_encode_acoustic_cached(vae_ctx, vcache, piece, nsamp, af);
                    g_ac_ms += now_ms() - ta2;
                    double ts2 = now_ms();
                    ns = vae_encode_semantic_cached(vae_ctx, vcache, piece, nsamp, sf);
                    g_sem_ms += now_ms() - ts2;
                }
                const int want_f = nsamp / 3200;
                if (na != want_f || ns != want_f) {
                    fprintf(stderr, "window %d piece %d: unexpected frames a=%d s=%d (want %d)\n",
                            w, p, na, ns, want_f);
                    return 1;
                }
                got_frames += na;
                rem_samples -= nsamp;
            }
            win_frames = got_frames;
            }
        } else {
            int na = vae_encode_acoustic(vae_ctx, window.data(), WINDOW_SAMPLES, afe.data());
            int ns = vae_encode_semantic(vae_ctx, window.data(), WINDOW_SAMPLES, sfe.data());
            if (na != FRAMES_PER_WINDOW || ns != FRAMES_PER_WINDOW) {
                fprintf(stderr, "window %d: unexpected frames a=%d s=%d\n", w, na, ns);
                return 1;
            }
        }
        vae_ms += now_ms() - t0;
        // sum acoustic+semantic (both are 1536-dim, connector already applied)
        for (int f = 0; f < win_frames; f++)
            for (int d = 0; d < n_embd; d++)
                speech_emb[f * n_embd + d] = afe[f * acoustic_dim + d] + sfe[f * semantic_dim + d];

        t0 = now_ms();
        ggml_mm_set_phase(2);   // lm_prefill (shipped inline path)
        llama_token t_start = TOK_SPEECH_START, t_end = TOK_SPEECH_END, t_tce = TOK_TEXT_CHUNK_END;
        // defer_now's definition, recomputed here because that name is scoped to the cache-setup block above.
        const char * denv2 = getenv("VAE_DEFER_LATE");
        const bool defer_here = (denv2 != nullptr) && (atoi(denv2) > 0) && (getenv("VAE_SEQ_ENCODERS") == nullptr);
        const bool bound_on = bound_batch && params.vae_pieces == 1 && !defer_here && !params.xwin;
        if (bound_on) {
            if (!bound_ready) {
                if (llama_token_embd_row(model, t_start, chunk_emb.data()) != n_embd) {
                    fprintf(stderr, "boundary embedding lookup failed\n"); return 1;
                }
                bound_ready = true;
            }
            // The end token goes AFTER the frames that exist in THIS window - with the tail flush the final
            // window has fewer than FRAMES_PER_WINDOW of them, so its row is not a constant index (Exp835:
            // assuming 27 fed a garbage row on the flushed window and the LM rambled to 57 tokens).
            memcpy(chunk_emb.data() + n_embd, speech_emb.data(), (size_t)win_frames * n_embd * sizeof(float));
            if (llama_token_embd_row(model, t_end,
                                    chunk_emb.data() + (size_t)(win_frames + 1) * n_embd) != n_embd) {
                fprintf(stderr, "boundary embedding lookup failed\n"); return 1;
            }
            if ((pos = feed_embeds(lctx, chunk_emb.data(), win_frames + 2, n_embd, pos,
                                   params.n_batch)) < 0) { fprintf(stderr, "frames failed\n"); context_note(w + 1); return 1; }
        } else {
            if ((pos = feed_token(lctx, t_start, pos)) < 0) { fprintf(stderr, "sp_start failed\n"); context_note(w + 1); return 1; }
            if ((pos = feed_embeds(lctx, speech_emb.data(), win_frames, n_embd, pos, params.n_batch)) < 0) {
                fprintf(stderr, "frames failed\n"); context_note(w + 1); return 1;
            }
            if ((pos = feed_token(lctx, t_end, pos)) < 0) { fprintf(stderr, "sp_end failed\n"); context_note(w + 1); return 1; }
        }
        g_prefill_ms += now_ms() - t0;
        ggml_mm_set_phase(3);   // lm_decode
        double tdec = now_ms();
        llama_token tok = llama_sampler_sample(smpl, lctx, -1);
        llama_sampler_accept(smpl, tok);

        std::vector<llama_token> chunk_ids;
        for (int i = 0; i < params.max_tokens_per_chunk; i++) {
            if (tok == TOK_TEXT_CHUNK_END || tok == TOK_EOS) break;
            chunk_ids.push_back(tok);
            if ((pos = feed_token(lctx, tok, pos)) < 0) { fprintf(stderr, "decode failed\n"); return 1; }
            tok = llama_sampler_sample(smpl, lctx, -1);
            llama_sampler_accept(smpl, tok);
        }
        // always end the cache on text_chunk_end (upstream invariant)
        if ((pos = feed_token(lctx, t_tce, pos)) < 0) { fprintf(stderr, "tce failed\n"); return 1; }
        g_decode_ms += now_ms() - tdec;
        ggml_mm_set_phase(0);
        lm_ms += now_ms() - t0;

        std::string text = detokenize(model, chunk_ids);
        for (auto s : strip_list) {
            size_t p;
            while ((p = text.find(s)) != std::string::npos) text.erase(p, strlen(s));
        }
        full_text += text;
        total_tokens += (int)chunk_ids.size();
        if (getenv("LATENCY_TRACE") != nullptr)
            fprintf(stderr, "LT w=%d vae=%.0f prefill=%.0f decode=%.0f tok=%d sum=%.0f\n",
                    w + 1, vae_ms - vae_prev, g_prefill_ms - pre_prev,
                    g_decode_ms - dec_prev, (int)chunk_ids.size(),
                    (vae_ms - vae_prev) + (lm_ms - lm_prev));
        printf("[%d/%d] %s\n", w + 1, n_windows, text.c_str());
        fflush(stdout);
    }
    }   // end of the windowed (non-xwin) branch opened by `} else {` above

    double gen_s = (now_ms() - gen_start) / 1000.0;
    double total_s = (now_ms() - total_start) / 1000.0;
    double load_s = (load_done - total_start) / 1000.0;
    double rtf = gen_s / audio.duration_sec;

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, " Audio: %.2fs | chunks: %d | tokens: %d | RTF: %.4f\n",
            audio.duration_sec, n_windows, total_tokens, rtf);
    fprintf(stderr, " load: %.1fs | VAE: %.1fs (ac %.1fs, sem %.1fs) | LM: %.1fs (prefill %.1fs, decode %.1fs) | total: %.1fs\n",
            load_s, vae_ms / 1000.0, g_ac_ms / 1000.0, g_sem_ms / 1000.0, lm_ms / 1000.0, g_prefill_ms / 1000.0, g_decode_ms / 1000.0, total_s);
    fprintf(stderr, "========================================\n");
    printf("\n--- Transcription ---\n%s\n", full_text.c_str());

    llama_sampler_free(smpl);
    if (vcache) vae_cache_free(vcache);
    llama_free(lctx);
    llama_free_model(model);
    llama_backend_free();
    vae_free(vae_ctx);
    vae_free_model(vae_model);
    return 0;
}
