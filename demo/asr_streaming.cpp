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
    int n_ctx = 4096;  // covers ~15 min audio; KV 112MB vs 450MB at 16384
    int n_batch = 512;
    int max_tokens_per_chunk = 256;
    int vae_pieces = 13;  // window split count; must divide 26 (frames). 13x6400 or 26x3200.
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
    fprintf(stderr, "  -c <n>                 Context size (default: 16384)\n");
    fprintf(stderr, "  --max-tokens <n>       Max new tokens per chunk (default: 256)\n");
    fprintf(stderr, "  --vae-pieces <n>       Window split count, must divide 26 (default: 13)\n");
    fprintf(stderr, "                         1 = legacy full-window encode; 13/26 = cached pieces\n");
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
    llama_token t_start = TOK_SPEECH_START, t_end = TOK_SPEECH_END, t_tce = TOK_TEXT_CHUNK_END;
    if ((pos = feed_token(lctx, t_start, pos)) < 0) return -1;
    if ((pos = feed_embeds(lctx, frames, FRAMES_PER_WINDOW, n_embd, pos, n_batch)) < 0) return -1;
    if ((pos = feed_token(lctx, t_end, pos)) < 0) return -1;
    g_prefill_ms += now_ms() - t0;
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
    return (got == want_frames) ? got : -1;
}


int main(int argc, char ** argv) {
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

    // ---- windows ----
    int n_samples = (int)audio.samples.size();
    int n_windows = (n_samples + HOP_SAMPLES - 1) / HOP_SAMPLES;
    fprintf(stderr, "Windows: %d (window=%d hop=%d)\n\n", n_windows, WINDOW_SAMPLES, HOP_SAMPLES);

    std::vector<float> window(WINDOW_SAMPLES);
    std::vector<float> afe(FRAMES_PER_WINDOW * acoustic_dim);
    std::vector<float> sfe(FRAMES_PER_WINDOW * semantic_dim);
    std::vector<float> speech_emb(FRAMES_PER_WINDOW * n_embd);
    vae_cache_t * vcache = (params.vae_pieces > 1) ? vae_cache_new() : nullptr;
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
    } else
    for (int w = 0; w < n_windows; w++) {
        int start = w * HOP_SAMPLES;
        int avail = std::min(WINDOW_SAMPLES, n_samples - start);
        if (avail <= 0) break;
        memcpy(window.data(), audio.samples.data() + start, avail * sizeof(float));
        if (avail < WINDOW_SAMPLES) memset(window.data() + avail, 0, (WINDOW_SAMPLES - avail) * sizeof(float));

        double t0 = now_ms();
        if (vcache) {
            // Piece-wise encode with carried conv state (reset per window =
            // upstream cold-window parity). Pieces are 3200-multiples, so every
            // strided layer grid stays aligned and outputs tile exactly.
            vae_cache_reset(vcache);
            for (int p = 0; p < params.vae_pieces; p++) {
                const float * piece = window.data() + p * piece_samples;
                float * af = afe.data() + p * piece_frames * acoustic_dim;
                float * sf = sfe.data() + p * piece_frames * semantic_dim;
                double ta2 = now_ms();
                int na = vae_encode_acoustic_cached(vae_ctx, vcache, piece, piece_samples, af);
                g_ac_ms += now_ms() - ta2;
                double ts2 = now_ms();
                int ns = vae_encode_semantic_cached(vae_ctx, vcache, piece, piece_samples, sf);
                g_sem_ms += now_ms() - ts2;
                if (na != piece_frames || ns != piece_frames) {
                    fprintf(stderr, "window %d piece %d: unexpected frames a=%d s=%d (want %d)\n",
                            w, p, na, ns, piece_frames);
                    return 1;
                }
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
        for (int f = 0; f < FRAMES_PER_WINDOW; f++)
            for (int d = 0; d < n_embd; d++)
                speech_emb[f * n_embd + d] = afe[f * acoustic_dim + d] + sfe[f * semantic_dim + d];

        t0 = now_ms();
        llama_token t_start = TOK_SPEECH_START, t_end = TOK_SPEECH_END, t_tce = TOK_TEXT_CHUNK_END;
        if ((pos = feed_token(lctx, t_start, pos)) < 0) { fprintf(stderr, "sp_start failed\n"); return 1; }
        if ((pos = feed_embeds(lctx, speech_emb.data(), FRAMES_PER_WINDOW, n_embd, pos, params.n_batch)) < 0) {
            fprintf(stderr, "frames failed\n"); return 1;
        }
        if ((pos = feed_token(lctx, t_end, pos)) < 0) { fprintf(stderr, "sp_end failed\n"); return 1; }
        g_prefill_ms += now_ms() - t0;
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
        lm_ms += now_ms() - t0;

        std::string text = detokenize(model, chunk_ids);
        for (auto s : strip_list) {
            size_t p;
            while ((p = text.find(s)) != std::string::npos) text.erase(p, strlen(s));
        }
        full_text += text;
        total_tokens += (int)chunk_ids.size();
        printf("[%d/%d] %s\n", w + 1, n_windows, text.c_str());
        fflush(stdout);
    }

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
