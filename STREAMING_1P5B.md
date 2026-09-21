# Streaming ASR (VibeVoice-ASR-Streaming-1.5B) on CPU

This fork extends VibeASR.cpp with chunked streaming inference for
[`microsoft/VibeVoice-ASR-Streaming-1.5B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B)
(Qwen2.5-1.5B decoder, per-2.93 s chunks with `Speaker N:` labels), complementing
the built-in offline BitNet path (`VibeVoice-ASR-BitNet`).

## Accuracy (validated 2026-09-09)

40-utterance LibriSpeech `test-clean` subset, jiwer, same normalization for all:

| system | vs ground truth | vs official PyTorch |
|---|---|---|
| `asr_streaming` (VAE F16 + LM Q4_K_M, `--vae-pieces 13`) | **WER 4.13%** (S=25 D=1 I=4) | **WER 2.2%** |
| official PyTorch `streaming_generate` (bf16) | WER 4.82% (S=28 D=3 I=4) | — |

Delta on ground truth: **-0.7 pp** (within subset noise) — accuracy is preserved.
69 s in-domain clip: parity WER 3.7% (11 substitutions, 0 del/ins).
5/5 reruns bit-identical (`-t 4`).

## Build

```bash
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target asr_streaming -j
```

## Convert weights (source: `microsoft/VibeVoice-ASR-Streaming-1.5B`)

The streaming checkpoint shares the VAE architecture and Qwen2.5-1.5B decoder
with BitNet, but its LM weights are normal BF16 (not ternary), so the BitNet
LM converter is bypassed:

```bash
# VAE encoder (works as-is, same tensor names)
python utils/convert_vae_to_gguf.py <streaming-checkpoint-dir> --outtype f16 \
    -o streaming-vae-encoder-f16.gguf          # ~1.4 GB
# LM (strips model.language_model.* -> model.*, flattens decoder_config, no ternary step)
python utils/convert_streaming_lm_stage.py <streaming-checkpoint-dir>
# -> models-streaming/streaming-lm-f16.gguf (~4.2 GB)
./build/bin/llama-quantize --token-embedding-type Q6_K \
    streaming-lm-f16.gguf streaming-lm-q4_k_m.gguf Q4_K_M   # ~1.1 GB
```

## Run

```bash
./build/bin/asr_streaming --vae-model streaming-vae-encoder-f16.gguf \
    --lm-model streaming-lm-q4_k_m.gguf --audio input_24k.wav -t 4 \
    [--vae-pieces 13] [--context "VibeVoice,diarization"] [--max-tokens 256]
```

`--vae-pieces` must divide 26 (1, 2, 13, 26; default 13). Each 83200-sample
window is encoded in 6400-sample pieces with carried conv state (reset per
window = upstream cold-window parity); frames are bit-exact vs full-window
encode. Peak RSS is flat in file length: 3.26 GB (17 s) -> 3.27 GB (69 s),
vs 12.5 GB for PyTorch CPU fp32 and 5.5 -> 15.5 GB for offline BitNet on long files.

## Implementation notes

- `demo/asr_streaming.cpp`: faithful port of upstream `streaming_generate`
  (plain-text prompt, `[speech_start]+26f+[speech_end]`, greedy to
  `<|text_chunk_end|>` 151665, feed-it-back invariant, per-chunk prints).
- `src/vae.{h,cpp}`: `vae_cache_t` streaming conv cache (histories ~0.65 MB/encoder).
  History splicing uses pad + add, deliberately NOT `ggml_concat`: this build's
  concat scrambles non-trivial inputs (unit-tested), while pad/add are proven ops.
  ggml dense tensors are Fortran-order (ne[0] fastest) — history gather/scatter
  must be per-channel strided copies, not one linear memcpy.
- RF of the VAE encoder exceeds 72 frames (> 26-frame window), so overlap-split
  cannot work (verified: prepending 83k zeros changes all 26 frames); the
  carried-state cache is required. The stock BitNet I8_S VAE path is untouched
  (`asr_infer` behavior unchanged); the streaming VAE stays F16 because naive
  PTQ I8_S collapses output (repetition loops) — that VAE was never QAT-trained.
- Env-gated diagnostics: `VAE_GRAPH_STATS=1` (per-op byte-traffic profile of the
  first built graph - MUL_MAT ~32%, IM2COL ~11%, ADD ~10%, MUL/gelu ~7% of a
  ~700 MB/piece materialising-op total; RESHAPE/PERMUTE are views and cost
  nothing), `VAE_CACHE_TRACE=1` (per-site checksums),
  `VAE_DUMP_FRAMES=<prefix>` (output frames), `VAE_DUMP_SITE=<sN|all>` (site inputs).\n
**Quote WER with its tool.** Two scorers disagree slightly on the 40-utt gate because their error accounting differs on
1-2 of 731 token alignments: `score_hyp.py` implies a 726-token reference (4.68 %) where `.auto/compare_arms.py`
uses 731 (4.65 %), with identical S+D+I. Convention: **absolute** accuracy numbers come from `score_hyp.py`,
**between-system** claims come from `compare_arms.py` (paired, McNemar + bootstrap), and the two are never mixed in
one sentence. Re-deriving a scorer would re-quote every historical number, so this stays a quoting rule (Exp847).

## Reproduce & verify (Exp513)

Both deployed artifacts are **bit-exactly reproducible** from `models-pt` with
the frozen recipes, and the copies on the phone match the host byte-for-byte:

| artifact | recipe | md5 |
|---|---|---|
| `vae-encoder-convint8.gguf` (**default since Exp690**) | `python3 .auto/conv_int8.py vae-encoder-q4x4ffn.gguf out.gguf --device` - needs `.auto/quant4x4_arm` (NDK clang++ against build-android/libggml.so). `--device` is mandatory, not cosmetic: x86 `ggml_quantize_chunk(Q4_0_4_4)` writes valid nibbles with **zero scales**. Verify: `VAE_CONV_I8_CMP=1 VAE_CONV_I8_REF=<source> ` (prints the dequant-vs-source ratio plus its own control) | `52884a747af5aa1d…` (sha256) |
| `vae-encoder-q4x4ffn.gguf` | `python3 utils/convert_vae_to_gguf.py models-pt --outtype q4_0_4x4_ffn -o out.gguf` (33 s) - the F16-conv reference build | `b909b7901d5d318d81ce4cbaeac31437` |
| `lm-q4_0_4_4.gguf` | `llama-quantize --allow-requantize --token-embedding-type q6_K streaming-lm-q4_k_m.gguf out.gguf Q4_0_4_4` (1.2 s) | `db67eecbd31bba707666414dd977902f` |
| `streaming-lm-q4_k_m.gguf` (intermediate) | `utils/convert_streaming_lm_stage.py` + quantize | `046be3d4775e10f8b635b03ec1bc79cb` |
| Android build (`asr_streaming`) | `cmake -B build-android ... -DCMAKE_C_FLAGS="-mcpu=cortex-a78"` (`.auto/setup.sh`) | `98b643ed2496cff89d9b8caec687c3c0` |

Build fingerprint on the phone: `libggml.so 40a1f284e75dbc3c180215267068c12e`,
`libllama.so 92ad2456979d99e2a1afee4a8cebad1d`. To re-run the full tier
measurement: `./.auto/measure.sh` (VAE_FILE now defaults to `vae-encoder-convint8.gguf` and
LM_FILE to `lm-q8head.gguf`; the script hash-checks both against the device and pushes on
difference - before Exp689 it pushed neither, so an experiment could measure stale device bytes and
report them as a result). 1 piece is the harness default (PIECES=1 since v4.0, Exp706; p2 is the documented fallback); the binary `--vae-pieces` flag itself
defaults to 13. WER gate: `./.auto/eval40.sh <tag> 0 40` +
`venv-vibe/bin/python .auto/score_hyp.py <tag>`; for paired output-equivalence use
`.auto/compare_arms.py --gate hyp-A hyp-B eval-librispeech/refs.json`.

### The RTF denominator is content-derived, not header-derived (verified, Exp850)

A WAV whose RIFF `data` size lies (inflated 4x, real data unchanged) does **not** flatter the metric: the pipeline
iterates over decoded samples, so it did the honest 10 s of work (`vae_s 12.0`) and reported rtf 2.00 - identical to
the honest clip. Likewise a file truncated to half its declared length processes the 5 s that exist and reports
rtf 2.03, not a halved number. Header-only and empty WAVs are refused with `[audio_io] Error: ...`. This is asserted
continuously by `.auto/fault_inject.sh`, whose audio section fails if a future change ever lets the header field set
the denominator (the lying-header probe would then read ~0.5 instead of ~2.0). Worth stating explicitly because
every published RTF in this document is only meaningful if the clip is honest - the clips themselves are hash-pinned
by `.auto/audit_harness.py`.

### Time-to-first-transcript and incremental emission (measured, Exp853)

Output really is incremental - the window loop does `printf` + `fflush(stdout)`, and sampling the output file's size
during a run shows bytes arriving throughout (0 -> 45 -> 84 -> 135 -> 172 -> 340), not one dump at the end. Measured
**time to first transcript**: **8.13 s** (shipped tier) and **8.90 s** (RAM-lean tier) for the 10 s protocol clip,
then one line per window every ~5.5 s.

TTFT closes against the latency tree: load 1.2 + system-prompt prefill 1.12 + window-1 VAE ~3.9 (incl. a 0.32 s
first-touch page-in) + window-1 prefill 1.07 + decode ~0.9 = ~8.2 s. So ~2.6 s (32 %) of TTFT is FIXED setup that
has nothing to do with the audio. Two consequences for a product:
* A live microphone falls behind by (RTF - 1) = 0.87 s per second of speech - this is a file/batch-friendly engine
  at RTF 1.87, not a sub-realtime streamer.
* TTFT levers are process reuse (kills the 1.2 s load), folding the prompt prefill into window 1's LM batch
  (~1.0 s of wall, +0.14 s of the loop's metric - see the note in the latency-tree section), and pre-touching the
  arena (~0.3 s). None of these improve RTF; judge them on wall clock.

### Page-cache behavior: the RTF numbers are not fragile to other apps (measured, Exp854)

Every log line shows `majflt 0`, so the published RTF is a warm-page-cache number by construction. To test whether
that hides a cold-start penalty, weights were evicted with a 4 GB and then a 12 GB x 2 sequential churn: RTF moved
**+0.7 %** (within state noise) and major faults went **0 -> 1** (not the ~400k an eviction would cause), because
the model pages are the most-recently-touched ones and a single-touch sequential churn preferentially recycles its
own pages. A model file untouched for **9 days** still reads at **3.8 GB/s** (DRAM speed, not flash), i.e. this
device keeps model files cached with real tenacity while free memory exists.

Consequence: co-running workloads that read large files do not degrade this engine's RTF, and idle time does not
either. What remains genuinely UNMEASURED is a cold start after boot or after heavy app-driven reclaim - forcing it
needs `drop_caches` (root; adbd runs as uid 2000, no `su`). Bound if it ever matters: first-touch cannot exceed the
~3.9 GB/s read rate observed here, and LM decode wants 12.6 GB/s, so a fully cold first pass would throttle decode
by roughly 3x until the weights are resident (~0.4 s of unavoidable stall for 1.6 GB, and that is a LOWER bound
because the measured rate is page-cache-served, not flash).

### Device state is a measurement axis: a rebooted phone runs the same binary ~1/3 faster (Exp880)

Every band cell in this document was measured on a phone with days of uptime - and a reboot moves them all.
A spontaneous device reboot (uptime 367 s, caught because `adb` dropped) produced protocol 1.19-1.31 and
138 s 1.49-1.59 against the long-uptime band of 1.85 / 2.16, with byte-identical transcripts (md5 `1a095c8496b4`)
and the same 39 tokens / 876 tokens / 48 windows. A wall-clock cross-check rules out a lying clock:
the binary's 13.2 s total vs 13.63 s host-measured wall. A DELIBERATE `adb reboot` reproduced the fast regime
(~1.31 at ~5 min uptime), so the regime follows the reboot, not a one-time update (build fingerprint
`OPPO/CPH2371 ... R.203be74`, patch 2025-10-01, recorded in `.auto/device_fingerprint.txt`). At 9.75 h uptime
the phone still read 1.18-1.24, so the slow state accumulates over DAYS, not hours - and the "what remains
unmeasured" paragraph of the page-cache section above is now measured: post-boot is not a first-pass stall,
it is a ~30 % regime on every phase (VAE 11.6 -> 7.1 s, LM 7.0 -> 4.8 s on the protocol clip).
Mechanism, partially: sustained cpu7 under load reads 1.43 GHz now vs the ledger's constant 1.3 GHz (+10 %),
which leaves ~20 % to the memory subsystem / accumulated background load - unseparated. Majflt 25-36 on the
first post-boot runs (page-cache refill) does not offset it.
Consequences, all enforced: (1) `measure.sh` prints `note: device uptime_s=... procs=... mem_avail_mb=...
cpu7_khz=...` next to every measurement and WARNS when uptime is under 900 s - a short-uptime number is never
comparable to a band cell, so the harness says so instead of letting a 1.3x read as a code win; (2) the tier
bands stay LONG-UPTIME numbers (headline.json untouched - the 1.85-era cells remain the comparable set);
(3) the loop still lacks the decay curve - QUEUED: protocol samples at ~10 min / 2 h / 6 h / 24 h / 72 h after
one reboot, idle between runs, to map the slow-state timescale and decide whether bands need a state qualifier
or a reboot-before-keep protocol.

### Resource limits: descriptors and threads (measured, Exp856)

Sampled every 10 s through a 138 s run (48 windows, 29 samples): **file descriptors = 3, constant** (min = max = 3,
net +0), **threads 2-3** depending on which phase is running, VmSize ~4.87 GB and RSS flat. No descriptor or thread
leak exists in the shipping path.

The fd count is 3 by design, not by luck: the loader `munmap`s and then **closes the gguf descriptor** (`src/vae.cpp`,
zero-copy path) - a memory mapping stays valid after its fd is closed, so steady-state execution holds only
stdin/stdout/stderr. Consequence: file-descriptor exhaustion cannot end a long session; the limit is 32 768 anyway,
while the real long-session ceiling is KV positions (~258 s, section above).

The counter itself was validated two ways before this row was trusted: `ls /proc/<pid>/fd` and `lsof -p <pid>` both
report exactly 3 for a sleeping process, and `rss_soak.py` agrees. A `No such file or directory` from that path means
the process is GONE, not that permission was denied.

### Ladder extension: 155 s (Exp855, same session as the position-law fit)

`chat155.wav` = chat138 + chat17, 154.97 s, sha256 `1cd4f3a84fbb`, 7 438 748 B: **RTF 2.1935**, 978 tokens,
peak RSS 2208 MB, majflt 0. Against the 138 s cell (2.183) this is +0.5 % for 12 % more windows, i.e. the length
gradient stays gentle exactly as the corrected position law predicts.

### Loader equivalence and config edges (Exp851)

**`--no-mmap` is output-equivalent to the shipped zero-copy loader**: same protocol transcript hash, rtf 1.8734,
`load_s 1.2` - so the fallback path that runs when `mmap` fails produces the same text at the same cost.

Degenerate/invalid configurations are refused or behave plainly, asserted by `.auto/fault_inject.sh` (12 probes):

| invocation | behavior |
|---|---|
| `-c 16` (below one window's rows) | exits 1, `frames failed` - no crash, no hang |
| `--vae-pieces 7` or `0` | exits 1: `pieces must divide 26 (1, 2, 13, 26)` |
| `--max-tokens 0` | **exit 0 with an empty transcript** (the cap is checked before the first token) - silent by design, so a typo here yields nothing rather than an error |
| `-t 1` vs `-t 2` | identical transcript hash (rtf 2.47 vs 1.87) - output is not thread-count specific |

### Corrupt / truncated model files fail loudly (Exp850)

A truncated `.gguf` used to load "successfully": the copy fallback discarded `fread`'s return value into a
zero-initialised buffer, so the missing tail tensors became **zero weights** - fluent, confident, wrong transcripts
with **exit 0 and entirely normal timing** (reproduced with `truncate -s -8MB` -> one word differs; `-64MB` -> most
of the transcript lost). The LM path was already guarded; the VAE path now checks the short read and names the
tensor, e.g.

    [VAE] Error: vae_trunc.gguf is truncated or corrupt: tensor 'semantic.stages.6.7.ffn.linear2.weight'
          needs 9437184 bytes at offset 394527104, read 1122304

`demo` then prints `VAE load failed` and exits 1. Regression board: **`.auto/fault_inject.sh`** (healthy control +
four damaged fixtures; run it after any change to the model loaders). Note the shape of this bug - it is invisible
to every quality metric, because the output stays fluent; only a fault-injection probe finds it.

### Long-session limit (measured, Exp849)

Continuous-audio position budget = **46.5 KV positions per window** (hop 70400 samples = 2.933 s): each window feeds
28 rows and emits ~18.5 tokens, and every emitted token also occupies a position. Measured directly from a 155 s
`LATENCY_TRACE` run (53 windows, 978 tokens, 2493 positions); Exp849's `-c` bracketing said 49.5 +- 3 - consistent.
With the default `n_ctx = 4096`, a single unbroken stream stops after **258 s on dense conversational audio
and 329 s on sparse read speech** - the limit is DENSITY, not seconds, because positions per window =
`28 + tokens emitted per window`. Measured on `long250.wav` (250.8 s of unique Common Voice audio, 86
windows, 728 tokens): 36.5 positions/window, 3140 positions total, and it COMPLETED - which is the limit
law confirmed from below, since 4096 positions at that density would allow 112 windows = 329 s (Exp849's
bracketing gave the same law at chat density: 46.5 pos/window, 88 windows, 258 s).
`n_ctx = 16384` reaches ~16 min. PRICED, not asserted (Exp880): on the 10 s protocol clip, `-c 8192` and
`-c 16384` read -0.8 % and -1.8 % vs `-c 4096` across 3 interleaved reps each (noise, and the sign is
favourable - there is no fixed reservation cost); on `chat138.wav` (1813 positions used) an order-symmetric
A,B,B,A sweep reads `-c 16384` at +0.4 % (an earlier +2.4 % with a fixed arm order was ordering bias, resolved
by the reversal). So raising `-c` costs 0 % of the metric at every length and only RAM: the KV buffer is
COMMITTED at load, not reserved - measured RSS rises 27.4 KB/position (+165 MB at 8192, +388 MB at 16384,
+1681 MB at 65536, all majflt 0), matching the 28.0 KB/position arithmetic. Product form: `-c 8192` doubles
the session to ~8.6 min dense / ~11 min sparse for +112 MB committed. The old code comment attached the
15-minute figure to 4096 - it belongs to 16384.
Failure modes of `-c` (runbook must match all three): (1) exhaustion mid-stream exits 1 with `decode failed`
OR `frames failed` and no final summary; (2) an ABSURD `-c` (28 GB of KV at `-c 1048576`) dies SILENTLY at
context creation - the log ends at `llama_new_context_with_model`, no error, no summary, empty transcript
(Exp880: not matched by the Exp849 rule, and a fresh failure class for any `-c` larger than RAM).

**Position cost law** (same run, 53 windows, R2 0.88, standard error 0.56 us/position, measured to P = 2450):

**Extended to P = 3140 on different content (Exp879, `long250.wav`, 86 windows, 728 tokens, bilingual read
speech):** `decode ms/token = 95.8 + 13.22 us x P` (R2 0.77, slope se 1.23 us) and
`prefill ms/row = 33.9 + 5.00 us x P` (R2 0.88, slope se 0.29 us). The prefill law is confirmed out of range
(-1.3 % on its total), but the 2025-era decode slope was a little shallow: on this run's decode total the old
law gives 77.6 s against 82.5 s measured (-5.9 %) while the refit gives 84.9 s (+2.9 %). The slopes differ by
1.7 sigma, so this is a widening of the same law rather than a contradiction - **use
`decode = 95.8 + 13.2 us x P` for long-session budgeting up to P ~ 3100.** Wall closes again at this length:
287.0 (VAE) + 100.5 (prefill) + 82.5 (decode) + 1.3 (load) = 471.4 s vs rtf x audio = 470.1 s (-0.3 %), and
`vae = 3.337 s/window` over 86 windows against the model's 3.343 + 0.39/N - the VAE term is content-independent
across a 2.5x change in token density, which is the assumption behind every paper-pricing argument here.
RSS 2218 MB at 3140 positions with majflt 0: the KV cache is allocated up front, so memory does NOT grow with
positions - only the position COUNT consumes the budget.

    decode ms/token = 89.1 + 11.13 us x P          prefill ms/row = 33.0 + 5.22 us x P          vae = 3428 +- 9 ms/window (flat)

This **corrects Exp807's 22 us/position**, which understated P by about 2x (it used rows per window, ~28, instead of
46.5) and therefore over-predicts this run's total decode time by 13.2 % (113.2 s predicted vs 100.0 s measured,
against 100.8 s for the corrected law). Refitting the 138 s portion of the clip on its own gives 11.11 us - so the
discrepancy was methodological, not data-dependent. Reading: KV traffic per token per position costs 2.28 us at the
measured 12.6 GB/s ceiling, so the term is ~4.9x its traffic cost - it is mostly attention compute and cache
management, which is why it cannot be bought back by streaming tricks.

On exhaustion the process prints one of `decode failed` / `frames failed` (whichever row group hits the boundary)
and **exits 1 with no final `--- Transcription ---` summary**; window lines emitted so far are already on stdout.
For sessions longer than the limit: raise `-c` (+28 MB per 1024 positions), or restart the session on a boundary.
An 8-bit KV cache would buy **1.88x positions per byte** (q8_0's per-32-element scale costs 6 %), i.e. ~485 s at
4096 slots - but **this build has no `--kv-type` flag**: the cache type is hardcoded to F16 in the vendored llama, and
passing the flag exits 1 with `Unknown arg`. Treating it as an available option is wrong; enabling it means a vendored
change plus a full accuracy gate, and its benefit is capacity, not speed (the position term is ~4.9x traffic-bound). To re-test the boundary cheaply, do NOT run a 4-minute clip - run the 138 s ladder clip with
a small `-c` (e.g. `-c 1536` dies at window 31/48), which reaches the same code path in a fraction of the runtime.

## Measurement model - how to read any number in this repo (Exp968-1016)

Every performance number here is a measurement of a *device state*, not just of code, and the harness now
sets and records that state. Read this before comparing two numbers from different sections or sessions.

**1. Three clock states, settable and witnessed.** The two big cores deliver very different frequencies
depending on whether the phone has recent user activity and its display is on:

| state | how it is produced | protocol clip (10 s) | mean delivered | witness |
|---|---|---|---|---|
| **armed** | screen on + an activity event (`.auto/measure.sh` does this automatically) | **1.22-1.25** | 2.0-2.4 GHz | `cpu7_deliv_mhz` >= 2000 |
| **unarmed** | screen on, no recent activity (what a bare wake leaves) | 1.67-1.70 | ~1.79 GHz | ~1790 |
| **screen off** | display off | 1.85 | ~1.28 GHz | ~1280 |

The arm is a burst (wake + three volume-key pairs, UI-indifferent) followed by a `KEYCODE_WAKEUP` stream
every 3 s that spans the run, stopped afterwards; `NO_ARM=1` disables it, `ARM_STREAM=0` keeps the burst
only, `ARM_PERIOD`/`ARM_SPARSE`/`ARM_DENSE_S` tune it. Each injected event costs ~50-60 ms of device CPU
(measured), which is why the stream is sparse and why the number is state-qualified rather than inflated.

**2. Witnesses, and a guard.** `measure.sh` emits the 2.4 GHz share (`cpu7_deliv2400_pct`), the MEAN
delivered frequency (`cpu7_deliv_mhz`, also a TSV column) and the share at >= 2.0 GHz. The 2.4 GHz share
alone is *not* sufficient: a run the governor steers to 2.15 GHz reads 0 % while being one of the fastest
cells of the day, which is how a whole class of "capped" readings appeared before Exp974. An ARMED run whose
mean falls below 2000 MHz prints a WARNING and must be retried, not averaged; audit check 19 fails if that
guard is ever removed from the harness.

**The witness is a step, not a line** (measured on 99 armed protocol rows, 1507-2383 MHz). Above 2000 MHz the
mean predicts almost nothing: -2.16 +/- 0.64 ms per 100 MHz (n=85, R2=0.12), and the mean rtf above the knee is
1.2335 +/- 0.0093. Below it the metric jumps +98 +/- 9 ms (+7.9 %, R2=0.556 - a better fit than the linear
model's 0.501 with fewer degrees of freedom). So the correct model is **"boosted or not", with the knee at the
guard's own threshold**, and a linear MHz coefficient fitted across the knee (-16.5 ms/100 MHz) is an artifact
of pooling both sides. Using that linear slope to "correct" measurements produces nonsense: Exp1024 did it to
rollback-ladder arms and got NEGATIVE hatch costs, which Exp1025 then reproduced with a paired ABAB - the hatch
really costs +3.97 % (cycles +3.69 % / +4.25 %) and the witness gap inside a pair explains only ~0.1 % of it.

**3. The metric drifts with device UPTIME inside one boot.** The supportable statement is a **slope: +0.21 ±
0.08 %/h** (t = 2.53, 35 clean armed sessions spanning 31.4–43.8 h, fitted at session level — `.auto/uptime_law.py`
recomputes it and is self-tested). Raw armed cells read 1.1939 at 12-20 h of uptime and 1.2242 at 26-40 h (same
arm era, delivered share flat), and the effect survives control for background process count, available memory and
battery temperature. **The two-parameter law formerly quoted here (1.1939 @ 16 h + 0.041/10 h) is rejected as a
predictor** — Exp1047 measured its bias on 35 clean sessions at **−3.0 %** (worst −5.6 %) — and there is **no
resolved curvature**, so neither a level correction nor a "drift saturates" story is available. A two-parameter
(uptime + arm) fit is not identifiable, and a 20-minute idle does not recover it. **Quote the uptime with
any number**, and read cross-epoch differences as "+2.5 to 4 % between epochs" rather than as code.

**4. The accuracy gate is armed too, and its WER column is the bit-stable part.** `.auto/eval40.sh` arms
each gate identically and records `gate_mean_mhz`. The 40-utterance WER has returned the identical S/D/I
profile for fifteen consecutive gates with zero discordant tokens of 731 against the frozen reference
(paired McNemar), while the gate's *mean rtf* varies by ~5 % across sessions - read the mean as a state
probe, the WER as the gate.

**5. Read speech is not corpus parity.** The 40-utt gate is read speech; a change can be output-identical
there and still move hard audio (measurement, Exp1010-1014: the v4.7 boundary batch is worth -7.1 pp on
overlapped speech, +0.64 pp on zh-TW, and is token-identical on consumer-mic English). Use
`.auto/hardaudio_watch.sh` (three watchdog sets, one command, references in its footer) for any change to
decode structure, boundary handling, quantization or the window protocol.

**6. Decode is a law, not a number.** `LATENCY_TRACE` per-window fits give
`ms/token = 83.5 + 11.4 us x KV-position` (two clips agreeing within 2 %, validated 2.3x out of range), so a
single ms/token figure is an average over a drifting quantity - quote the law or an interval.

**7. The machine-readable state.** `.auto/headline.json` is the single statement of the era, tier ladders
and device-state rules; `audit_harness.py` (137 checks, 22 proven by planted faults) fails if the prose in
this file, `RESULTS.md`, `README.md` or `.auto/prompt.md` disagrees with it. `.auto/device_state.tsv` holds
one row per run with the state witnesses, so any claim above can be re-derived from data.

## Phone evaluation (OPPO CPH2371, Dimensity 1300, 8 GB RAM, Android 13)

Cross-built with NDK r26d (`arm64-v8a`, `android-33`, `GGML_ARM_DOTPROD=ON`;
baseline kernels — the ggml dotprod probe can't run under cross-compilation).
Binaries + 2.5 GB GGUFs pushed to `/data/local/tmp/vibeasr`
(`libomp.so` from the NDK must sit beside them for `LD_LIBRARY_PATH=.`).

65-experiment optimization loop on-device (CPU-only, accuracy-guarded;
baseline 12.24 → best 6.52, −47%). Key finding: this SoC is **6× Cortex-A55
+ 2× Cortex-A78** — mask `F0` was 2 little + 2 big, and every `-t 4` run
straggled on little cores. The optimum is **`-t 2` on the two big cores**
(plain `-t 2` suffices; EAS places them correctly, no `taskset` needed to
ship). A78s sustain ~1.3 GHz under load (mobile sustained equilibrium, not
throttling); no i8mm/SVE exists, so NEON-F32 + DOTPROD is the full ISA story.

### Second wave: A78 codegen + loader (Exp457-473, 12.24 → 5.96/6.02)

Three findings re-opened the loop after it had converged at 6.52:

1. **Harness bug**: `measure.sh` pushed only `bin/asr_streaming`, never the
   `libggml.so`/`libllama.so` it links against (the binary contains no ggml
   code — `llvm-objdump` shows 0 sdot there, all 646 in `libggml.so`). Every
   *ggml-side* experiment had silently measured the old kernels, which
   invalidated the early `-mcpu=cortex-a78` and ThinLTO discards. Fixed with an
   md5-diff push (same guard added to `eval40.sh`).
2. **`-mcpu=cortex-a78` for the whole build** is worth **−6.5% (10 s), −7.7%
   (69 s)**, all of it VAE-side and A/B/A-bracketed. `GGML_ARM_DOTPROD=ON`
   already sets `-march=armv8.2-a+dotprod`, so this only adds the arch features
   + tuning the default `armv8-a` tune lacks: `sdot` 598→646, `fmla`
   1284→1542 in the shipped `libggml.so`. Tune-only (`-mtune`) captures 41% of
   it; portable in-tree defaults stay armv8.0-safe (device build only).
   The F16 path benefits most (−51% VAE): it was **instruction-bound** on
   software f16 handling, not bandwidth-bound. Consequence: the F16 VAE now
   matches Q8-mixed (43.0 s vs 43.9 s on the 10 s clip) despite 2× the weight
   bytes — **activation traffic dominates weight traffic** in the encoder.
3. **mmap VAE loader** (`src/vae.cpp`): the old loader value-initialised a
   fresh `std::vector<char>` per tensor (a full 1.4 GB zero-fill for F16) and
   copied twice (file→buf→tensor). mmap + one memcpy: startup load
   5.4 s → 2.5-4.1 s, and since Exp531 the copy itself runs on four workers
   (load_s 1.6 → 1.4 s on the shipped tier, transcript byte-identical). RTF
   excludes load by construction (RTF = VAE+LM), so this is wall-clock, not RTF,
   but it is free. The rest of the load is the LM side (llama.cpp, off-limits).

PGO was re-tested properly after the push fix (Exp471: protocol-trained,
instrumented train/use cycle) and gives **0%** — the codegen axis is saturated.

### Third wave: LM blocked-int8 kernels (Exp476-477, 6.01 → 5.13)

The LM used 29% of the time and was **not** compute-bound: this fork's
`ggml.c` dispatches to blocked `gemv`/`gemm` kernels only when the *weight
type* carries them, and `Q4_K_M`/plain `Q4_0` carry `vec_dot` only — so every
prefill re-streamed the 1.1 GB weight set **once per input row** (a 26-frame
window ≈ 29 GB of traffic). `GGML_TYPE_Q4_0_4_4` ships hand-written asm
gemv/gemm kernels (160 `sdot`/`udot`, guarded by `__aarch64__`+`__ARM_NEON`,
so they run **without** i8mm — unlike the 8x8 variant which needs
`__ARM_FEATURE_MATMUL_INT8` and is unusable on this SoC).

Re-quantising the LM to `Q4_0_4_4` (`llama-quantize --allow-requantize …
Q4_0_4_4`, 1.3 s on the host) gives, at **equal output length** (69 s clip,
443 vs 442 tokens):

* RTF 4.8179 vs 5.6089 (**−14.1%**), LM 76.6 s vs 130.6 s (**−41%**):
  prefill 29.2 vs 68.8 (−58%), decode 47.4 vs 61.7 (−23%).
* Control that isolates the mechanism: plain `Q4_0` (same 4.5 bpw, no blocked
  kernels) = LM 17.2 s, i.e. identical to `Q4_K_M` — the win is the kernel
  path, not the bit width.
* 40-utt on-device gate: **WER 5.10%** (S=31 D=2 I=4) vs 4.55% (S=27 D=1 I=4)
  for `Q4_K_M` = **+0.55 pp**; paired text diff 2.34%, 15/40 utts differ.
  The `Q4_0_4x4` quantizer uses symmetric absmax (`d = amax/8`) instead of
  plain `Q4_0`'s asymmetric min/max, which is where the extra error comes from
  (`ggml-aarch64.c`, 3rdparty).

**Follow-up (Exp494-495): almost all of that +0.55 pp was the embedding table,
not the transformer body.** llama-quantize's default demotes `token_embd.weight`
q6_K → q4_0; keeping it at the source precision with
`--token-embedding-type q6_K` (no new rounding at all) costs +60 MB and recovers
the accuracy:

| LM file | 40-utt WER | protocol RTF | zh-clip tokens |
|---|---|---|---|
| Q4_K_M (accuracy-first) | 4.55% | 6.00 | 45 |
| Q4_0_4x4, token_embd q4_0 | 5.10% | 5.13 | 37 |
| **Q4_0_4x4, token_embd q6_K** | **4.41%** | **5.21** | **42** |

Mechanism: the embedding rows for the control tokens (`[speech_start]`,
`[speech_end]`, `<|text_chunk_end|>`) are what the LM uses to decide chunk
boundaries, so their quantization error shifts end-of-chunk decisions. The
recipe for this artifact is therefore
`llama-quantize --allow-requantize --token-embedding-type q6_K \
 <q4_k_m> <out> Q4_0_4_4`.

An **imatrix** was also tested (Exp492) and is inert for this type:
`quantize_q4_0_4x4()` does `UNUSED(quant_weights)`.
* Caveat: on the 10 s zh protocol clip the fast LM emitted 37 tokens vs 45
  (content truncated) — the 10 s RTF ratio (5.13 vs 6.01) therefore flatters
  itself by generating fewer tokens; quote the 69 s equal-length number.

### Seventh wave: OpenMP off for the 1-thread chains (Exp535, 3.53 → 3.47)

The concurrent-encoder design runs each chain with **one** thread, but a
OpenMP-enabled ggml still entered a parallel region per op (860 ops × 208
launches per clip) for zero parallelism. Building ggml without OpenMP
(`-DGGML_OPENMP=OFF`, now part of `.auto/setup.sh`) removes that bookkeeping;
llama.cpp falls back to its own threadpool for the LM (unchanged at 8.7 s).

A/B/A in one session at equal temperature (34.6-34.9 °C): OMP-off 3.4885 /
3.4660 (VAE 26.0 s both) vs OMP-on 3.5070 (VAE 26.4 s) → **−1.1 % RTF, −1.5 %
VAE**. New shipped band **3.47-3.49**.

### Sixth wave: concurrent encoders (Exp521, 3.99 → 3.53)

The acoustic and semantic encoders are independent — separate weights, separate
streaming caches, separate compute arenas. Thread-scaling data (Exp519: the VAE
scales only 1.57× from 1→2 threads; Exp678 bounds that at ≤1.85× and measures the mode choice
itself at only ~5 % — concurrent 17.2 s vs sequential 18.1 s wall for the same two chains, so
this is a small, already-banked win rather than a throughput lever) showed the dependent chains do not saturate
two cores, so the two encoders now run **concurrently, one thread each**
(`vae_encode_parallel_cached`, a second `(ggml_context, arena)` pair; default
on, `VAE_SEQ_ENCODERS=1` opts back out). Opting out is now a **last resort**:
at fine granularity the per-encoder arenas are small, so concurrency costs only
~30 MB and is worth −11 % (Exp641–643, see the RESULTS.md ladder).

* VAE 31.4 → **26.7 s (−14.7 %)**, RTF 3.99 → **3.53 (−11.6 %)** on the 10 s
  protocol; 69 s sustained 3.83 → **3.43 (−10.4 %)**; LM unchanged.
* Cost: **+165 MB RSS** (2.07 GB) — exactly the second arena's *used* footprint
  (`ggml_used_mem` = 169.3 MB). The former `+512 MB` arena slack constant was
  dead weight and is now 64 MB; unused arena pages were never resident, so the
  sequential path's RSS is unchanged. The `ac`/`sem` phase timers now *overlap*
  and no longer sum to the VAE total.
* Parity is byte-level, verified four ways: desktop transcript, phone 10 s
  transcript, the first three LibriSpeech gate utterances vs the previous
  sequential gate files, and the 69 s sustained transcript — all identical.
  (Per-element work is split-invariant; the only reductions are per-row or
  per-4-row-group, so 1-thread and 2-thread results agree exactly.)

### Fifth wave: F16 im2col for the convs (Exp503-508, 4.24 → 3.99)

`ggml_conv_1d`/`ggml_conv_1d_dw` hardcode a **F32 im2col**: that doubles the
im2col write+read traffic and forces `mul_mat` to materialise a second full
F16 copy of its src1 (`vec_dot_type` is F16 for F16 weights). Our conv weights
are F16, so `src/vae.cpp` now builds the convs itself with
`ggml_im2col(..., GGML_TYPE_F16)` + `mul_mat` (the stock ops' structure copied
verbatim, only the im2col dtype changed):

* VAE 33.8 → **31.4 s (−7.5%)**, RTF 4.24 → **3.99 (−6.3%)**, RSS 2069 MB.
* Not bit-identical (the F32→F16 rounding happens in the copy instead of in
  `mul_mat`), so the 40-utt gate was re-run: **WER 4.41% (S=28 D=1 I=3, H=697),
  identical error counts** to the pre-change gate. Mean 40-utt RTF 4.47.
* All four clips agree: 17 s 3.50 (−6.7%), 69 s 3.83 (−5.9%), slice-B 4.15
  (−5.9%), 40-utt mean 4.47 (−5.7%), every token count identical.
* Other tiers gain too (their convs are F16): balanced 5.14 → 4.91,
  accuracy-first 6.01 → 5.80, lean (p26) 4.39 → **4.00 @ 1.91 GB**.

**Gotcha for anyone repeating this:** do *not* use `ggml_im2col_asym` — its op
(`GGML_OP_IM2COL_ASYM`) is hardwired to `ggml_compute_forward_im2col_i8_s`, so
an F16 `dst_type` silently writes I8_S data into an F16 tensor (caught on the
x86 desktop build: output collapsed to 7 tokens). Use `ggml_im2col`.

### Fourth wave: VAE blocked-int8 kernels (Exp480-483, 6.01 → 4.26)

The same trick applies to the VAE, and the VAE is the bigger prize (71% of
time). Its 104 `ffn.linear` weight tensors are ~82% of the VAE file bytes and
their matmuls also went through the `vec_dot` path (weights re-read per
activation row). `utils/convert_vae_to_gguf.py` gained a `q4_0_4x4_ffn`
outtype: the quantizer is ggml's own per-row Q4_0 (`quantize_row_q4_0_ref`),
only the **byte packing** is the interleaved `block_q4_0x4` layout
(`make_block_q4_0x4`, xor 0x88) ported to numpy (`quantize_q4_0_4x4()`);
everything else stays F16.

* VAE 42.7 s → **34.0 s (−20.4%)**, RTF 6.00 → 5.14, RSS 2.16 GB (from 2.99),
  load 1.6 s. Versus the earlier plain-`Q4_0` VAE tier (VAE 47.4 s) this is
  −28%: again the kernel path, not the bit width.
* 40-utt on-device gate: **WER 4.82%** (S=29 D=2 I=4) vs 4.55% (S=27 D=2 I=4)
  = **+0.27 pp** — the best accuracy-per-speed trade of the loop.
* Combined with the fast LM (`Q4_0_4x4` on both) → the max-speed tier:
  10 s **4.26**, 69 s equal-token **4.05** (−27.9% vs accuracy-first), 40-utt
  mean **4.77**, WER 5.10% (S=31 D=3 I=3), RSS **2.05 GB**, 69 s transcript
  diff vs accuracy-first 3.93%.

**Current best: 1.25 (era v4.8, ARMED measurement state = burst + wake-hint stream, mean delivered
~2045 MHz)** — the historical 10 s cell 1.19 was the SAME byte-identical binary at mean ~2377 MHz, and
the unarmed device reads 1.67-1.70 (screen on) or 1.85 (screen off), so quote the state AND its
`cpu7_deliv_mhz` witness with any number (Exp968-980). The ladder in
this section is history only; the live ladder is in
RESULTS.md and the machine-readable statement is `.auto/headline.json`, which `audit_harness.py`
check 9 enforces against every current-state claim in these docs.

### Final tier ladder (Exp525–542-era snapshot — SUPERSEDED)

> **Superseded in v4.0** (whole-window pieces: PIECES=1 is −1.2 % with ZERO gate discordants, Exp706): the shipped
default is PIECES=1 at RTF **~2.43** (v3.6 fused f32 depthwise-conv kernel −3.5 %, v3.7 layer-scale
epilogue −1.1 %, v3.8 gelu+bias −0.8 %, **v3.9 blocked-int8 conv weights −3.0 % with the gate paired
equivalent at McNemar p=1.0 - Exp690**)
> (69 s 2.41, 138 s 2.48, RSS 2.46 GB, gate 4.51 % with 0/731 tokens differing from the p2 gate); the current ladder
> lives in RESULTS.md. The table below (all on-device gated, 10 s protocol
> `-t 2`/C0, 26-piece default) is preserved as history.

| tier | files | 10 s | 17 s | 40-utt mean | 69 s (equal tokens) | WER (40-utt) | RSS |
|---|---|---|---|---|---|---|---|
| **max-speed (shipped default as of Exp542): 4x4 VAE + 4x4 LM (q6_K emb) + F16 im2col + concurrent encoders + OMP off, 26 pieces** | 2.0 GB | **3.48** (3.47-3.50, n=7, sd 0.010) | **3.05** | **3.86** | **3.38** | **4.41%**\*\*\* | 2.07 GB |
| _max-speed, RAM-lean (`VAE_SEQ_ENCODERS=1`)_ | 2.0 GB | _4.01_ | — | — | — | 4.41% | **1.91 GB** |
| balanced: VAE Q4_0_4x4-FFN + LM Q4_K_M (final build) | 1.9 GB | **4.40** | — | — | — | 4.82% | 2.07 GB |
| fast-LM: VAE F16 + LM Q4_0_4x4 (q6_K emb) _(pre-im2col)_ | 2.5 GB | 5.21 | — | 5.73 | — | 4.41% | 2.94 GB |
| balanced-lean: VAE Q4_0_4x4-FFN + 26 pieces (Q4_K_M LM) _(pre-im2col)_ | 1.9 GB | 5.21 | — | — | 4.94 | — | 1.98 GB |
| _max-speed @13 pieces, pre-concurrency (history)_ | 2.0 GB | _3.99_ | — | — | _3.83_ | 4.41% | 2.07 GB |
| accuracy-first: VAE F16 + LM Q4_K_M (final build) | 2.5 GB | **5.34** | — | — | 5.62 | 4.55% | 2.95 GB |
| ultra-lean (plain Q4-FFN + 26 pieces, history) | 1.6 GB | 6.61 | — | — | — | 5.23% | 1.97 GB |
| Q8-mixed VAE (history) | 1.9 GB | 6.14 | — | 6.71 | 5.76 | 4.55% | 2.44 GB |
| _pre-A78 F16 (history)_ | 2.5 GB | _10.5_ | — | — | _9.73_ | 4.13% (desktop) | 2.99 GB |

\*\* Gate re-run at the shipped 26-piece default (Exp514): WER 4.41%
(S=28 D=1 I=3, H=697) with **byte-identical transcripts** to both the 13-piece
gate and the pre-F16-im2col gate - every protocol and kernel change since then
is numerically inert on this gate.
\*\*\* Full 40-utt gates re-run twice more - after the concurrent-encoder change
(Exp525) and after GGML_OPENMP=OFF (Exp537): WER 4.41% each time with the same
error counts and **all 40 transcripts byte-identical** across all three gates -
numerical identity proven over the whole gate set for every threading change.
Mean 40-utt phone RTF: 4.4745 (sequential) -> 3.9113 (concurrent) -> **3.8597**
(+OMP off).

Note (Exp542): with concurrent encoders every tier carries the same two-arena
footprint, so the balanced tier no longer has a RAM advantage over the shipped
tier (both ~2.07 GB); its niche is now purely the clean zh transcript (the 4x4
LM garbles rare/proper tokens, Exp499). RAM-constrained devices use the
RAM-lean tier instead (`--vae-pieces 13 VAE_DEFER_LATE=1`, encoders still
concurrent): 2.54 on the protocol clip at 1.75 GB (Exp708; defer is REQUIRED at
fine granularity - without it p13 pays 2.67 in the deep-layer GEMV regime). The
old default (PIECES=13, no flags) predates the v4.0 defer rework and now runs
piece-wise by default, so the flag must be set explicitly.

With the corrected embedding the **max-speed tier Pareto-dominates the
accuracy-first tier**: 28% faster on the 40-utt mean and 27.5% on the 69 s clip,
equal-or-better WER (4.41 vs 4.55, one substitution apart), and 0.9 GB less RAM.
The 69 s token count is 438 vs 442 (-0.9%), i.e. the old short-clip truncation
(37/45) is mostly gone. Remaining zh-clip caveat, measured on both short clips:
the 10 s protocol clip ends 3 tokens early (42/45) and the 17 s clip is
content-complete but garbles the brand name ("YyY … YSR" vs "Y-voice … Y-voice
ASR") — i.e. the 4-bit body's residual cost is rare/proper-token fidelity, not
missing content. A controlled test (Exp499) localises it: a plain-`Q4_0`
requant of the same source - same tensor demotions, but the `vec_dot` path -
transcribes "Y-voice" cleanly, so the garble comes from the blocked-int8
kernel/layout path itself, not from which tensors are 4-bit. Use the balanced
tier (Q4_K_M LM, clean zh transcripts) when proper nouns matter; the 69 s
English WER story is unaffected (4.41%).

\* The `Q4_0_4x4` LM shifts greedy end-of-chunk decisions on some short clips,
so its 10 s / 17 s token counts are below baseline (40/45 and 64/68) and those
RTF ratios are partly fewer-decode-tokens effects; the 69 s clip is the only
one where its output length matches (444 vs 442-443), so **4.05 (−28%) is the
headline speed claim** for the max-speed tier. The balanced tier (4x4 VAE +
Q4_K_M LM) does not truncate (46/45, 69/68) and is the best all-round tier:
−14% RTF for +0.27 pp WER.

The **balanced-lean** variant (`--vae-pieces 26` on the same files) traded
+1% (10 s) / +2.4% (69 s) RTF for −180 MB RSS at *identical* transcripts in the
Exp542-era concurrent regime. The current RAM-lean tier is **p13, concurrent
encoders, defer explicitly ON** (`--vae-pieces 13` + `VAE_DEFER_LATE=1` — required
since v4.1, whose default is defer-OFF): 2.54 on the protocol clip (n=3, warm-session read;
same-session m2-OFF control 2.57), **2.47 on 138 s** and **2.70 on the 40-utt mean** (both re-measured
at v4.4, Exp786/789), at 1.75 GB and WER 4.79 % (2 tokens of 731 from the default tier, p = 0.5) — Exp643/708/711/717/767/786/789.

That also corrected the earlier "finer splits lose" conclusion, which held only
on the time axis: the deferred late pass captures the deep-layer batching
benefit window-wide at ANY fine piece count, so per-piece overhead is all that
is left — and the RAM driver is the *early-stage activation arena*, which scales
with piece size. The measured map over the runnable set {1, 2, 13, 26}
(non-divisors need a window-loop rework, Exp605), re-swept at v4.2 (conv-int8 + m2) with 3-4
interleaved reps per arm (Exp772), is **defer-OFF**: p1 **2.378** @ 2.37 GB (default,
time-optimal), p2 2.454 @ 2.07 (+3.2 %, fallback), p26 2.707 @ 1.71 (+13.8 %); **defer-ON**:
p13 2.531 @ 1.75 (the RAM-lean tier), p26 2.554 @ 1.72 — so p1 is the speed default while p13 is
the RAM sweet spot. **Those absolutes predate v4.5** (Exp821 moved p1 to 2.18 and its RSS to 2.19 GB), and
the shift is *asymmetric by construction*: the in-kernel conv left pad only fires when a window is a single
piece with no deferred late pass, so p1 gained ~3 % while p2/p13/p26 did not — which widens the p1-vs-p2 gap
from ~3 % to ~6 % and re-arms this axis under its own documented trigger (a per-piece cost-structure change).
Re-sweep only if a p1-vs-p2 RAM decision ever matters; the ratios among the fine counts (p13/p26) are
unaffected because none of them can take the fast path. `VAE_LATE_SPLIT=5` on this tier is **parity, not a win** (3 interleaved reps 2.425 vs 2.424 at
+19 MB, byte-identical output, Exp794): the −0.9 % recorded for it in Exp565 predates conv-int8, and
at p13 a piece is already 8 frames, so the deferred stage sat in the GEMM regime before the boundary
moved — there is nothing left for it to batch. split=6 stays the default and no third rung exists
between the shipping tiers. The p1-vs-p2 gap is ~3 % on the shipping config; a same-session 2×2 (Exp773)
puts it at +3.0 % with int8 convs and +1.9 % with F16 convs, so roughly 1 pp of it is the int8 conv
path's per-piece cost (paid twice per window when pieces are finer), and the blocked-tail kernel is
NOT a factor (`GGML_MM_M2_OFF=1` costs +0.7 % at p1 and +0.8 % at p2 — equal, so it is the LM's
prefill that benefits, not the VAE's piece shape). The v3.9 map's −1.2 % datum came from an
aborted/partly-confounded interleave, so treat the apparent "widening" as mostly a correction of
that number rather than a regression. Mechanism for the remaining ~2 %: finer pieces re-stream stage
weights and re-run graph build/launches per piece, i.e. twice per window at p2. Re-swept at v3.9 and
v4.2 only — do not re-sweep without another cost-structure change. Output relations changed with the fusions:
p1/p2 are byte-identical everywhere measured (protocol transcript hash-equal in 6+ reps;
40/40 gate transcripts identical, b=0/c=0); p13 differs from p2 by 2 gate tokens (p=0.5). The old
"p2 differs (40 vs 39 tokens)" sentence was the taps-vs-im2col era (Exp640) and no longer holds.
Deferred-vs-piece-wise has a granularity boundary inside the coarse regime too (Exp746):
at p1 defer-OFF wins −1.2 % (batching nothing, pure staging cost), at p2 it ties
(2.4621 vs 2.46, both eras - Exp639's verdict is the one pre-int8 closure that
survived), at p13 defer-ON wins −5 % (deep layers GEMV-shaped per piece). Rule of
thumb: defer pays iff pieces are GEMV-shaped; at GEMM-shaped pieces it ties (p2)
or costs (p1).

Against the original baseline (12.24) the max-speed tier is **−65%**; against
the pre-A78 loop best (6.52) it is −35%. The balanced tier dominates the fast
tier (same 10 s RTF, better WER, 0.7 GB less RAM) and the old ultra-lean tier
(much faster, better WER, +0.19 GB).

40-utt gates above are **on-device** (`.auto/eval40.sh` + `.auto/score_hyp.py`,
hyp sets in `eval-librispeech/hyp-{a78,f16a78}/`) because a codegen gate must run
on the target ISA; the desktop numbers (4.13%/4.41%) remain as history, with a
cross-arch offset of the same class (<1 pp). F16 vs Q8 on-device differ by one
substitution (4.55% both) while against the PyTorch reference F16 is closer
(2.06% vs 2.61%).

Correctness and flat memory hold on-device at all lengths (cross-device WER
< 1% same-config). **Determinism (Exp518):** two consecutive runs of the shipped config produce
byte-identical transcripts, matching a reference saved several builds earlier
(greedy decoding + fixed kernels => reproducible outputs).

**Gate-transfer verification (Exp520):** the F16-VAE tiers' 40-utt gate was
measured before the F16-im2col change. Re-running the accuracy-first tier on
the 69 s clip reproduced the pre-change transcript byte-for-byte (442 tokens),
so that gate transfers unchanged; together with the shipped tier's 40/40
byte-identical gate transcripts, every tier's accuracy number is valid for the
current kernels without a fresh 40-minute gate per tier.

**Sustained long-form check (Exp515 pre-, Exp526 post-concurrency, Exp538 at
the frozen build):** a 138 s clip (the 69 s chat concatenated with itself) runs
at RTF **3.45** (3.90 sequential -> 3.52 concurrent -> 3.45 with OMP off) with RSS flat at 2.09 GB (+165 MB for the concurrent encoders'
second arena, no growth over 8 minutes, majflt 0) and the two halves of the
transcript matching at 4.25% WER - identical to the pre-concurrency run, i.e.
no thermal cliff, no memory growth, no context drift, and no threading
instability over 36 chunks / 72 parallel encoder launches. `--xwin` (cross-window VAE carry) is now ~6 % SLOWER
on the protocol clip (re-measured at v4.1, Exp714: 2.55 vs 2.39 interleaved - it was -3.2 % pre-defer-flip;
reference now 2.38 at v4.2, sign unchanged;
the carry's per-chunk splice cost scales with piece size, and p1 doubles it). It remains a useful DIAGNOSTIC
(it moves diarization attribution on the multi-speaker probe) and a server-path option, not a speed lever.
It drifts (+11.5 % WER) on 69 s as well, so auto-selection by length was declined twice - a product
decision, not a loop protocol. The VAE runs at ~50% of DRAM roofline (Exp75: 72% of time in GEMM
kernels, fusion ceiling ≈1.2×); remaining kernel upside needs fused NEON
intrinsics, i.e. a 3rdparty change. **RTF < 1 on this phone class requires
retraining** (QAT INT8 VAE and/or a smaller encoder+LM), not more porting.

## Deeper quantization (measured)

40-utt LibriSpeech `test-clean` subset, same normalization (LM candidates are
requants from Q4_K_M, i.e. slightly pessimistic; VAE-Q8 also checked on the 69 s
multi-window clip to exercise cache carries):

| LM \ VAE | F16 (1.4 GB) | Q8-mixed (0.8 GB) | I8_S (0.67 GB) |
|---|---|---|---|
| Q4_K_M (1.1 GB) | **4.13%** ✅ shipped | **4.41%** ✅ (+0.3pp) | collapsed (loops) |
| Q4_0 (1.0 GB) | 5.6% on 5-file screen | — | — |
| Q3_K_M (0.9 GB) | 6.20% (+2.1pp) | — | — |
| Q2_K (0.7 GB) | 7.58% (+3.5pp) | — | — |

Recommended max-quant combo: **VAE Q8-mixed + LM Q4_K_M** (files 1.9 GB,
desktop RSS 2.72 GB, RTF ~1.1; phone: RTF 6.5/5.7/6.3 on 10/17/69 s,
RSS 2.44 GB flat, `-t 2`). 69 s WER 3.67%, identical parity class.
`--outtype q8_0_mixed` in `convert_vae_to_gguf.py` quantizes large weights
(last dim % 32 == 0) to Q8_0, keeps conv kernels/bias/norms in F16/F32
(Q8_0 blocks need 32-wide rows; depthwise kernels can't quantize).
Below Q4 the LM falls off a cliff (+2pp at Q3, +3.5pp at Q2) — rejected.

## 9. Final-window flush — v4.6 protocol change (Exp829/Exp830)

The fixed-window protocol always zero-pads the **last** window to 26 frames, so the encoder spends most of
that window's work on silence: `vae_s = 0.21 + 3.45 x windows` with `windows = ceil(len/70400)`, i.e. a 10 s
clip encodes 332,800 samples for 240,000 of audio (1.387x). The last window now encodes only the real frames
(rounded up to a 2-frame piece) and the LM is fed the frames that exist instead of 26.

Measured (paired, interleaved reps):

| length | padded | flushed | delta | padded fraction of last window |
|---|---|---|---|---|
| 10 s protocol | 2.1866 | **1.9395** | −11.8 % | 0.653 |
| 17 s | 2.1760 | 2.0890 | −4.0 % | 0.193 |
| 69 s | 2.2091 | 2.1759 | −1.5 % | 0.080 |
| 138 s | 2.2542 | ~2.25 | ~0 % | 0.003 |
| 40-utt gate mean | 2.4456 | **1.9811** | −19.0 % | ~0.65 per clip |

**This is a protocol change, so it was gated, not just benchmarked.** 40-utterance gate: hybrid WER
4.38 % vs 4.51 % (jiwer-path scorer 4.68 % vs 4.82 %), paired token test **0 discordant of 731**
(McNemar p = 1.0), and one FEWER insertion. The direction is the point: the padded tail frames had been
generating tokens from silence (17 s 108 → 106 tokens, 138 s 877 → 876), so removing them removed an
insertion source rather than trading accuracy for speed. The gate improved MORE than the protocol clip,
which is the opposite of an overfit signature.

Hatch: `FLUSH_TAIL_OFF=1` restores the fixed-26-frame tail and reproduces the pre-Exp829 protocol exactly
(2.1866, transcript `55ac39b635cb`), so the old benchmark remains reachable.

The RAM-lean tier (`--vae-pieces 13` + `VAE_DEFER_LATE=1`) now flushes too (Exp830): its boundary buffers
are allocated at full-window spacing and packed down to the frames actually present, so a short final window
is self-consistent. Lean protocol cell 2.40 → **2.12** (−11.5 %) at ~1.75 GB, and its transcript is now
byte-identical to the shipped tier's — the previously documented 1-token lean/shipped difference was the
padded tail interacting with the deferred late path, not a real divergence.

## Guard-coverage audit: 11/13 checks fired on their own fault; one was silent (Exp873)

`.auto/audit_selftest.py` plants one fault per audit check and requires THAT check to fail. Until now the
Exp660 rule ("a self-check is untrustworthy until a planted fault makes it fail") had been applied only
check-by-check, as each check was written - never as a property of the audit as a whole. 13 plantable
classes, 3 accepted as uncontrolled and named as such (device binary hashes need a real rebuild; a
co-runner would poison the timings it is meant to detect; check 12 has its own --dry control).

**One guard was silent, and it was the connectivity guard**: the device-state test was
`'device' in r.stdout`, but adb's failure text is `error: device 'X' not found` - which contains 'device'.
An unreachable or mistyped serial therefore PASSED, and every device-side section below it then reported
missing files, which reads like a broken phone rather than a broken serial. Success is now the literal line
`device` and nothing else. Cost of the bug in practice: zero wrong measurements so far (the file checks
would have failed loudly), but it is exactly the class that produces a plausible number from the wrong
system, which this loop has been burned by repeatedly.

Two further gaps were found while designing the plants, both fixed: check 8 PRINTED `gate refs.json:
MISSING` without failing (so deleting the gate reference set changed no verdict), and section 7 iterated
the manifest, so device clips nobody declared were invisible - the Exp675 blind spot. That new check found
16 undeclared clips on the first run: fixtures a tool regenerates, phase probes held for reuse, and three
aligned holdout sets that were real assets and are now blessed. The device asset surface is now fully
declared (36 manifest entries + 7 accounted-for fixture/probe patterns + gate utterances by name).

Method notes kept because they generalize:
* A plant can be INVALID rather than a check SILENT - my dirty-tree plant modified `.auto/config.json`, and
  section 3b deliberately ignores `.auto/` (the autoresearch log writes there). Distinguish the two before
  "fixing" a working check: it fired once aimed at a tracked file outside `.auto`.
* A fault catalog must be excluded from the path-existence check: `audit_selftest.py` legitimately names
  files that must NOT exist, and an over-claiming guard gets muted (Exp660).

## An unrun test is not a test (Exp874)

`score_mixed.py --selftest` has been **failing since the commit that introduced it** (869e73e). The fixture
comment says "1 char inserted" but the string duplicated two characters (`你好世界` → `你好世界世界`), so the
scorer correctly returned 2/4 = 0.50 against an expectation of 0.25. The scorer was right, the fixture
contradicted its own comment - and because no harness step ever ran `--selftest`, a red test sat there
across hundreds of measurements. No published number is affected (it is a test fixture, not a scorer bug),
but the lesson generalizes: Exp660's "prove the guard can fail" applies to TESTS as well as CHECKS, and a
test that nothing invokes is worse than no test because it reads as coverage.

Fixes: audit check 14 now runs `compare_arms / rss_soak / score_mixed / score_stream --selftest` on every
session start (≤0.5 s total) and WARNs on any tracked tool that has a `--selftest` mode nobody calls. Fault
14 in `audit_selftest.py` proves the check fires by breaking a scorer's own expectation.

Two harness defects surfaced while wiring it:
* Section 12 rebound the name `sh` - the subprocess helper - to a PATH. Any section below it that calls
  `sh()` would die. It bit check 14 within one run. Renamed to `sweep`.
* `audit_selftest.py` reverted its plants with `git checkout --`, which restores the **committed** state and
  silently destroys uncommitted work in the same file (it ate this round's fixture fix within minutes).
  Plants now snapshot the file's bytes and write them back - a plant must restore the pre-plant state, not
  the last commit.

## Long-clip provenance: the ladder's long rungs are ONE 69-second recording (Exp875)

The manifest used to *note* what the long clips were. Proving it on bytes (audit check 7e, hashed on the
device via ranged `tail`/`head`, no host copies) gave:

| clip | proven composition | unique audio |
|---|---|---|
| `chat69.wav` | 68.987 s, `data` at offset 224 (ffmpeg writes LIST/INFO first) | 100 % |
| `chat138.wav` | `chat69` ⊕ `chat69` — **the same 69 s twice, byte for byte** | **50 %** |
| `chat155.wav` | `chat138` ⊕ `chat17` (join at sample 3,311,352 → phase 2,552 mod 70,400) | ~57 % |
| `chat17.wav` | the first 408,000 samples of `chat69` | — |

Consequences, in the order they matter:
1. **The ladder's length steps are content-free.** 10 → 17 → 69 → 138 s compares the *same conversation* at
   increasing length, so a 69→138 s change cannot be blamed on content. That is a property worth having, and
   it explains how flat the long end of the ladder is.
2. **Long-form CONTENT statements need the caveat.** Anything about "what the model does in long audio"
   based on `chat138`/`chat155` sees 69 s of unique signal repeated. Exp644's conclusion that chat138
   "genuinely contains refrains" is now half-explained by construction (a repeat guarantees duplicated
   transcript n-grams), which matters because that number was the baseline for repetition/loop detectors.
3. Rate, RSS, KV-position and convertibility results are unaffected — they are per-window or per-position.

Method note: the first comparison used a fixed 44-byte payload offset and reported "both parts differ" for
byte-identical clips, because ffmpeg-written files put `data` at 224 while others put it at 44. The check
parses the RIFF chunk chain per file (`wav_data_range`) and asserts payload-length arithmetic before hashing.
