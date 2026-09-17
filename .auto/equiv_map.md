# output-equivalence map vs hyp-bound835
generated 2026-09-17 by .auto/equiv_map.py - paired token distance, tool tokeniser
reference: 729 tokens / 40 utterances; stamp: `VAE_FILE=vae-encoder-convint8.gguf LM_FILE=lm-q8head.gguf THREADS=2 MASK=C0 PIECES=1 EXTRA_ENV= BIN=541d8b9349bb8525409dc25618521b64`

**1 of 72 archived sets are byte-identical to the reference:** hyp-gate845

Everything else differs by >=1 token. These are RAW TOKEN DIFFERENCES (presence), not McNemar
correctness discordants - for accuracy claims use `compare_arms.py --gate A B refs.json`. Per
Exp657's resolution limit, 4-8 differing tokens is two to four words out of 731 and is NOT
evidence of an accuracy difference: read this table as 'same system?', not 'which is better'.
Distances are relative to THIS reference only; re-run with a different REF_DIR to re-anchor.

| set | tokens differ | only in set | only in ref | provenance |
|---|---|---|---|---|
| hyp-head4x4 | 6 | 3 | 3 | unstamped (pre-Exp617) |
| hyp-alloc2 | 7 | 4 | 3 | unstamped (pre-Exp617) |
| hyp-allocp26 | 7 | 4 | 3 | unstamped (pre-Exp617) |
| hyp-defer5 | 7 | 4 | 3 | unstamped (pre-Exp617) |
| hyp-dwtaps | 7 | 4 | 3 | unstamped (pre-Exp617) |
| hyp-latealloc | 7 | 4 | 3 | unstamped (pre-Exp617) |
| hyp-max4x4 | 7 | 4 | 3 | unstamped (pre-Exp617) |
| hyp-zerocopy | 7 | 4 | 3 | unstamped (pre-Exp617) |
| hyp-defer | 8 | 5 | 3 | unstamped (pre-Exp617) |
| hyp-f16im | 8 | 5 | 3 | unstamped (pre-Exp617) |
| hyp-finalp26 | 8 | 5 | 3 | unstamped (pre-Exp617) |
| hyp-max4x4e | 8 | 5 | 3 | unstamped (pre-Exp617) |
| hyp-omp_off | 8 | 5 | 3 | unstamped (pre-Exp617) |
| hyp-shipped | 8 | 5 | 3 | unstamped (pre-Exp617) |
| hyp-q5head | 9 | 5 | 4 | unstamped (pre-Exp617) |
| hyp-axpyneon64 | 10 | 6 | 4 | stamped |
| hyp-ctblock | 10 | 6 | 4 | unstamped (pre-Exp617) |
| hyp-dw670 | 10 | 6 | 4 | stamped |
| hyp-dwchk68 | 10 | 6 | 4 | stamped |
| hyp-dwfix69 | 10 | 6 | 4 | stamped |
| hyp-gate613 | 10 | 6 | 4 | unstamped (pre-Exp617) |
| hyp-gate645 | 10 | 6 | 4 | stamped |
| hyp-gelb673 | 10 | 6 | 4 | stamped |
| hyp-leanflusht | 10 | 5 | 5 | stamped |
| hyp-leanseq | 10 | 6 | 4 | unstamped (pre-Exp617) |
| hyp-lsfuse671 | 10 | 6 | 4 | stamped |
| hyp-q8head | 10 | 6 | 4 | unstamped (pre-Exp617) |
| hyp-axpy64g | 11 | 6 | 5 | stamped |
| hyp-flusht829 | 11 | 6 | 5 | stamped |
| hyp-lean694 | 11 | 6 | 5 | stamped |
| hyp-lean711 | 11 | 6 | 5 | stamped |
| hyp-lean711b | 11 | 6 | 5 | stamped |
| hyp-lean734b | 11 | 6 | 5 | stamped |
| hyp-leannorm785 | 11 | 6 | 5 | stamped |
| hyp-vae4x4 | 11 | 6 | 5 | unstamped (pre-Exp617) |
| hyp-convint8b | 12 | 7 | 5 | stamped |
| hyp-gate721 | 12 | 7 | 5 | stamped |
| hyp-gate778 | 12 | 7 | 5 | stamped |
| hyp-gate791 | 12 | 7 | 5 | stamped |
| hyp-gatedef1 | 12 | 7 | 5 | stamped |
| hyp-gatem2 | 12 | 7 | 5 | stamped |
| hyp-gatep1 | 12 | 7 | 5 | stamped |
| hyp-gelufuse781 | 12 | 7 | 5 | stamped |
| hyp-lean734-nodefer | 12 | 7 | 5 | stamped |
| hyp-lm4x4e | 12 | 7 | 5 | unstamped (pre-Exp617) |
| hyp-lpad821 | 12 | 7 | 5 | stamped |
| hyp-norm784 | 12 | 7 | 5 | stamped |
| hyp-gate694p2 | 13 | 7 | 6 | stamped |
| hyp-geluquick | 14 | 8 | 6 | unstamped (pre-Exp617) |
| hyp-leanp13c | 14 | 8 | 6 | stamped |
| hyp-leanp13t | 14 | 8 | 6 | stamped |
| hyp-leanp13u | 14 | 8 | 6 | stamped |
| hyp-leanp26c | 14 | 8 | 6 | stamped |
| hyp-a78 | 16 | 9 | 7 | unstamped (pre-Exp617) |
| hyp-convint8 | 16 | 9 | 7 | stamped |
| hyp-gate694p13 | 16 | 9 | 7 | stamped |
| hyp-leanv34 | 16 | 9 | 7 | unstamped (pre-Exp617) |
| hyp-lm4x4_40 | 16 | 9 | 7 | unstamped (pre-Exp617) |
| hyp-lm4x4im | 16 | 9 | 7 | unstamped (pre-Exp617) |
| hyp-vaeq4 | 16 | 8 | 8 | unstamped (pre-Exp617) |
| hyp-f16a78 | 19 | 10 | 9 | unstamped (pre-Exp617) |
| hyp-q8leg | 19 | 11 | 8 | unstamped (pre-Exp617) |
| hyp-t2 | 19 | 11 | 8 | unstamped (pre-Exp617) |
| hyp-vaeq8 | 19 | 11 | 8 | unstamped (pre-Exp617) |
| hyp-xwin | 23 | 11 | 12 | unstamped (pre-Exp617) |
| hyp-q8xwin | 27 | 13 | 14 | unstamped (pre-Exp617) |
| hyp-q3km | 33 | 15 | 18 | unstamped (pre-Exp617) |
| hyp-q2k | 71 | 35 | 36 | unstamped (pre-Exp617) |
| hyp-dedup833 | 155 | 34 | 121 | stamped |
| hyp-cpp | 787 | 781 | 6 | unstamped (pre-Exp617) |
| hyp-aco | 1285 | 634 | 651 | unstamped (pre-Exp617) |
