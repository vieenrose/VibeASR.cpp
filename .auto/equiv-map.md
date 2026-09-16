# output-equivalence map vs hyp-gate791
generated 2026-09-16 by .auto/equiv_map.py - paired token distance, tool tokeniser
reference: 731 tokens / 40 utterances; stamp: `VAE_FILE=vae-encoder-convint8.gguf LM_FILE=lm-q8head.gguf THREADS=2 MASK=C0 PIECES=1 EXTRA_ENV= BIN=2a748d3c7f03fa51056b712bc33cd31b`

**9 of 66 archived sets are byte-identical to the reference:** hyp-convint8b, hyp-gate721, hyp-gate778, hyp-gatedef1, hyp-gatem2, hyp-gatep1, hyp-gelufuse781, hyp-lean734-nodefer, hyp-norm784

Everything else differs by >=1 token. These are RAW TOKEN DIFFERENCES (presence), not McNemar
correctness discordants - for accuracy claims use `compare_arms.py --gate A B refs.json`. Per
Exp657's resolution limit, 4-8 differing tokens is two to four words out of 731 and is NOT
evidence of an accuracy difference: read this table as 'same system?', not 'which is better'.
Distances are relative to THIS reference only; re-run with a different REF_DIR to re-anchor.

| set | tokens differ | only in set | only in ref | provenance |
|---|---|---|---|---|
| hyp-axpyneon64 | 4 | 2 | 2 | stamped |
| hyp-ctblock | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-defer | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-dw670 | 4 | 2 | 2 | stamped |
| hyp-dwchk68 | 4 | 2 | 2 | stamped |
| hyp-dwfix69 | 4 | 2 | 2 | stamped |
| hyp-f16im | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-finalp26 | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-gate613 | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-gate645 | 4 | 2 | 2 | stamped |
| hyp-gelb673 | 4 | 2 | 2 | stamped |
| hyp-leanv34 | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-lsfuse671 | 4 | 2 | 2 | stamped |
| hyp-max4x4e | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-omp_off | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-q8head | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-shipped | 4 | 2 | 2 | unstamped (pre-Exp617) |
| hyp-axpy64g | 5 | 2 | 3 | stamped |
| hyp-lean694 | 5 | 2 | 3 | stamped |
| hyp-lean711 | 5 | 2 | 3 | stamped |
| hyp-lean711b | 5 | 2 | 3 | stamped |
| hyp-lean734b | 5 | 2 | 3 | stamped |
| hyp-leannorm785 | 5 | 2 | 3 | stamped |
| hyp-leanseq | 6 | 3 | 3 | unstamped (pre-Exp617) |
| hyp-alloc2 | 7 | 3 | 4 | unstamped (pre-Exp617) |
| hyp-allocp26 | 7 | 3 | 4 | unstamped (pre-Exp617) |
| hyp-defer5 | 7 | 3 | 4 | unstamped (pre-Exp617) |
| hyp-dwtaps | 7 | 3 | 4 | unstamped (pre-Exp617) |
| hyp-latealloc | 7 | 3 | 4 | unstamped (pre-Exp617) |
| hyp-zerocopy | 7 | 3 | 4 | unstamped (pre-Exp617) |
| hyp-leanp13c | 8 | 4 | 4 | stamped |
| hyp-leanp13t | 8 | 4 | 4 | stamped |
| hyp-leanp13u | 8 | 4 | 4 | stamped |
| hyp-leanp26c | 8 | 4 | 4 | stamped |
| hyp-lm4x4e | 8 | 4 | 4 | unstamped (pre-Exp617) |
| hyp-max4x4 | 9 | 4 | 5 | unstamped (pre-Exp617) |
| hyp-q5head | 9 | 4 | 5 | unstamped (pre-Exp617) |
| hyp-geluquick | 14 | 7 | 7 | unstamped (pre-Exp617) |
| hyp-head4x4 | 14 | 6 | 8 | unstamped (pre-Exp617) |
| hyp-gate694p2 | 17 | 8 | 9 | stamped |
| hyp-lm4x4_40 | 18 | 9 | 9 | unstamped (pre-Exp617) |
| hyp-lm4x4im | 18 | 9 | 9 | unstamped (pre-Exp617) |
| hyp-a78 | 20 | 10 | 10 | unstamped (pre-Exp617) |
| hyp-convint8 | 20 | 10 | 10 | stamped |
| hyp-gate694p13 | 20 | 10 | 10 | stamped |
| hyp-vae4x4 | 21 | 10 | 11 | unstamped (pre-Exp617) |
| hyp-f16a78 | 23 | 11 | 12 | unstamped (pre-Exp617) |
| hyp-q8leg | 23 | 12 | 11 | unstamped (pre-Exp617) |
| hyp-t2 | 23 | 12 | 11 | unstamped (pre-Exp617) |
| hyp-vaeq8 | 23 | 12 | 11 | unstamped (pre-Exp617) |
| hyp-xwin | 23 | 10 | 13 | unstamped (pre-Exp617) |
| hyp-vaeq4 | 26 | 12 | 14 | unstamped (pre-Exp617) |
| hyp-q8xwin | 27 | 12 | 15 | unstamped (pre-Exp617) |
| hyp-q3km | 31 | 13 | 18 | unstamped (pre-Exp617) |
| hyp-q2k | 69 | 33 | 36 | unstamped (pre-Exp617) |
| hyp-cpp | 791 | 782 | 9 | unstamped (pre-Exp617) |
| hyp-aco | 1285 | 633 | 652 | unstamped (pre-Exp617) |
