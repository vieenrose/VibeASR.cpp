#!/usr/bin/env python3
"""Score an on-device hyp set: .auto/score_hyp.py <tag>  -> eval-librispeech/hyp-<tag>/*.txt
Same normalization as score_accuracy.py (speaker tags / brackets stripped)."""
import sys, os, json, re, glob
import jiwer

tag = sys.argv[1] if len(sys.argv) > 1 else "a78"
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval-librispeech")
refs = json.load(open(os.path.join(root, "refs.json")))
pt = {f["audio"].replace(".wav", ""): f["text"] for f in
      json.load(open(os.path.join(root, "results_pt_gpu_ref.json")))["files"]}

strip_spk = re.compile(r"speaker\s*\d+\s*:", re.I)
strip_brk = re.compile(r"\[.*?\]")
norm = lambda s: strip_brk.sub(" ", strip_spk.sub(" ", s))
tr = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.Strip(),
                    jiwer.ReduceToListOfListOfWords()])

keys = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(root, f"hyp-{tag}", "*.txt")))
keys = [k for k in keys if open(os.path.join(root, f"hyp-{tag}", k + ".txt"), encoding="utf-8").read().strip()]
print(f"hyp-{tag}: {len(keys)}/{len(refs)} utterances scored")
g = [refs[k] for k in keys]
h = [norm(open(os.path.join(root, f"hyp-{tag}", k + ".txt"), encoding="utf-8").read()) for k in keys]
out = jiwer.process_words(g, h, reference_transform=tr, hypothesis_transform=tr)
print(f"{tag} vs GROUND TRUTH : WER {out.wer:.4f} (S={out.substitutions} D={out.deletions} I={out.insertions} H={out.hits})")
pk = [k for k in keys if k in pt]
if pk:
    o2 = jiwer.process_words([norm(pt[k]) for k in pk], [norm(open(os.path.join(root, f"hyp-{tag}", k + ".txt"), encoding="utf-8").read()) for k in pk],
                             reference_transform=tr, hypothesis_transform=tr)
    print(f"{tag} vs PT-OFFICIAL: WER {o2.wer:.4f} (S={o2.substitutions} D={o2.deletions} I={o2.insertions})  [{len(pk)} utts]")
per = sorted(((jiwer.process_words([refs[k]], [norm(open(os.path.join(root, f'hyp-{tag}', k + '.txt'), encoding='utf-8').read())],
             reference_transform=tr, hypothesis_transform=tr).wer, k) for k in keys), reverse=True)
print("worst:", ", ".join(f"{k}:{w:.2f}" for w, k in per[:4]))
