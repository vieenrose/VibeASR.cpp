Exp842 paired arms on eval-bilingual/holdout_zh.wav (213 s, 468 ref zh tokens), lean tier (PIECES=13 VAE_DEFER_LATE=1); ON arm additionally had BOUND_FORCE=1 (knob since removed).
Re-pair: python3 .auto/compare_arms.py .auto/hyp-stream/zh842-lean-batchOFF.txt .auto/hyp-stream/zh842-lean-batchON.txt ../eval-bilingual/manifest_holdout_zh.json
Result at commit time: b=2 c=1 McNemar p=1.0, WER 15.38% vs 15.17%, ins 2/2, del 8/8.
