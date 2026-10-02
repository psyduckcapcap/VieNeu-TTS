import numpy as np, soundfile as sf, re, sys, json
from vieneu import Vieneu
D = "tts_test/out/clone_duc/"
paras = open("tts_test/chapter.txt", encoding="utf-8").read().split("\n\n")
text = "\n\n".join([paras[2], paras[3], paras[6]])
open(D+"tune_text.txt","w").write(text)
tts = Vieneu(precision="int8")
enc = tts.engine._ensure_speaker_encoder()
def emb(p):
    w, sr = sf.read(p); w = w.mean(1) if w.ndim > 1 else w
    e = np.asarray(enc.embed(w.astype(np.float32), sr)).ravel(); return e/np.linalg.norm(e)
full = emb(D+"goc.wav")
cfgs = [("B_t08","refB",True,0.8),("B_t06","refB",True,0.6),("A_t08","refA",True,0.8),
        ("A_t06","refA",True,0.6),("C_t06","refC",True,0.6),("B_noref","refB",False,0.8)]
for name, ref, urc, temp in cfgs:
    if ref not in tts._preset_voices: tts.add_voice(ref, D+ref+".wav", denoise=False)
    a = tts.infer(text, voice=ref, temperature=temp, use_ref_codes=urc)
    p = f"{D}tune_{name}.wav"; tts.save(a, p)
    print(f"{name}: dur {len(a)/48000:.1f}s sim {float(full@emb(p)):.3f}", flush=True)
