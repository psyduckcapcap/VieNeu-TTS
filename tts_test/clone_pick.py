import numpy as np, soundfile as sf, time
from vieneu import Vieneu
D = "tts_test/out/clone_duc/"
tts = Vieneu(precision="int8")
enc = tts.engine._ensure_speaker_encoder()
def emb(p):
    w, sr = sf.read(p); w = w.mean(1) if w.ndim > 1 else w
    e = np.asarray(enc.embed(w.astype(np.float32), sr)).ravel(); return e/np.linalg.norm(e)
full = emb(D+"goc.wav")
text = "Mùa thu năm ấy, làng tôi chìm trong làn sương mỏng. Mỗi sáng, mẹ tôi dậy từ lúc gà gáy canh tư, nhóm bếp, nấu nồi cơm độn khoai."
for r in ["refA", "refB", "refC"]:
    for dn in [True, False]:
        name = f"{r}_{'dn' if dn else 'raw'}"
        tts.add_voice(name, D+r+".wav", denoise=dn)
        sims = []
        for k in range(2):
            a = tts.infer(text, voice=name); p = f"{D}{name}_{k}.wav"; tts.save(a, p); sims.append(float(full @ emb(p)))
        print(f"{name}: sim {np.mean(sims):.3f} ({', '.join(f'{s:.3f}' for s in sims)})", flush=True)
print("baseline Hải Đăng:", round(float(full @ emb("tts_test/out/clone_baseline_haidang.wav")),3))
