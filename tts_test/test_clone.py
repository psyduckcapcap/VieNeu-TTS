import time, numpy as np, soundfile as sf
from vieneu import Vieneu
tts = Vieneu(precision="int8")
ref = "examples/audio_ref/example_2.wav"
print("ref dur", round(sf.info(ref).duration,1), "s")
t = time.time(); tts.add_voice("GiongClone", ref); print(f"enroll {time.time()-t:.1f}s")
text = "Đêm ấy trời đổ mưa to. Tôi ngồi bên cửa sổ, lắng nghe tiếng mưa rơi lộp độp trên mái tôn, lòng bỗng nhớ về những ngày xưa cũ."
a = tts.infer(text, voice="GiongClone"); tts.save(a, "tts_test/out/clone.wav")
b = tts.infer(text, voice="Hải Đăng"); tts.save(b, "tts_test/out/clone_baseline_haidang.wav")
enc = tts.engine._ensure_speaker_encoder()
def emb(p):
    w, sr = sf.read(p); w = w.mean(1) if w.ndim > 1 else w
    e = np.asarray(enc.embed(w.astype(np.float32), sr)).ravel(); return e/np.linalg.norm(e)
r = emb(ref)
print("cos(ref, clone)    =", round(float(r @ emb("tts_test/out/clone.wav")),3))
print("cos(ref, Hải Đăng) =", round(float(r @ emb("tts_test/out/clone_baseline_haidang.wav")),3))
