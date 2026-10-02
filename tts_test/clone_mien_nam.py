"""Giọng Đức (âm sắc) + accent miền Nam (reference codes của giọng có sẵn miền Nam)."""
import json, numpy as np, soundfile as sf
from vieneu import Vieneu
D = "tts_test/out/clone_duc/"
TEXT = ("Hồi nhỏ, nhà tôi ở cạnh một con kênh nhỏ. Chiều nào ba tôi cũng chèo xuồng ra đồng, "
        "còn tôi ngồi phía mũi, ngắm mấy cây bần nghiêng mình soi bóng xuống mặt nước. "
        "Giờ nghĩ lại, sao mà thương quá chừng.")
tts = Vieneu(precision="int8")
enc = tts.engine._ensure_speaker_encoder()
def emb(p):
    w, sr = sf.read(p); w = w.mean(1) if w.ndim > 1 else w
    e = np.asarray(enc.embed(w.astype(np.float32), sr)).ravel(); return e / np.linalg.norm(e)
duc = np.array(json.load(open(D + "giong_duc.json"))["speaker_emb"], dtype=np.float32)
goc = emb(D + "goc.wav")
for nam in ["Đức Trí", "Thái Sơn", "Minh Triết", "Adam"]:
    p = tts.get_preset_voice(nam)
    tag = nam.replace(" ", "")
    a = tts.infer(TEXT, voice={"speaker_emb": duc, "codes": p["codes"]}); f = f"{D}nam_{tag}.wav"; tts.save(a, f)
    b = tts.infer(TEXT, voice=nam); fb = f"{D}nam_{tag}_preset.wav"; tts.save(b, fb)
    print(f"{nam}: dur {len(a)/48000:.1f}s | giống Đức gốc {float(goc @ emb(f)):.3f} "
          f"(preset gốc: {float(goc @ emb(fb)):.3f}) | giống preset {float(emb(fb) @ emb(f)):.3f}", flush=True)
