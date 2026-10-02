"""Clone giọng từ Giong_duck_goc.m4a và đọc tts_test/chapter.txt.

Cấu hình đã chọn sau khi so sánh (xem KET_QUA.md): đoạn 15.9–23.9 s của file gốc,
không lọc nhiễu (file đã sạch), use_ref_codes=False (chỉ lấy âm sắc, không lấy
nhịp ngắt của đoạn mẫu) → ổn định nhất, ít lặp từ nhất.
"""
import json, sys, time
from vieneu import Vieneu
D = "tts_test/out/clone_duc/"
tts = Vieneu(precision="int8")
tts.add_voice("Đức", D + "refB.wav", denoise=False, use_ref_codes=False)
emb = tts.get_preset_voice("Đức")["speaker_emb"]   # lưu riêng giọng này; không sửa file voices gốc của repo
json.dump({"name": "Đức", "speaker_emb": [round(float(x), 6) for x in emb.ravel()], "codes": None},
          open(D + "giong_duc.json", "w"), ensure_ascii=False)
src = sys.argv[1] if len(sys.argv) > 1 else "tts_test/chapter.txt"
out = sys.argv[2] if len(sys.argv) > 2 else D + "duc_chapter.wav"
text = open(src, encoding="utf-8").read()
t = time.time(); a = tts.infer(text, voice="Đức", use_ref_codes=False); el = time.time() - t
tts.save(a, out); print(f"audio {len(a)/48000:.1f}s gen {el:.1f}s -> {out}")
