import time
from vieneu import Vieneu
text = open("tts_test/chapter.txt", encoding="utf-8").read()
tts = Vieneu(precision="int8")
t = time.time(); audio = tts.infer(text, voice="Hải Đăng"); el = time.time()-t
dur = len(audio)/48000
tts.save(audio, "tts_test/out/long_chapter.wav")
print(f"words {len(text.split())}, audio {dur:.1f}s ({len(text.split())/dur*60:.0f} words/min), gen {el:.1f}s, RTF {el/dur:.2f}")
