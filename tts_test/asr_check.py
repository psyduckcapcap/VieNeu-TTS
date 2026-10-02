import sys, glob
from faster_whisper import WhisperModel
m = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
for f in sorted(glob.glob(sys.argv[1])):
    segs, _ = m.transcribe(f, language="vi", beam_size=5)
    print("==", f.split("/")[-1]); print(" ".join(s.text.strip() for s in segs)); print()
