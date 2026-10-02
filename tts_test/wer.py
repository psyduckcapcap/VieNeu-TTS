import sys, re, glob, unicodedata
from faster_whisper import WhisperModel
def norm(s):
    s = unicodedata.normalize("NFC", s.lower()); s = re.sub(r"[^\w\s]", " ", s); return s.split()
def wer(r, h):
    d = list(range(len(h)+1))
    for i in range(1, len(r)+1):
        prev, d[0] = d[0], i
        for j in range(1, len(h)+1):
            cur = min(d[j]+1, d[j-1]+1, prev + (r[i-1] != h[j-1])); prev, d[j] = d[j], cur
    return d[len(h)]/len(r)
ref = norm(open(sys.argv[1]).read())
m = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
for f in sorted(glob.glob(sys.argv[2])):
    segs, _ = m.transcribe(f, language="vi", beam_size=5)
    hyp = " ".join(s.text for s in segs)
    print(f"{f.split('/')[-1]}: WER {wer(ref, norm(hyp)):.1%} | words ref {len(ref)} hyp {len(norm(hyp))}", flush=True)
