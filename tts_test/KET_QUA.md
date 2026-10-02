# Kết quả test VieNeu-TTS v3 Turbo cho đọc sách tiếng Việt

Môi trường: CPU 4 lõi (cloud, không GPU), `uv sync` (bản CPU, ONNX), mô hình mặc định v3 Turbo 48 kHz.

## Tốc độ
| Cấu hình | RTF (thời gian sinh / thời lượng audio) |
|---|---|
| fp32, giọng Hải Đăng | 0.87 – 1.15 |
| int8, giọng Ngọc Huyền | 0.64 – 1.09 |
| int8, cả chương ~445 từ (106 giây audio) | 0.68 |

Trên CPU 4 lõi này, máy đọc nhanh xấp xỉ thời gian thực. Như vậy một cuốn sách khoảng 10 giờ audio mất khoảng 7–10 giờ để sinh. Máy có GPU NVIDIA nhanh hơn khoảng 25–50 lần.

## Chất lượng (kiểm bằng nhận dạng giọng nói Whisper large-v3-turbo)
- Ngày tháng, số thập phân, %, viết tắt đều được đọc đúng. Ví dụ: `2/9/1945` → "2 tháng 9 năm 1945", `TP.HCM` → "Thành phố Hồ Chí Minh", `101,3`, `7,09%`, `1.700 km`.
- Với văn xuôi và hội thoại có ngoặc kép, gần như toàn bộ câu được nhận dạng đúng. Máy ngắt câu theo dấu câu hợp lý.
- Với cả một chương dài (gọi `infer()` một lần), không bị mất câu, không lặp và không bị "ảo giác".
- Có vài lỗi phát âm hoặc nhận dạng ở từ hiếm hoặc từ láy (ví dụ "róc rách", "khúc khuỷu", "Nghiêng nghiêng"). Từ "Chương" ở đầu văn bản mấy lần bị nghe thành "Trường/Trong". Một phần lỗi có thể do chính Whisper. Nên nghe thử để đánh giá.
- Tốc độ đọc khoảng 250 âm tiết/phút, hơi nhanh so với sách nói. SDK chưa có tham số chỉnh tốc độ. Có thể làm chậm khi hậu kỳ, ví dụ `ffmpeg -filter:a atempo=0.9` (xem `samples/long_chapter_cham_0.9x.mp3`).

## Mẫu nghe
Các file nằm trong `samples/`: 4 đoạn (văn xuôi, hội thoại, số liệu, từ khó) × 2 giọng, cùng một chương dài ở tốc độ gốc và tốc độ 0.9x.

## Chạy lại
```bash
uv sync
uv run python tts_test/test_book.py "Hải Đăng" fp32   # hoặc int8 nếu CPU có VNNI
uv run python tts_test/test_long.py                   # đọc tts_test/chapter.txt
```

## Clone giọng từ file của người dùng (`Giong_duck_goc.m4a`, 24 giây)
Không commit file ghi âm và embedding giọng lên repo (nằm trong `tts_test/out/`, đã gitignore).

So sánh cách clone (`clone_pick.py`, `clone_tune.py`, chấm bằng `wer.py`, độ giống = cosine speaker embedding so với file gốc):

| Đoạn mẫu | Cấu hình | Độ giống | WER (3 đoạn văn) |
|---|---|---|---|
| B: 15.9–23.9 s | **`use_ref_codes=False`, không lọc nhiễu** | **0.81** | **6.2%** |
| B | temperature 0.6 | 0.78 | 6.6% |
| B | mặc định (temperature 0.8) | 0.80 | 31.8% (lặp câu) |
| C: 6.3–13.75 s | temperature 0.6 | 0.75 | 14.2% |
| A: 1.15–8.0 s | temperature 0.6 / 0.8 | 0.73–0.75 | ~17% |
| (giọng có sẵn Hải Đăng) | — | 0.25 | — |

- Lọc nhiễu làm giảm độ giống (0.65–0.68 so với 0.68–0.76), vì file gốc vốn đã sạch.
- `use_ref_codes=False`: chỉ lấy âm sắc, không bắt chước nhịp ngắt của đoạn mẫu (đọc tin tức, ngắt nhiều). Cách này giảm hẳn lỗi lặp câu.
- Đọc cả chương: WER 11.0% (giọng Hải Đăng: 7.2%). Còn 1 chỗ lặp cụm từ ("đĩa lạc rang"). Nhịp đọc chậm hơn (161 s so với 106 s), khoảng 165 từ/phút, hợp với sách nói.
- Câu "Hãy subscribe cho kênh…" ở cuối bản chép là lỗi "ảo giác" quen thuộc của Whisper ở đoạn im lặng cuối file, audio không có câu này.
