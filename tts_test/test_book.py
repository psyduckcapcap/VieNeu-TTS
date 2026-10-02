import time, sys
from vieneu import Vieneu

TEXTS = {
 "van_xuoi": "Chương một. Mùa thu năm ấy, làng tôi chìm trong làn sương mỏng. Mỗi sáng, mẹ tôi dậy từ lúc gà gáy canh tư, nhóm bếp, nấu nồi cơm độn khoai. Tiếng chổi tre xào xạc ngoài ngõ, tiếng ai gọi nhau í ới bên kia sông, tất cả hòa vào nhau thành một bản nhạc quen thuộc mà mãi về sau, khi đã đi xa, tôi vẫn còn nhớ như in.",
 "hoi_thoai": "Ông lão nhìn tôi, chậm rãi hỏi: “Cháu đi đâu mà vội thế?” Tôi đáp: “Dạ, cháu lên tỉnh học ạ.” Ông gật gù, cười hiền: “Ừ, học cho giỏi vào, rồi về giúp làng mình nhé!”",
 "so_lieu": "Ngày 2/9/1945, tại Quảng trường Ba Đình, Chủ tịch Hồ Chí Minh đọc bản Tuyên ngôn độc lập. Đến năm 2024, dân số Việt Nam đạt khoảng 101,3 triệu người, GDP tăng 7,09%. TP.HCM và Hà Nội là hai đô thị lớn nhất, cách nhau hơn 1.700 km.",
 "tu_kho": "Nghiêng nghiêng bóng nguyệt khuya, khuya khoắt; thoang thoảng hương ngâu, ngào ngạt. Những tiếng lách tách, róc rách, khúc khuỷu, ngoằn ngoèo của con suối nhỏ nghe sao mà thương đến thế.",
}
voice = sys.argv[1] if len(sys.argv) > 1 else "Hải Đăng"
precision = sys.argv[2] if len(sys.argv) > 2 else "fp32"
t0 = time.time(); tts = Vieneu(precision=precision); print(f"load {time.time()-t0:.1f}s")
for name, text in TEXTS.items():
    t = time.time(); audio = tts.infer(text, voice=voice); el = time.time()-t
    dur = len(audio)/48000
    out = f"tts_test/out/{voice.replace(' ','_')}_{precision}_{name}.wav"
    tts.save(audio, out)
    print(f"{name}: audio {dur:.1f}s, gen {el:.1f}s, RTF {el/dur:.2f} -> {out}")
