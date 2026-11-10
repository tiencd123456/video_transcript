# vtranscript

Tool lấy transcript từ video, **chuẩn hóa kỹ** (chính tả, tên riêng, thuật ngữ, từ tiếng Anh bị phiên âm…), tóm tắt nội dung và in phụ đề vào video.

## Đầu ra

Kết quả nằm trong `output/<tên video>/`:

| File | Nội dung |
|---|---|
| `01_transcript_goc.txt` / `.srt` | Transcript gốc do ASR nhận dạng (mặc định ElevenLabs Scribe v2 qua API), giữ nguyên lỗi |
| `02_transcript_chuan_hoa.txt` / `.srt` / `.json` | Transcript đã chuẩn hóa |
| `02_bao_cao_chuan_hoa.md` | Báo cáo chuẩn hóa: từ điển đã áp dụng, **danh sách chỗ cần kiểm tra lại**, từng thay đổi (~~cũ~~ **mới**) |
| `03_tom_tat.md` | Tổng hợp nội dung và ý chính, kèm mốc thời gian và ảnh slide/màn hình của từng mục (ảnh nằm trong `03_tom_tat_hinh/`) |
| `04_video_phu_de.mp4` | Video đã in cứng phụ đề lấy từ bản chuẩn hóa |
| `_work/` | Dữ liệu trung gian và cache (cho phép chạy tiếp/chạy lại từng bước) |

## Cách dùng

```bash
vtranscript "video.mp4"
vtranscript "video.mp4" --glossary glossary.txt     # thêm từ điển tên riêng của anh
vtranscript "video.mp4" --redo normalize            # chạy lại từ bước chuẩn hóa
```

Lệnh `vtranscript` có sau khi cài gói (xem mục *Cài đặt*). Trong thư mục dự án thì kích hoạt môi trường trước (`source .venv/bin/activate`) hoặc gọi thẳng `.venv/bin/vtranscript`; `python -m vtranscript` cũng chạy được.

Các tùy chọn hay dùng (xem đủ bằng `--help`):

- `--glossary FILE`: từ điển tên riêng/thuật ngữ, mỗi dòng `Cách viết chuẩn = biến thể 1, biến thể 2` (xem `glossary.example.txt`). Đây là cách hiệu quả nhất để sửa đúng tên người, tên sản phẩm và tên nội bộ.
- `--asr-backend api | local`: `api` gọi ElevenLabs Scribe v2 qua OpenRouter (không cần GPU, khoảng $0.22/giờ audio); `local` chạy faster-whisper trên máy. Mặc định `auto`: có `OPENROUTER_API_KEY` thì dùng `api`. Đổi model bằng `--asr-model` (ví dụ `openai/whisper-large-v3` qua API, hoặc `large-v3-turbo` khi chạy `local` trên CPU).
- `--verbatim`: giữ nguyên tiếng ngập ngừng (ờ, ừm…) và chỗ nói lắp. Mặc định tool chỉ bỏ những chỗ này; các từ của văn nói như "là", "thì", "mà", "cái" vẫn được giữ.
- `--no-visual`: không đọc chữ trên màn hình/slide.
- `--llm-backend openrouter | claude-cli | anthropic`: chọn nơi gọi LLM (xem mục *Chọn model và chi phí*). Mặc định `auto`: có `OPENROUTER_API_KEY` thì dùng OpenRouter, không thì dùng Claude Code CLI, cuối cùng là Claude API.
- `--llm-model`, `--llm-strong-model`, `--llm-vision-model`: đổi model cho từng việc (sửa từng cụm/soát/tóm tắt; lập từ điển; đọc khung hình).
- `--profile best | budget | cheap`: chọn cả bộ model cùng lúc (xem mục *Cấu hình*).
- `--reuse-from output/<video>`: dùng lại audio, ASR, khung hình và cache LLM của một lần chạy trước, để thử bộ model khác mà không trả tiền lại các bước giống nhau.
- `--batch-size 4` hoặc `0`: giảm mức dùng VRAM nếu GPU báo hết bộ nhớ (chỉ với `--asr-backend local`).
- `--skip-video`: chỉ tạo transcript và tóm tắt, không xuất video.
- `--no-summary-images`: không chèn ảnh màn hình vào bản tóm tắt.

## Quy trình xử lý

1. **Tách âm thanh** 16 kHz mono (ffmpeg).
2. **Đọc ngữ cảnh hình ảnh**: chọn các khung hình có nội dung thay đổi (chuyển slide, đổi màn hình chia sẻ), cho LLM đọc chữ trên slide, tên người tham gia và tên thương hiệu. Đây là nguồn chính tả chuẩn cho tên riêng và thuật ngữ.
3. **ASR**, có mốc thời gian và độ tin cậy từng từ. Có hai cách:
   - **Qua API (mặc định):** ElevenLabs Scribe v2 qua OpenRouter. Audio được chia thành các phần khoảng 20 phút, cắt ở chỗ lặng, gửi song song; kết quả từng phần được cache nên chạy lại không mất tiền lần nữa. OpenRouter chưa hỗ trợ truyền từ khóa (keyterms) cho Scribe, nên tên riêng do bước chuẩn hóa sửa.
   - **Trên máy:** faster-whisper, có VAD. Tên riêng tìm được ở bước 2 được dùng để mồi (hotwords). Quãng nào Whisper "ảo giác" (ví dụ "Hãy subscribe cho kênh…" đè lên lời nói thật) sẽ được tự phát hiện và nhận dạng lại ở chế độ cẩn thận hơn. Trên máy chỉ có CPU nên dùng `--asr-model large-v3-turbo`: nhanh hơn khoảng 2,7 lần và không bị ảo giác như large-v3 trong lần đo thử.
4. **Chuẩn hóa 3 vòng**:
   - **Vòng 1, lập từ điển**: đọc toàn bộ transcript cùng ngữ cảnh hình ảnh để xác định chủ đề, người nói, tên riêng và thuật ngữ, kèm các biến thể ASR hay nghe nhầm (ví dụ "pát nơ" → partner, "kê pi ai" → KPI).
   - **Vòng 2, hiệu đính từng cụm** (khoảng 2.800 ký tự, có ngữ cảnh trước và sau, chạy song song). Vòng này sửa lỗi đồng âm, phiên âm tiếng Anh, tên riêng, chính tả, số liệu, dấu câu và viết hoa, đồng thời xóa ảo giác hoặc câu lặp của Whisper. Những từ Whisper nhận dạng kém tự tin được đánh dấu ‹ › để LLM soi kỹ hơn.
   - **Vòng 3, soát toàn cục**: bắt chỗ viết không nhất quán giữa các cụm, lỗi còn sót và chỗ sửa quá tay (so lại với bản ASR gốc).
   - Chỗ không đủ căn cứ để sửa thì **giữ nguyên và đưa vào danh sách "Cần kiểm tra lại"** trong báo cáo, không đoán bừa.
5. **Tóm tắt** nội dung từ bản chuẩn hóa. Sau đó, với mỗi mục có mốc thời gian, tool chèn ảnh màn hình hiển thị lâu nhất trong quãng đó (slide, trang được chia sẻ). Khung chỉ có avatar hoặc màn hình trống được bỏ qua, và một màn hình không bị lặp lại ở nhiều mục. Bước này không gọi LLM.
6. **Phụ đề**: căn từng từ của bản chuẩn hóa với mốc thời gian từng từ của ASR, chia câu tối đa 2 dòng × khoảng 42 ký tự, rồi in vào video (libass + libx264, giữ nguyên audio).

Mọi bước đều có cache. Nếu bị ngắt giữa chừng, chạy lại đúng lệnh cũ là tool chạy tiếp từ chỗ dừng.

## Cài đặt

Cần Python 3.10 trở lên. ffmpeg (có libass để in phụ đề) đi kèm gói `imageio-ffmpeg`, không phải cài riêng. Muốn dùng ffmpeg khác thì đặt `VT_FFMPEG=/đường/dẫn/ffmpeg`.

**Trên máy đang phát triển** (đã cài sẵn trong `.venv` của thư mục này, chế độ `-e` nên sửa mã nguồn là có hiệu lực ngay):

```bash
uv venv .venv && uv pip install -p .venv -e ".[gpu,anthropic]"
```

**Cài thành lệnh dùng ở mọi thư mục** (ví dụ trên máy chủ):

```bash
uv build                                             # tạo dist/vtranscript-<phiên bản>-py3-none-any.whl
uv tool install dist/vtranscript-0.1.0-py3-none-any.whl     # hoặc: pipx install dist/vtranscript-...whl
mkdir -p ~/.config/vtranscript && cp .env.example ~/.config/vtranscript/.env && chmod 600 ~/.config/vtranscript/.env
```

Sau đó lệnh `vtranscript` và `vtranscript-bench` chạy được ở bất kỳ thư mục nào. Kết quả ghi vào `output/<tên video>/` của thư mục hiện tại, hoặc nơi chỉ định bằng `-o`.

Nhóm cài thêm (ghi trong ngoặc vuông, ví dụ `".[local,anthropic]"`):

| Nhóm | Khi nào cần |
|---|---|
| (không có) | ASR và LLM đều gọi qua API OpenRouter. Máy chủ không cần GPU. Đây là cách dùng mặc định. |
| `local` | Chạy ASR trên máy bằng faster-whisper (`--asr-backend local`), CPU |
| `gpu` | Như `local`, kèm thư viện CUDA để chạy trên card NVIDIA |
| `anthropic` | Gọi LLM thẳng Claude API (`--llm-backend anthropic`) |

Cần thêm một trong ba: `OPENROUTER_API_KEY` (điền vào `.env`, xem mục *Cấu hình*), Claude Code (lệnh `claude`, đã đăng nhập) hoặc `ANTHROPIC_API_KEY`. Khi dùng Claude API, tool bật cơ chế fallback phía server: nếu bộ lọc an toàn từ chối nhầm, API tự chạy lại yêu cầu trên model dự phòng.

## Cấu hình: `.env` và hồ sơ model

```bash
cp .env.example .env && chmod 600 .env   # rồi điền OPENROUTER_API_KEY
```

- **`.env`** giữ API key và bộ model mặc định của máy này. File nằm trong `.gitignore` nên không lên git. Đổi key (ví dụ sau khi xoay key) thì chỉ cần sửa file này. Tool dùng file đầu tiên tìm thấy:
  1. đường dẫn trong biến `VT_ENV_FILE`;
  2. `.env` ở thư mục hiện tại;
  3. `.env` ở thư mục mã nguồn;
  4. `~/.config/vtranscript/.env`.

  Dòng `Cấu hình: …` ở đầu log cho biết đã dùng file nào.
- **Hồ sơ model** là các bộ model đặt sẵn, chọn bằng `--profile <tên>` cho từng lần chạy.
  - Có sẵn `best`, `budget`, `cheap` (trong `vtranscript/profiles/`, đi kèm gói).
  - Hồ sơ tự tạo đặt ở `./profiles/<tên>.env` hoặc `~/.config/vtranscript/profiles/<tên>.env`; trùng tên thì đè hồ sơ có sẵn.
  - Cũng có thể truyền thẳng đường dẫn: `--profile thu/abc.env`.
  - Hồ sơ chỉ ghi model, không chứa key. Muốn tạo hồ sơ mới thì chép một file có sẵn rồi sửa các dòng `VT_*`.
- **Thứ tự ưu tiên:** tham số dòng lệnh > biến đã `export` trong shell > `--profile` > `.env` > mặc định trong code.

| Biến | Bước | Tham số dòng lệnh tương ứng |
|---|---|---|
| `VT_ASR_BACKEND`, `VT_ASR_MODEL` | Nhận dạng giọng nói | `--asr-backend`, `--asr-model` |
| `VT_LLM_BACKEND` | Nơi gọi LLM | `--llm-backend` |
| `VT_LLM_VISION_MODEL` | Đọc chữ trên khung hình | `--llm-vision-model` |
| `VT_LLM_STRONG_MODEL` | Lập từ điển tên riêng, thuật ngữ (1 lượt gọi) | `--llm-strong-model` |
| `VT_LLM_MODEL` | Sửa từng cụm, soát toàn cục, tóm tắt | `--llm-model` |

Mỗi lần chạy ghi chi phí thực tế (theo model) vào `output/<video>/_work/chi_phi.jsonl`. Bước nào lấy từ cache thì không tính tiền.

## Thử model rẻ hơn và so chất lượng

```bash
vtranscript-bench balance          # số dư key OpenRouter, xem trước và sau khi thử

# Chạy lại video đã có bản tham chiếu, bằng bộ model khác, vào thư mục riêng:
vtranscript "video.mp4" -o output/_thu/budget --profile budget --reuse-from output/<video> --skip-video

# So với bản tham chiếu (thường là lần chạy bằng hồ sơ best):
vtranscript-bench compare output/<video> output/_thu/budget --out output/_thu/so_sanh.md
```

- `--reuse-from` lấy lại từ lần chạy cũ những gì không phụ thuộc model đang thử: audio, khung hình, kết quả ASR (nếu cùng model ASR), kết quả đọc khung hình (nếu cùng model vision) và cache từng lượt gọi LLM. Nhờ vậy chỉ trả tiền cho các bước thật sự đổi model.
- Bảng so sánh gồm: model từng bước, **% lệch sau chuẩn hóa** so với bản tham chiếu, % nội dung bị mất (kèm mốc thời gian), số mục từ điển, số chỗ cần kiểm tra lại, số đoạn LLM lỗi, câu ảo giác còn sót, độ dài tóm tắt, chi phí; bảng đếm số lần xuất hiện từng tên riêng/thuật ngữ (⚠ khi lần thử thấp hơn 60% bản tham chiếu); 3 phút lệch nhiều nhất của mỗi lần thử để nghe lại.
- Bản tham chiếu không phải sự thật tuyệt đối (chính nó còn khoảng 2% lỗi). Con số chỉ cho biết lần thử gần hay xa bản tốt nhất; vẫn cần đọc các phút lệch nhiều và bản tóm tắt.

## Chọn model và chi phí

| Backend | Model mặc định | Chi phí cho video 68 phút |
|---|---|---|
| `openrouter` | ASR: `elevenlabs/scribe-v2` · Đọc khung hình: `anthropic/claude-sonnet-5.5` · Lập từ điển: `anthropic/claude-opus-5.5` · Sửa từng cụm, soát, tóm tắt: `deepseek/deepseek-v4.1-flash` | khoảng $1 (ASR khoảng $0.25) |
| `claude-cli` | `opus` cho mọi bước | Tính vào gói Claude đang dùng, không trả thêm |
| `anthropic` | `claude-opus-5-5` cho mọi bước | ước tính khoảng $4 |

```bash
vtranscript "video.mp4"                    # bộ model trong .env (mặc định: best)
vtranscript "video.mp4" --profile budget   # dùng hồ sơ budget
```

| Hồ sơ | Đọc khung hình | Lập từ điển | Sửa cụm, soát, tóm tắt | Chi phí video 1 giờ | Tình trạng |
|---|---|---|---|---|---|
| `best` | Sonnet 5.5 | Opus 5.5 | DeepSeek V4.1 Flash | khoảng $1 | Đã kiểm chứng trên 2 video |
| `budget` | Haiku 5.5 | Sonnet 5.5 | DeepSeek V4.1 Flash | khoảng $0.6 | Đã thử từng bước, chưa chạy trọn |
| `cheap` | Haiku 5.5 | DeepSeek V4.1 Flash | DeepSeek V4.1 Flash | khoảng $0.5 | Chưa kiểm chứng |

Cả ba hồ sơ đều dùng ASR Scribe v2 (khoảng $0.22/giờ). Mức tiết kiệm có giới hạn vì ASR và bước sửa từng cụm (khoảng $0.25) là chi phí cố định.

Lý do chọn như vậy. Số liệu dưới đây đo trên một video đào tạo 68 phút, so với bản chuẩn hóa của Opus. "Lệch Opus" là tổng số từ khác bản Opus, không tính dấu câu; để tham chiếu, Sonnet lệch 182 từ.

- **Lập từ điển là bước quyết định chất lượng.** Model rẻ lập từ điển quá mỏng (9–39 mục, Opus lập được 72), nên bỏ sót các biến thể ASR như "2BX" → BingX, "tép" → ref, "kèo" → KOL. Bước này chỉ có 1 lượt gọi nên dùng Opus (khoảng $0.25). Muốn tiết kiệm thì `--llm-strong-model anthropic/claude-sonnet-5.5` (khoảng $0.12).
- **Khi đã có từ điển tốt, model rẻ sửa từng cụm rất sát Opus.** Với từ điển của Opus, DeepSeek V4.1 Flash chỉ lệch 118 từ, tốn $0.21 và chạy hết 6 phút.
- **Các model rẻ khác không đạt:**
  - GPT-6 Luna và Gemini 3.1 Flash Lite xóa hết "là", "thì", "cái" và làm rơi nội dung (lệch hơn 600 từ).
  - GLM-5.3 dồn chữ sang đoạn khác, làm lệch mốc thời gian phụ đề.
  - Qwen 3.7 Plus đổi nghĩa ("hạn chế" → "chiết khấu").
  - Haiku giữ nguyên câu ảo giác của Whisper.
- **Đọc khung hình:** Sonnet đọc đầy đủ nhất ($0.28/video). Haiku chỉ tốn $0.02 và đọc đủ tên người, tổ chức, nhưng lẫn một số tên rác (chữ trong khung chat).
- **Tóm tắt** bằng DeepSeek V4.1 Flash tốn khoảng $0.015, nội dung đúng và chi tiết.

- **ASR:** Scribe v2 có tỉ lệ lỗi từ tiếng Việt thấp nhất trong các model có trên OpenRouter (FLEURS: Scribe v2 khoảng 5,6%, Whisper large-v3 khoảng 7–11%) và độ tin cậy từng từ sát thực tế, nên việc đánh dấu chữ nghe kém ở bước chuẩn hóa vẫn dùng được. Các model ASR rẻ hơn đã thử trọn quy trình trên video họp Lark 52 phút (cùng bộ LLM, so với bản chuẩn hóa từ Scribe):
  - `openai/whisper-large-v3-turbo`: lệch 19,5% sau chuẩn hóa. Qua API không có VAD nên sinh khoảng 40 câu ảo giác; bước chuẩn hóa xóa được, nhưng lời nói thật bên dưới bị mất (khoảng 2 phút báo cáo ở 0:30–2:30, tóm tắt cũng mất phần này).
  - `x-ai/grok-stt-1.0` (khoảng $0.10/giờ): lệch 13,7%, không ảo giác nhưng bỏ sót vài quãng ngắn.
  - `qwen3-asr`: không trả mốc thời gian từng từ nên không làm được phụ đề.
  - Mỗi video chỉ tiết kiệm được $0.1–0.17, nên vẫn giữ Scribe; muốn tiết kiệm thì đổi model ở phía LLM.
