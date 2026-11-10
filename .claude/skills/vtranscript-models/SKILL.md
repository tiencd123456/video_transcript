---
name: vtranscript-models
description: Cấu hình API key và model cho vtranscript (.env, hồ sơ model), và quy trình thử model nhỏ/rẻ hơn rồi so chất lượng với bản tham chiếu. Dùng khi người dùng muốn đổi model, đổi key, tạo hồ sơ model mới, thử model rẻ hơn, hỏi chi phí, hoặc so chất lượng giữa các lần chạy.
---

# Cấu hình và thử model cho vtranscript

Trả lời người dùng bằng tiếng Việt (người dùng xưng "anh", em xưng "em").

## Cấu hình nằm ở đâu

- `.env` ở gốc dự án: `OPENROUTER_API_KEY` và bộ model mặc định của máy. File này nằm trong `.gitignore`, quyền 600. Mẫu đầy đủ ở `.env.example`.
  - Khi đã cài thành lệnh thì dùng `~/.config/vtranscript/.env`, hoặc trỏ tới file bằng `VT_ENV_FILE`.
  - Dòng `Cấu hình: …` ở đầu log cho biết đã nạp file nào.
- Hồ sơ model chọn bằng `--profile <tên>`. Hồ sơ không chứa key.
  - Có sẵn trong gói: `vtranscript/profiles/{best,budget,cheap}.env`.
  - Hồ sơ thử nghiệm tự tạo đặt ở `./profiles/<tên>.env` (thư mục dự án); trùng tên thì đè hồ sơ có sẵn.
- Thứ tự ưu tiên: tham số dòng lệnh > biến đã `export` trong shell > `--profile` > `.env` > mặc định trong code (`vtranscript/config.py`).
- Các biến:
  - `VT_ASR_BACKEND` (`api` | `local`), `VT_ASR_MODEL`;
  - `VT_LLM_BACKEND` (`openrouter` | `claude-cli` | `anthropic`);
  - `VT_LLM_VISION_MODEL`: đọc khung hình;
  - `VT_LLM_STRONG_MODEL`: lập từ điển, 1 lượt gọi;
  - `VT_LLM_MODEL`: sửa từng cụm, soát toàn cục, tóm tắt.

## Quy tắc về key và tiền (bắt buộc)

- **Không bao giờ in, echo, cat hay ghi key ra log hoặc ra file có theo dõi trên git.** Muốn kiểm tra `.env` thì dùng `grep -c '^OPENROUTER_API_KEY=.\+' .env`, không in giá trị. Không commit `.env`.
- Người dùng đổi key thì chỉ sửa dòng `OPENROUTER_API_KEY` trong `.env`.
- Key OpenRouter có giới hạn chi tiêu. **Luôn giữ lại ít nhất $2.** Trước mỗi lần thử có tốn tiền:
  1. chạy `.venv/bin/vtranscript-bench balance`;
  2. ước tính chi phí (bảng bên dưới);
  3. **hỏi người dùng trước khi chạy**, trừ khi người dùng đã duyệt đúng lần thử đó.
- Sau khi thử, chạy lại `bench balance` và báo số tiền đã tiêu.
- Ưu tiên lần thử rẻ nhất trả lời được câu hỏi: dùng `--reuse-from`, `--skip-video`, chỉ đổi một bước mỗi lần.

## Chi phí tham khảo (video 1 giờ, qua OpenRouter)

| Bước | Model | Chi phí |
|---|---|---|
| ASR | `elevenlabs/scribe-v2` | $0.22 |
| ASR | `x-ai/grok-stt-1.0` | $0.10 |
| ASR | `openai/whisper-large-v3(-turbo)` | khoảng $0.03 |
| Đọc khung hình (khoảng 50 khung) | Sonnet 5.5 / Haiku 5.5 / Gemini 3.1 Flash Lite | $0.28 / $0.016 / $0.02 |
| Lập từ điển | Opus 5.5 / Sonnet 5.5 | $0.25 / $0.12 |
| Sửa cụm + soát + tóm tắt | DeepSeek V4.1 Flash | khoảng $0.25 |

Tổng: best khoảng $1, budget khoảng $0.6, cheap khoảng $0.5.

## Quy trình thử một bộ model

1. Chọn bản tham chiếu: một lần chạy bằng hồ sơ `best` đã có, ví dụ `output/lark_meeting_original` (họp Lark 52 phút).
2. Tạo hoặc chọn hồ sơ. Hồ sơ mới thì chép `vtranscript/profiles/best.env` thành `profiles/<tên>.env` (tạo thư mục `profiles/` ở gốc dự án), sửa các dòng `VT_*`, và ghi ở đầu file chi phí ước tính cùng tình trạng ("chưa kiểm chứng").
3. Kiểm tra số dư, ước tính chi phí, hỏi người dùng.
4. Chạy:
   ```bash
   .venv/bin/vtranscript "<video>" -o output/_thu/<tên> --profile <tên> \
       --reuse-from output/<bản tham chiếu> --skip-video
   ```
   - `--reuse-from` hardlink audio, khung hình và cache LLM theo từng lượt gọi. Nó chỉ lấy `asr_raw.json` khi cùng model ASR, và chỉ lấy `visual_context.json` khi cùng model vision. Vì vậy chỉ các bước đổi model mới tốn tiền.
   - Video dài thì chạy nền và theo dõi log, đừng chờ ở tiền cảnh.
5. So sánh:
   ```bash
   .venv/bin/vtranscript-bench compare output/<bản tham chiếu> output/_thu/<tên> \
       --out output/_thu/<tên>/so_sanh.md
   ```
   Có thể đưa nhiều lần thử vào cùng một bảng.
6. Đọc kết quả, đừng chỉ nhìn một con số:
   - **Lệch sau chuẩn hóa:** đổi ASR thì dưới khoảng 5% là tương đương; Scribe thô so với bản chuẩn hóa của chính nó khoảng 5%. Chỉ đổi model LLM với cùng ASR thì nên dưới khoảng 3%.
   - **Nội dung bị mất:** khác 0% là dấu hiệu xấu; mở đúng các mốc thời gian được liệt kê.
   - **Bảng tên riêng (⚠):** từ điển mỏng sẽ làm tên bị viết sai hoặc không thống nhất.
   - **Đoạn không sửa được (LLM lỗi):** khác 0 nghĩa là có cụm còn nguyên bản thô.
   - Đọc 3 phút lệch nhiều nhất (bảng in sẵn tham chiếu và lần thử) để xem bên nào đúng. Bản tham chiếu cũng còn khoảng 2% lỗi.
   - Mở `03_tom_tat.md` của lần thử, xem có thiếu người, số liệu hoặc mục nào không.
7. Báo lại người dùng: bảng so sánh rút gọn, ví dụ cụ thể (đúng/sai), chi phí đã tiêu, khuyến nghị. Nếu khuyến nghị đổi mặc định thì hỏi trước rồi mới sửa `.env` hoặc `vtranscript/profiles/best.env`.
8. Ghi kết quả đo mới vào phần "Kết quả đã đo" bên dưới và vào README (mục *Chọn model và chi phí*).

## Kết quả đã đo (tháng 11/2026)

- **Lập từ điển quyết định chất lượng.**
  - Model rẻ lập từ điển mỏng (9–39 mục, Opus lập được 72) và sót biến thể ASR (2BX → BingX, tép → ref, kèo → KOL).
  - Sonnet: chưa đo đầu-cuối trên video Lark.
  - DeepSeek làm từ điển: chưa đo (đó là hồ sơ `cheap`).
- **Sửa từng cụm** (khi dùng từ điển Opus):
  - DeepSeek V4.1 Flash lệch 118 từ so với Opus; Sonnet lệch 182.
  - Bị loại: GPT-6 Luna và Gemini 3.1 Flash Lite (xóa "là/thì/cái"), GLM-5.3 (dồn chữ sang đoạn khác), Qwen 3.7 Plus (đổi nghĩa), Haiku (giữ câu ảo giác).
- **Đọc khung hình:**
  - Sonnet gần bằng Opus.
  - Haiku đủ tên người nhưng lẫn tên rác lấy từ khung chat.
  - Gemini 3.1 Flash Lite sạch nhưng sót.
  - DeepSeek nhiễu.
- **ASR** (họp Lark, cùng bộ LLM):
  - whisper-large-v3-turbo lệch 19,5%. Không có VAD nên sinh khoảng 40 câu ảo giác, và mất 3,1% nội dung ở 0:30–2:30.
  - grok-stt-1.0 lệch 13,7% và mất 1,4%.
  - qwen3-asr không có mốc thời gian từng từ.
  - Kết luận: giữ Scribe.

## Lỗi đã biết khi thử

- DeepSeek V4.1 Flash đôi khi chạm max_tokens:
  - `review()` đã tự chia đôi cửa sổ rồi thử lại.
  - Sửa từng cụm thì **chưa**: cụm lỗi bị giữ nguyên bản thô và hiện ở dòng "Đoạn không sửa được". Lần thử whisper bị 34 đoạn như vậy.
- Whisper qua API không có VAD, nên chỗ im lặng hoặc nhạc sinh câu ảo giác.
- Scribe qua OpenRouter từ chối `keyterms` (HTTP 400).
- Một bước được coi là "xong" chỉ dựa vào file kết quả có tồn tại hay không. Đổi model vision hoặc ASR cho một thư mục đã chạy rồi thì phải thêm `--redo visual` / `--redo asr`; tool sẽ cảnh báo khi model khác.
- Các lần chạy cũ chưa có `chi_phi.jsonl` sẽ hiện chi phí "?".
