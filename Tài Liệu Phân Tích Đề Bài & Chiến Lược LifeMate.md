# TÀI LIỆU TỔNG QUAN & PHÂN TÍCH CHIẾN THUẬT: LIFEMATE (ZALO AI CHALLENGE)
**Vai trò tài liệu:** Cẩm nang kỹ thuật và quy chuẩn bắt buộc cho toàn bộ thành viên trong Team.

---

## 1. Bản Chất Bài Toán (Executive Summary)

**LifeMate** yêu cầu xây dựng một **Voice-driven AI Agent** (Tác tử AI điều khiển bằng giọng nói) có khả năng biến các câu lệnh tiếng Việt tự nhiên (có pha trộn từ tiếng Anh thông dụng như *"book"*, *"AC"*) thành **chuỗi hành động gọi công cụ (multi-step tool calls)** chính xác trên các thiết bị không màn hình hoặc màn hình nhỏ (loa thông minh, xe hơi, tai nghe, đồng hồ...).

> **Triết lý cốt lõi của đề bài:** *"The agent is judged on what it does, not what it says."* (Hệ thống được chấm điểm dựa trên **trạng thái ứng dụng sau khi thực thi** và **danh sách các hàm đã gọi**, không chấm câu trả lời văn bản giao tiếp với người dùng).

### Ba kịch bản đầu ra (Output Scenarios) mà Agent phải xử lý:
1. **Thực thi hành động (Execute actions):** Lập kế hoạch và gọi tuần tự các công cụ (`read` $\rightarrow$ `write`) để đưa dữ liệu của các ứng dụng về đúng trạng thái cuối cùng mà người dùng yêu cầu.
2. **Hỏi thêm thông tin bị thiếu (Request missing info):** Khi yêu cầu của người dùng chưa đủ tham số bắt buộc, gọi công cụ đặc biệt `sys.ask_missing_info` để chỉ rõ `tool_name` dự định gọi và danh sách `params` còn thiếu. *(Lưu ý: Lệnh này có thể được gọi SAU KHI Agent đã thực hiện một số lệnh `read` để kiểm tra dữ liệu).*
3. **Từ chối khéo / Ngoài phạm vi (Out-of-scope):** Trả về danh sách rỗng `[]` nếu không có ứng dụng nào được cài đặt (`context.installed_apps`) có thể giải quyết yêu cầu.

---

## 2. Giải Phẫu Dữ Liệu & Môi Trường Thực Thi (Data & Environment)

### 2.1. Quy mô các tập dữ liệu (Splits)
* **Train:** $\approx 1,600$ mẫu (Có đầy đủ nhãn: `audio_text` transcript, `tool_calls` chuẩn, `tool_results`, và `final_state`).
* **Public Test:** $\approx 600$ mẫu (Dùng để đánh giá trên Public Leaderboard trong quá trình phát triển).
* **Private Test:** $\approx 600$ mẫu (Mở sau khi đã đóng băng Docker; quyết định thứ hạng chung cuộc).

### 2.2. Cấu trúc Input của mỗi mẫu (Sample)
1. **Audio (`audio/<id>/<id>.wav`):**
   * Định dạng: WAV, $16\text{ kHz}$, mono, $16\text{-bit}$.
   * Đặc điểm: $100\%$ giọng thu âm người thật ở nhiều độ tuổi, giới tính và đủ 3 vùng miền (Bắc, Trung, Nam), chứa cả câu lệnh mới nhất lẫn các lượt nói trước đó trong lịch sử hội thoại. Có hiện tượng code-switching (chèn từ tiếng Anh).
2. **Ngữ cảnh (`context` trong `<split>.jsonl`):**
   * `current_time`, `timezone`, vị trí hiện tại (nếu có), `user_profile` (ví dụ: địa chỉ nhà, địa chỉ công ty).
   * `installed_apps`: Danh sách các ứng dụng đang được cài trên thiết bị của mẫu đó. **Chỉ được phép gọi các tool nằm trong `installed_apps` này (cộng với `sys.ask_missing_info`).**
3. **Lịch sử hội thoại (`history`):** Các lượt tương tác trước đó trong cùng một phiên hội thoại.
4. **Danh mục công cụ (`<split>_tools_catalog.json`):**
   * Định nghĩa toàn bộ JSON Schema của các tool.
   * Phân loại `kind`: `read` (đọc dữ liệu), `write` (thay đổi dữ liệu), `meta` (`sys.ask_missing_info`).
   * **Ba thuộc tính Schema cực kỳ quan trọng cần lưu ý:**
     * `x-ref`: Tham số bắt buộc phải là một ID hợp lệ tồn tại trong dữ liệu ứng dụng (VD: `"device_id"` có `"x-ref": "smart_home.devices"`). **Hệ quả:** Agent bắt buộc phải gọi hàm `read` (như `smart_home.get_status`) trước để lấy ID thật, tuyệt đối không tự bịa (hallucinate) ID!
     * `x-match: "fuzzy"`: Tham số dạng văn bản tự do (free-text), được chấm bằng độ tương đồng từ vựng (token-F1) và không ảnh hưởng tới `state_hash`.
     * `x-one-of`: Yêu cầu ít nhất một trong các nhóm tham số được liệt kê phải xuất hiện (VD: truyền `contact_id` HOẶC `phone_number`).

### 2.3. Cơ chế của Môi trường `zaic_tools` (`ToolEnv`)
* Mỗi mẫu chạy trên **duy nhất 1 instance** `ToolEnv`:
  ```python
  env = ToolEnv(SPLIT, item["id"], root=ROOT)
  ```
* Mọi lệnh gọi `env.call(tool_name, arguments)` — **dù thành công (`"ok": true`) hay thất bại (`"ok": false`)** — đều bị ghi nhận vào lịch sử `env.get_tool_calls()`.
* Sau khi kết thúc phiên suy luận của một mẫu, ta nộp `env.get_tool_calls()` và mã băm trạng thái cuối cùng `env.get_state_hash()`.

---

## 3. Phân Tích Toán Học Hàm Đánh Giá (Evaluation Metrics) & Các "Cái Bẫy"

Đây là phần quan trọng nhất mà mọi thành viên làm Modeling và Agent Logic phải thuộc lòng.

### 3.1. Công thức tính điểm cho từng mẫu ($\text{Score}_i$)

**Trường hợp 1: Mẫu Out-of-scope (Danh sách nhãn chuẩn $G$ rỗng, tức $|G| = 0$)**
* Nếu danh sách nộp lên cũng rỗng (`[]`): $\text{Score}_i = 1$.
* Nếu danh sách nộp lên khác rỗng: $\text{Score}_i = 0$.

**Trường hợp 2: Danh sách nhãn chuẩn khác rỗng ($|G| > 0$)**
Điểm của mẫu được tính theo công thức:

$$\text{Score}_i = \begin{cases} 0, & \text{nếu } \text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}} \text{ hoặc } M = 0 \\ \frac{2 \cdot P \cdot R}{P + R}, & \text{ngược lại} \end{cases}$$

Trong đó Precision ($P$), Recall ($R$) và Điểm chung cuộc ($\text{Final score}$) trên $N$ mẫu là:

$$P = \frac{M}{M + U}, \quad R = \frac{M}{|G|}$$

$$\text{Final score} = \frac{100}{N} \sum_{i=1}^{N} \text{Score}_i$$

**Giải nghĩa các biến số:**
* $|G|$: Tổng số lệnh gọi hàm trong nhãn chuẩn (gold calls).
* $M$: Số lệnh gọi hàm của hệ thống **khớp (matched)** với nhãn chuẩn (quan hệ ghép cặp $1-1$, không xét thứ tự khi tính $M$).
* $U$: Số lệnh gọi hàm của hệ thống **không khớp (unmatched)** với nhãn chuẩn (**bao gồm cả các lệnh gọi bị lỗi schema hoặc trả về `"ok": false`**).

### 3.2. Điều kiện để 1 lệnh gọi hàm được tính là Khớp (Matched)
Một lệnh gọi hàm nộp lên được tính là khớp với một lệnh trong nhãn chuẩn khi thỏa mãn đồng thời 3 điều kiện:
1. **Tên công cụ (`name`)** giống hệt nhau.
2. **Mọi tham số không phải fuzzy (non-fuzzy arguments)** phải giống hệt nhau hoàn toàn (exact match).
3. **Mọi tham số fuzzy (`x-match: "fuzzy"`)** phải đạt điểm $\text{token-F1} > 0.6$ sau khi đã chuyển về chữ thường (lowercasing) và chuẩn hóa bảng mã **Unicode NFC**.

### 3.3. Bốn "Tử Huyệt" từ công thức điểm (Chiến lược sống còn)
1. **Cổng tử thần `state_hash`:** Nếu trạng thái cuối cùng của các app sai ($\text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}}$), toàn bộ điểm của mẫu đó lập tức bằng $0$, bất kể bạn gọi đúng bao nhiêu hàm trước đó. Do đó, các lệnh `write` phải đúng tuyệt đối $100\%$.
2. **Cái giá của việc "Gọi thử sai sửa" (Trial-and-Error Penalty):** Nhiều Agent framework cho phép LLM gọi sai hàm, nhận lỗi từ `env.call()`, rồi gọi lại. Trong cuộc thi này, lệnh gọi lỗi đầu tiên vẫn bị lưu vào log và làm tăng $U$ (số lệnh unmatched), kéo tụt Precision $P = \frac{M}{M+U}$!
   * *Giải pháp:* Phải dùng hàm `env.validate()` (hoặc bộ kiểm tra JSON Schema tự viết bên ngoài) để kiểm tra tính hợp lệ của tham số trước khi thực thi `env.call()`. (Cần kiểm tra mã nguồn `zaic_tools` xem `env.validate` có ghi log không; tự viết validator bằng Python dựa trên `<split>_tools_catalog.json` là an toàn tuyệt đối nhất).
3. **Không gọi thừa lệnh `read`:** Nếu nhãn chuẩn chỉ gọi `smart_home.get_status` mà Agent của ta cẩn thận quá gọi thêm `contacts.list` hoặc `banking.get_balance` không cần thiết, thì dù `state_hash` vẫn đúng, các lệnh `read` thừa đó sẽ biến thành $U$, làm giảm điểm F1 của mẫu!
4. **Chuẩn hóa Unicode NFC:** Tiếng Việt có 2 kiểu dựng dấu (NFC dựng sẵn và NFD tổ hợp). Bắt buộc mọi chuỗi văn bản sinh ra từ ASR và LLM phải đi qua bước `unicodedata.normalize('NFC', text)` trước khi đưa vào `arguments`.

---

## 4. Ràng Buộc Kỹ Thuật & Bài Toán Quy Hoạch Tài Nguyên

Mọi giải pháp phải chạy **Offline $100\%$** trong Docker container trên cấu hình phần cứng cố định của Ban Tổ Chức:

| Hạng mục | Quy định giới hạn | Phân tích & Hệ quả Kỹ thuật |
| :--- | :--- | :--- |
| **Tổng tham số toàn hệ thống** | $< 6\text{B}$ parameters | Tổng kích thước của ASR + LLM + Reranker/Classifier (nếu có) không được vượt quá $6$ tỷ tham số. |
| **Giới hạn LLM** | $< 5\text{B}$ parameters, Context $\le 16,384$ tokens | Không được dùng model $7\text{B}/8\text{B}$. Các ứng cử viên sáng giá nhất thuộc phân khúc $3\text{B} - 4\text{B}$ (VD: họ Qwen 3B/4B, Llama-3.2-3B...). |
| **Phần cứng chấm thi** | $1 \times$ NVIDIA A30 GPU ($24\text{ GB}$ VRAM), CPU 16-core Xeon Gold 6442Y, $64\text{ GB}$ RAM | $24\text{ GB}$ VRAM là khá thoải mái để load đồng thời 1 model ASR ($\approx 0.8\text{B} - 1.5\text{B}$) và 1 model LLM ($\approx 3\text{B} - 4\text{B}$) ở chuẩn `bfloat16`/`fp16` mà không cần lượng tử hóa (quantization) làm giảm độ chính xác. |
| **Thời gian chạy (Latency)** | $\le 30\text{ giây} / \text{mẫu}$ | Mẫu nào quá $30\text{s}$ hoặc văng lỗi (crash) sẽ bị $0$ điểm. Cần tối ưu tốc độ suy luận bằng `vLLM`, `SGLang`, hoặc `CTranslate2`/`faster-whisper`. |
| **Tính tái lập (Reproducibility)** | Cố định toàn bộ Random Seeds, Greedy Decoding / `temperature = 0` | Bắt buộc tắt sampling (`do_sample=False`, `temperature=0.0`). Nếu chạy lại ra kết quả khác, BTC sẽ chạy $5$ lần lấy trung bình. |
| **Tiêu chí phụ (Tie-breaking)** | 1. Điểm F1 bỏ qua `state_hash`<br>2. Thời gian xử lý trung bình thấp hơn | Tốc độ inference nhanh và gọi hàm chuẩn xác ngay cả khi sai state vẫn giúp thắng khi bằng điểm. |

---

## 5. Các Quy Tắc Cấm & Bản Quyền Dữ Liệu (Strict Compliance)

* **TUYỆT ĐỐI KHÔNG đọc trực tiếp `state.json`:** Agent hoặc code giải pháp không được phép đọc, copy hay parse bất kỳ file nào nằm trong thư mục `<split>_environment/`. Mọi tương tác dữ liệu bắt buộc phải đi qua đúng 1 đối tượng `ToolEnv` cho mỗi mẫu. Vi phạm = Loại trực tiếp khi BTC review code.
* **Thời hạn Open-Source Models/Data:** Chỉ được sử dụng các mô hình mã nguồn mở và tập dữ liệu công khai được phát hành **trước ngày 26/10/2026** với giấy phép hợp lệ.
* **Quy định về Sinh dữ liệu nhân tạo (Synthetic Data):**
  * **ĐƯỢC PHÉP:** Dùng các mô hình lớn bên ngoài hoặc dịch vụ API (LLM lớn, TTS...) để sinh thêm dữ liệu huấn luyện (Synthetic Data) trong giai đoạn phát triển (Development/Training).
  * **NGHIÊM CẤM:** Sử dụng bất kỳ dịch vụ bên ngoài, mô hình lớn hay gán nhãn thủ công (manual labeling) trực tiếp lên tập `public_test` hoặc `private_test` để huấn luyện hoặc tinh chỉnh (tune) mô hình.

---

## 6. Lộ Trình & Các Mốc Thời Gian Quan Trọng (Timeline)

1. **Vòng 1 — Public Test (26/10 – 16/11):**
   * Tối đa **$5$ lần nộp bài / ngày** (file `.jsonl`) lên cổng Zalo AI Challenge.
   * **Hạn chót nộp Docker (16/11 lúc 23:59 GMT+7):** Nộp toàn bộ giải pháp đóng gói trong Docker (bao gồm cả training code, inference code và data) qua link Google Drive kèm mã kiểm tra `checksum`. Tối đa $2$ lần nộp, BTC chỉ chấm bản cuối cùng.
2. **Vòng 2 — Private Test (20/11 – 23/11):**
   * **09:00 sáng ngày 20/11:** BTC mở tập `private_test`.
   * Các đội dùng chính bản giải pháp đã đóng băng chạy trên `private_test` và nộp file `.jsonl` trước **23:59 GMT+7 ngày 23/11**.
   * BTC sẽ tự chạy lại Docker image trên hệ thống chấm để xác minh kết quả và kiểm tra mã nguồn chống gian lận.
3. **Công bố kết quả:** Ngày **02/12** (Vinh danh Top 2 đội xuất sắc nhất).

---

## 7. Định Hướng Kiến Trúc & Phân Công Nhiệm Vụ Ngay Cho Team

Để hướng tới vị trí Top đầu, chúng ta không xây dựng một pipeline nối ghép đơn thuần mà phải thiết kế một hệ thống **Chặt chẽ - Có kiểm soát lỗi - Tối ưu hóa theo đúng Metric**:

1. **Nhánh 1: Data Engineering & Môi trường (Ưu tiên số 1 ngay hôm nay)**
   * Đọc toàn bộ mã nguồn của thư viện `zaic_tools` trong tập `train` và `public_test`.
   * Xây dựng **Offline Evaluation Script** tính chính xác $100\%$ công thức điểm ($\text{hash}_{\text{sys}} = \text{hash}_{\text{gold}}$, $P, R$, token-F1 với Unicode NFC) và thiết lập **Local Validation Split** (chia từ $1,600$ mẫu `train`).
   * Phân tích thống kê trên tập `train`: Có bao nhiêu domain/app? Tỷ lệ mẫu Out-of-scope (`[]`) và Missing Info (`sys.ask_missing_info`) là bao nhiêu phần trăm? Các chuỗi gọi hàm phổ biến nhất dài bao nhiêu bước?
2. **Nhánh 2: Khối Nhận dạng Giọng nói (ASR & Post-processing)**
   * Thử nghiệm các mô hình ASR mạnh nhất cho tiếng Việt + từ mượn tiếng Anh (VD: *Whisper-large-v3-turbo*, *PhoWhisper*...) sao cho vừa ngân sách tham số ($< 1.5\text{B}$ để nhường $< 4.5\text{B}$ cho LLM).
   * Đánh giá lỗi ASR trên tập `train` (so sánh với `audio_text` chuẩn), đặc biệt chú ý: tên riêng danh bạ, địa chỉ đường phố, số tiền, thời gian, và từ khóa tiếng Anh (*"AC"*, *"book"*).
3. **Nhánh 3: Khối Agentic Tool-Calling LLM & Guardrails**
   * Lựa chọn Backbone LLM $< 5\text{B}$ có khả năng Function Calling tốt nhất hiện tại (như dòng Qwen $3\text{B}/4\text{B}$).
   * Thiết kế **Pre-call Validator & Schema Guardrail**: Chặn mọi lệnh gọi sai tên hàm, sai kiểu dữ liệu, thiếu trường `required` hoặc vi phạm `x-one-of` / `x-ref` **trước khi** lệnh đó chạm tới `env.call()`.
   * Lên kế hoạch Fine-tuning (SFT / GRPO) LLM trên dữ liệu `train` kết hợp sinh thêm dữ liệu giả lập (Synthetic multi-step trajectories) mô phỏng đúng các tool trong `<split>_tools_catalog.json`.