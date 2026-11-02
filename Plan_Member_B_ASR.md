# KẾ HOẠCH HÀNH ĐỘNG CHI TIẾT — THÀNH VIÊN B
## CHUYÊN GIA ASR & XỬ LÝ TIẾNG NÓI (AUDIO & SPEECH-TO-TEXT SPECIALIST)
> **Dự án:** LifeMate Agent (Zalo AI Challenge 2026)  
> **Phiên bản:** v3.0 (Cập nhật ngày 02/11/2026)  
> **Tài nguyên chỉ định:** Google Colab GPU T4 / Kaggle GPU T4x2  
> **Người giao việc:** Thành viên A (Lead & System Architect)

---

## 1. Mục Tiêu Tối Thượng & Ranh Giới Trách Nhiệm

### 1.1. Mục tiêu cốt lõi
Biến đổi file âm thanh giọng nói tiếng Việt (`.wav`) thành văn bản chuẩn xác. **Lưu ý cốt tử theo bản Review v3:** 
> *"Điểm thi không chấm theo WER (Word Error Rate). Một câu đạt WER 95% nhưng nghe sai số điện thoại, nghe thiếu từ 'đừng/không', hoặc viết sai tên người nhận tiền sẽ khiến toàn bộ chuỗi tool calls sau đó bị 0 điểm."*

Nhiệm vụ của bạn là **bảo toàn 100% các trường thông tin quyết định hành động (Decision-Critical Slots)**.

### 1.2. Ranh giới trách nhiệm (Scope of Work)
* **Những gì B làm chủ (Owner):**
  * Mô hình ASR (Whisper / faster-whisper / PhoWhisper).
  * Bộ trích xuất & chuẩn hóa văn bản ([src/part_b/normalizer.py](file:///D:/code/.vscode/LifeMateAgent/src/part_b/normalizer.py)).
  * Module suy luận âm thanh ([src/part_b/asr_module.py](file:///D:/code/.vscode/LifeMateAgent/src/part_b/asr_module.py)).
  * Benchmark độ trễ và tỷ lệ lỗi slot ([src/part_b/asr_benchmark.py](file:///D:/code/.vscode/LifeMateAgent/src/part_b/asr_benchmark.py)).
  * Notebook chạy trên GPU T4 ([notebooks/01_asr_colab_kaggle.ipynb](file:///D:/code/.vscode/LifeMateAgent/notebooks/01_asr_colab_kaggle.ipynb)).
* **Những gì B KHÔNG làm:**
  * Không can thiệp vào logic gọi tool hay logic ToolEnv của Thành viên C.
  * Không sửa công thức tính điểm hay chia dữ liệu của Thành viên A.

---

## 2. Các Ràng Buộc Kỹ Thuật Bắt Buộc (Hard Constraints)

| Thông số | Giới hạn quy chế | Khuyến nghị cho Thành viên B |
| :--- | :--- | :--- |
| **Thời gian ASR** | Tối đa **5.0 giây/mẫu** | Mục tiêu P95 trên T4: **< 2.5 giây/mẫu** |
| **Mô hình ASR** | Tổng hệ thống < 6B params | Dùng **faster-whisper large-v3-turbo** (~800M params, FP16) |
| **Bảng mã tiếng Việt** | Bắt buộc **Unicode NFC** | Chuẩn hóa `unicodedata.normalize('NFC', text)` trên mọi đầu ra |
| **Môi trường chạy** | Offline, GPU T4 | Sử dụng CTranslate2 runtime trong `faster-whisper` |

---

## 3. Bốn "Tử Huyệt" ASR Cần Xử Lý Triệt Để

Dựa trên phân tích 1.545 mẫu tập `train` (95.9% mẫu có Code-Switching Việt-Anh):

### Tử huyệt 1: Mất số 0 ở đầu số điện thoại
* **Nguy cơ:** Khi nghe người dùng đọc *"0912 345 678"*, Whisper thường phiên âm thành dạng số *"912345678"*. Khi đưa vào tool `contacts.search` hoặc `phone.call`, tool sẽ không tìm thấy hoặc báo lỗi schema.
* **Quy tắc xử lý:** Bất kỳ chuỗi số nào có 9-10 chữ số biểu thị số điện thoại Việt Nam **phải được giữ nguyên số 0 đầu**.

### Tử huyệt 2: Bỏ sót từ phủ định sống còn
* **Nguy cơ:** *"Đừng bật đèn phòng khách"*, *"Hủy hẹn lúc 9 giờ sáng mai"*, *"Không chuyển khoản cho Nam nữa"*. Nếu Whisper nghe sót từ *"Đừng"* hoặc *"Không"*, Agent sẽ thực hiện hành động ngược lại $\rightarrow$ Sai State $\rightarrow$ **0 điểm toàn bộ mẫu**.
* **Quy tắc xử lý:** Kiểm tra độ nhạy âm học (Acoustic sensitivity) của các từ phủ định: `không`, `đừng`, `chưa`, `hủy`, `thôi`, `ngưng`.

### Tử huyệt 3: Sai lệch số tiền & đơn vị giao dịch ngân hàng
* **Nguy cơ:** *"Năm mươi nghìn"* $\rightarrow$ `50000`, *"Hai triệu rưỡi"* $\rightarrow$ `2500000`, *"Năm lốp"* $\rightarrow$ `500000`.
* **Quy tắc xử lý:** Xây dựng parser chuẩn hóa số tiền bằng regex / lookup table trong `normalizer.py`.

### Tử huyệt 4: Nhầm lẫn thực thể tiếng Anh (Code-Switching 95.9%)
* **Nguy cơ:** Người dùng nói *"check mail"*, *"set alarm"*, *"book xe"*, *"call cho sếp"*. Whisper tiếng Việt gốc dễ phiên âm sai thành từ vô nghĩa.
* **Quy tắc xử lý:** Dùng **Contextual Prompt Biasing** (tiêm danh sách 46 app và 147 tool vào tham số `initial_prompt` của Whisper).

---

## 4. Hợp Đồng Giao Tiếp API (API Contract với Thành viên A & C)

Thành viên B có nhiệm vụ cung cấp đúng interface hàm sau trong file `src/part_b/asr_module.py`:

```python
def transcribe_audio(
    audio_path: str | Path,
    context: dict | None = None,
    initial_prompt: str | None = None
) -> dict[str, str]:
    """Hàm chuẩn ASR của Thành viên B.
    
    Args:
        audio_path: Đường dẫn tới file audio .wav của mẫu
        context: Dữ liệu context của mẫu (chứa installed_apps, user_profile, current_time)
        initial_prompt: Prompt gợi ý từ khóa (tùy chọn)
        
    Returns:
        {
            "raw_transcript": str,        # Văn bản gốc chưa chỉnh sửa
            "normalized_transcript": str  # Văn bản chuẩn NFC, đã xử lý số tiền, phone, phủ định
        }
    """
```

---

## 5. Quy Trình 5 Bước Thực Thi Trên Colab T4 / Kaggle

### Bước 1: Mở Notebook và kiểm tra GPU
1. Mở file [notebooks/01_asr_colab_kaggle.ipynb](file:///D:/code/.vscode/LifeMateAgent/notebooks/01_asr_colab_kaggle.ipynb).
2. Kiểm tra GPU: Chạy `!nvidia-smi` (đảm bảo hiển thị GPU Tesla T4).
3. Cài đặt thư viện: `pip install -q faster-whisper jiwer librosa soundfile`.

### Bước 2: Thiết lập Prompt Biasing
Khởi tạo prompt gợi ý từ khóa theo ngữ cảnh của mẫu:
```python
APPS_VOCABULARY = (
    "LifeMate, smart_home, light_set, get_status, bật đèn, tắt đèn, "
    "contacts, search, phone, call, banking, transfer, chuyển tiền, "
    "calendar, create_event, nhắc nhở, reminders, alarm, báo thức, "
    "không, đừng, hủy, dời lịch, số điện thoại 09, 08, 03, 07, 05."
)
```

### Bước 3: Benchmark trên 50 mẫu Dev
Chạy thử nghiệm trên 50 mẫu đầu tiên của tập `dev` để đo 3 chỉ số:
1. **P95 Latency**: Thời gian xử lý mỗi file audio (mục tiêu: < 2.5s).
2. **WER / CER**: Đo bằng `jiwer`.
3. **Slot Accuracy**: Kiểm tra thủ công 10 ca có số điện thoại và số tiền.

### Bước 4: Hoàn thiện Normalizer
Cập nhật các hàm chuẩn hóa trong [src/part_b/normalizer.py](file:///D:/code/.vscode/LifeMateAgent/src/part_b/normalizer.py):
* Chuẩn hóa Unicode NFC.
* Chuẩn hóa từ chỉ thời gian: *"mai"*, *"ngày kia"*, *"chiều nay"*.
* Bảo toàn số 0 đầu trong số điện thoại.

### Bước 5: Bàn giao kết quả
1. Chạy xuất file toàn bộ transcript cho tập Dev (265 mẫu): `outputs/transcripts_dev.json`.
2. Gửi file JSON này cho Thành viên A và C để chạy so sánh điểm end-to-end.

---

## 6. Tiêu Chí Nghiệm Thu (Definition of Done — DoD)

* [ ] Mô hình chạy hoàn toàn offline trên GPU T4 với thời gian trung bình $\le 2.5$ giây/mẫu.
* [ ] 100% văn bản đầu ra đạt chuẩn **Unicode NFC**.
* [ ] Không làm mất số 0 đầu ở các chuỗi số điện thoại.
* [ ] Nhận diện chính xác 100% các từ phủ định (`đừng`, `không`, `hủy`) trong tập kiểm tra.
* [ ] File `transcripts_dev.json` được tạo thành công cho 265 mẫu tập dev.
