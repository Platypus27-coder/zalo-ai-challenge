# KẾ HOẠCH HÀNH ĐỘNG CHI TIẾT — THÀNH VIÊN C
## CHUYÊN GIA AGENT LLM & ĐIỀU PHỐI CÔNG CỤ (AGENT LLM & TOOL-CALLING SPECIALIST)
> **Dự án:** LifeMate Agent (Zalo AI Challenge 2026)  
> **Phiên bản:** v3.0 (Cập nhật ngày 02/11/2026)  
> **Tài nguyên chỉ định:** Kaggle GPU T4x2 / Google Colab GPU T4  
> **Người giao việc:** Thành viên A (Lead & System Architect)

---

## 1. Mục Tiêu Tối Thượng & Ranh Giới Trách Nhiệm

### 1.1. Mục tiêu cốt lõi
Xây dựng bộ não LLM điều khiển chu trình gọi công cụ tuần hoàn (Sequential Tool-Calling Loop) để chuyển đổi yêu cầu người dùng thành danh sách các lệnh gọi hàm chính xác tuyệt đối.

Phân loại chuẩn xác 3 kịch bản theo yêu cầu đề bài:
1. **Thực thi (Execute - chiếm 78.8%):** Gọi chuỗi lệnh `read` $\rightarrow$ lấy ID $\rightarrow$ gọi lệnh `write`.
2. **Hỏi lại (Ask - chiếm 14.0%):** Khi thiếu thông tin bắt buộc không thể tìm thấy trong hệ thống, gọi đúng một lệnh `sys.ask_missing_info`.
3. **Từ chối (Out-of-Scope / OOS - chiếm 7.2%):** Khi yêu cầu thuộc ứng dụng chưa cài đặt, **trả về mảng rỗng `[]` ngay lập tức** (được 1 điểm trọn vẹn).

### 1.2. Ranh giới trách nhiệm (Scope of Work)
* **Những gì C làm chủ (Owner):**
  * Lựa chọn & triển khai LLM (<5B params, ví dụ: `Qwen/Qwen2.5-3B-Instruct` hoặc `7B` 4-bit AWQ).
  * Vòng lặp suy luận gọi tool tuần tự ([src/part_c/agent.py](file:///D:/code/.vscode/LifeMateAgent/src/part_c/agent.py)).
  * Kỹ thuật Prompting & System Prompt tiếng Việt ([src/part_c/prompt.py](file:///D:/code/.vscode/LifeMateAgent/src/part_c/prompt.py)).
  * Cơ chế lọc công cụ theo ứng dụng cài đặt ([src/part_c/tool_filter.py](file:///D:/code/.vscode/LifeMateAgent/src/part_c/tool_filter.py)).
  * Sổ bằng chứng & Khử ID ảo ([src/part_c/evidence_ledger.py](file:///D:/code/.vscode/LifeMateAgent/src/part_c/evidence_ledger.py)).
  * Theo dõi tiến độ công việc đa bước ([src/part_c/task_tracker.py](file:///D:/code/.vscode/LifeMateAgent/src/part_c/task_tracker.py)).
  * Notebook thực nghiệm trên GPU ([notebooks/02_agent_colab_kaggle.ipynb](file:///D:/code/.vscode/LifeMateAgent/notebooks/02_agent_colab_kaggle.ipynb)).
* **Những gì C KHÔNG làm:**
  * Không can thiệp vào mô hình ASR (Thành viên B phụ trách).
  * Không sửa đổi code ToolEnv / State / Evaluator (Thành viên A phụ trách).

---

## 2. Các Ràng Buộc Kỹ Thuật Bắt Buộc (Hard Constraints)

| Thông số | Giới hạn quy chế | Khuyến nghị cho Thành viên C |
| :--- | :--- | :--- |
| **Kích thước LLM** | Bắt buộc **< 5.0 tỷ tham số** (5B) | Dùng **Qwen2.5-3B-Instruct** (hoặc Qwen2.5-7B nếu BTC cho phép quantize 4-bit) |
| **Context Window** | Tối đa **16.384 tokens** | Prompt chỉ nạp tool của installed_apps (thường < 2.000 tokens) |
| **Thời gian LLM** | Tối đa **16.0 giây/mẫu** | Mục tiêu P95: **< 10.0 giây/mẫu** (dùng vLLM hoặc Greedy Decoding) |
| **Số bước gọi tool** | Tối đa **10 bước/mẫu** | Dữ liệu train: Min=0, Max=6, Avg=2.3 bước. Dừng ngay khi xong |
| **Tính tất định** | Bắt buộc tái lập được 100% | Đặt `temperature=0.0`, `do_sample=False`, `seed=42` |

---

## 3. Ba Vũ Khí Chiến Lược Cần Nắm Vững (Từ Bản Upgrade v3)

### Vũ khí 1: Lọc Tool Động theo Ngữ Cảnh (Tool Filtering per Sample)
* **Quy tắc:** Đề thi có 46 app và 147 tools. Tuyệt đối **KHÔNG ĐƯỢC nạp cả 147 tools** vào prompt!
* **Cách thực hiện:** Dùng `src/part_c/tool_filter.py`: Chỉ trích xuất các tool nằm trong `context.installed_apps` của mẫu đó + công cụ `sys.ask_missing_info`.
* **Hiệu quả:** Giảm 85% lượng token tiêu thụ, tăng tốc độ sinh lời, và ngăn chặn 100% lỗi gọi nhầm tool của app chưa cài.

### Vũ khí 2: Sổ Bằng Chứng Khử ID Ảo (Evidence Ledger & Entity Resolver - §5.1)
* **Vấn đề tử huyệt:** LLM thường có xu hướng "tự bịa" ID (ví dụ: bịa `device_id: "light_1"` khi chưa gọi `smart_home.get_status` để xem ID thật). Nếu gọi write với ID sai, state sẽ sai $\rightarrow$ **0 điểm**.
* **Cách thực hiện:** Sử dụng class `EvidenceLedger` ([src/part_c/evidence_ledger.py](file:///D:/code/.vscode/LifeMateAgent/src/part_c/evidence_ledger.py)):
  1. Khi nhận kết quả từ các lệnh đọc (`contacts.search`, `calendar.list_events`), nạp các thực thể thật vào Sổ.
  2. Phân giải `canonical_id` từ Sổ thay vì để LLM tự điền ID.
  3. **Quy tắc Invalidation:** Sau bất kỳ lệnh `write` nào, tự động đánh dấu các thực thể liên quan là `stale` (hết hạn), buộc Agent phải đọc lại nếu cần cập nhật tiếp.

### Vũ khí 3: Quản Lý Phụ Thuộc Đa Bước (Task State Tracker - §5.4)
* **Vấn đề:** Với các câu ghép phức tạp (*"Tắt bình nóng lạnh rồi hẹn giờ giặt lúc 10h"*):
  * LLM hay bị dừng sớm sau bước 1 hoặc lặp lại bước 1 nhiều lần.
* **Cách thực hiện:** Sử dụng `TaskTracker` ([src/part_c/task_tracker.py](file:///D:/code/.vscode/LifeMateAgent/src/part_c/task_tracker.py)) để theo dõi trạng thái `pending` $\rightarrow$ `blocked` $\rightarrow$ `done`, unblock bước tiếp theo khi bước trước có kết quả thành công.

---

## 4. Hợp Đồng Giao Tiếp API (API Contract với Thành viên A)

Thành viên C có nhiệm vụ cung cấp class `Agent` chuẩn trong file `src/part_c/agent.py`:

```python
class LifeMateAgent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", mode: str = "transformers"):
        """Khởi tạo mô hình và bộ lọc tool."""
        pass

    def run_sample(
        self,
        transcript: str,
        sample: dict,
        env: Any,
        validator: Any
    ) -> list[dict]:
        """Thực thi chu trình gọi tool cho một mẫu.
        
        Args:
            transcript: Câu yêu cầu của người dùng (từ Thành viên B)
            sample: Dữ liệu mẫu (context, history, installed_apps)
            env: Đối tượng ToolEnv (do Thành viên A cung cấp)
            validator: Bộ kiểm tra schema pre-call
            
        Returns:
            list[dict]: Danh sách tool calls đã thực hiện, ví dụ:
            [
                {"name": "contacts.search", "arguments": {"query": "Mẹ"}},
                {"name": "phone.call", "arguments": {"contact_id": "c_001"}}
            ]
        """
```

---

## 5. Quy Trình 5 Bước Thực Thi Trên Kaggle GPU T4x2 / Colab

### Bước 1: Mở Notebook và Tải Model
1. Mở file [notebooks/02_agent_colab_kaggle.ipynb](file:///D:/code/.vscode/LifeMateAgent/notebooks/02_agent_colab_kaggle.ipynb).
2. Tải mô hình `Qwen/Qwen2.5-3B-Instruct` với `torch.bfloat16` và `device_map="auto"`.
3. Kiểm tra VRAM: Mô hình 3B chỉ chiếm ~6GB VRAM, hoàn toàn chạy mượt mà trên T4 (16GB VRAM).

### Bước 2: Thiết Lập Vòng Lặp Tuần Tự (Sequential Tool Loop)
1. Bước 1: LLM đọc `transcript` + `installed_apps`.
2. Nếu không có app nào phù hợp $\rightarrow$ dừng ngay, trả về `[]` (OOS).
3. Nếu cần ID $\rightarrow$ sinh lệnh gọi hàm `read` $\rightarrow$ nạp kết quả vào `EvidenceLedger`.
4. Nếu đủ ID và dữ liệu $\rightarrow$ sinh lệnh gọi hàm `write` $\rightarrow$ kiểm tra bằng `validator` trước khi gửi vào `env.call()`.
5. Sau khi write thành công $\rightarrow$ dừng chu trình (DONE).

### Bước 3: Đấu Nối Validator Trước Khi Call
Trước mỗi lần `env.call(name, args)`, bắt buộc gọi `validator.validate_call(name, args, tool_specs)`:
* Nếu thiếu tham số required hoặc sai type: yêu cầu LLM sửa lại tại chỗ thay vì gửi lệnh lỗi vào ToolEnv (tránh bị trừ điểm $U$ và hỏng state).

### Bước 4: Chạy Thử Nghiệm Trên 50 Mẫu Dev
Chạy thử nghiệm Agent với transcript chuẩn (`gold transcript`) trên 50 mẫu tập Dev:
1. Ghi nhận thời gian trung bình mỗi bước (mục tiêu: < 1.5s / bước).
2. Đánh giá tỷ lệ State Hash match bằng evaluator của Thành viên A.

### Bước 5: Bàn Giao File Candidate
1. Xuất file kết quả chạy trên toàn bộ 265 mẫu tập Dev: `outputs/submissions/candidate_c_dev.jsonl`.
2. Bàn giao cho Thành viên A để đưa vào `src/part_a/loss_attribution.py` phân tích điểm mất và tỷ lệ ROI.

---

## 6. Tiêu Chí Nghiệm Thu (Definition of Done — DoD)

* [ ] Mô hình LLM có số lượng tham số thực tế **< 5.0B** (đáp ứng 100% quy chế thi).
* [ ] Tốc độ xử lý trung bình $\le 10.0$ giây/mẫu trên GPU T4.
* [ ] 100% các mẫu OOS (Out-of-Scope) trả về danh sách rỗng `[]` (không bị gọi nhầm tool).
* [ ] Không tự bịa ID (`device_id`, `contact_id`, `event_id`); mọi ID trong lệnh `write` phải được bảo chứng từ Sổ bằng chứng (`EvidenceLedger`).
* [ ] Chạy thành công 265 mẫu tập Dev và xuất file `candidate_c_dev.jsonl`.
