# LifeMate — Đánh giá kế hoạch và phân công đội 3 người

Ngày rà soát: 01/11/2026. Phiên bản: v2 — review lần hai. Múi giờ kế hoạch: GMT+7.

Tài liệu đầu vào: “Tài Liệu Phân Tích Đề Bài & Chiến Lược LifeMate.md”. Đây là bản đề xuất mới để đội thảo luận và triển khai; không sửa bản gốc, không phải bản thể lệ của BTC.

## 1. Kết luận đánh giá

**Hướng tiếp cận của tài liệu gốc đúng, nhưng bản gốc thiên về diễn giải đề hơn là kế hoạch thực thi.** Điểm tốt nhất là chú ý đồng thời trạng thái cuối và lịch sử tool call, nhận diện ảnh hưởng của call thừa/call lỗi, và tách ASR khỏi Agent. Nên giữ ba ý này.

Những thiếu sót của bản gốc được bản kế hoạch này xử lý:

1. Nguồn kiểm chứng cho từng quy định, đặc biệt metric, giới hạn model và giao thức chạy Docker.
2. Baseline chạy xuyên suốt, split validation cố định và cách tách lỗi ASR khỏi lỗi Agent.
3. Hợp đồng dữ liệu giữa ba người để không đợi nhau hoàn thành mới tích hợp.
4. Tiêu chí chọn model, tiêu chí dừng thử nghiệm, ngân sách thời gian theo mẫu.
5. Owner, đầu ra và điều kiện nghiệm thu cho từng giai đoạn.

**Đề xuất:** làm pipeline ASR → Agent tuần tự có validator → ToolEnv; tích hợp sớm; ưu tiên sửa lỗi có ảnh hưởng lớn đến điểm local. Chỉ SFT khi đã có baseline và dữ liệu trajectory đúng. Không đặt GRPO trên đường găng.

## 2. Mức độ xác minh và giả định

Đã đọc toàn bộ tài liệu đính kèm và đối chiếu nội dung truy xuất được từ trang chính thức. Chưa có bộ dữ liệu, mã `zaic_tools`, evaluator, máy chạy hay repository của đội, nên chưa thể chứng thực metric hoặc benchmark hiệu năng.

### 2.1. Đối chiếu nguồn

| Nội dung | Trạng thái | Hành động |
|---|---|---|
| Bài toán giọng nói → hành động, có nhánh hỏi thiếu thông tin và ngoài phạm vi | Phù hợp mô tả chính thức [1] | Giữ |
| “100% giọng người thật” | Mâu thuẫn mô tả chính thức có giọng tổng hợp và thật [1] | Bỏ khẳng định 100%; kiểm kê dữ liệu thực tế |
| Audio chứa cả lượt mới và các lượt lịch sử | Trang mô tả audio yêu cầu mới nhất, history riêng [1] | Không mặc định nối history vào audio; kiểm tra loader và mẫu |
| Deadline giải pháp 16/11, private 20–23/11 | Phù hợp lịch trang chủ [2] | Lập lịch ngược từ 16/11, kiểm tra thông báo cập nhật |
| 02/12 là công bố kết quả cuối cùng | Trang chủ phân biệt finalists 02/12 và final standing 12/12 [2] | Sửa tên mốc |
| Số mẫu, codec, fuzzy threshold, cách matching, hash, schema mở rộng | Chỉ có trong tài liệu đội; chưa đối chiếu evaluator | A xác minh từ kit chính thức |
| Giới hạn tham số, GPU, 30 giây, cutoff 26/10, synthetic, số lượt nộp và thủ tục Docker | Chưa xác minh được toàn bộ tab Rule/Submission | A ghi nguồn và phiên bản trước khi chốt model |

Nguồn tra cứu ngày 01/11/2026:

- [1] https://challenge.zalo.ai/portal/lifemate — phần mô tả truy xuất được; chưa đọc được đầy đủ các tab động.
- [2] https://challenge.zalo.ai/ — lịch thi chính thức.

### 2.2. Giả định để lập lịch

- Đội có 3 người; chưa biết tên, sở trường hay số giờ rảnh nên dùng A/B/C theo vai trò.
- Có thể làm trung bình khoảng 4–6 giờ tập trung/người/ngày từ 02–14/11, có GPU phục vụ inference và khả năng chạy thử môi trường gần máy chấm. Đây là giả định năng lực, không phải cam kết thời lượng.
- Nếu thiếu GPU/giờ làm: giữ baseline, validator, evaluation và đóng gói; cắt SFT, synthetic diện rộng và mọi thử nghiệm RL trước.
- Mọi giới hạn số học lấy từ bản gốc trong các phần sau đều là **tạm thời, chờ xác minh**. Không dùng tên model hay kích thước quảng cáo làm bằng chứng tuân thủ.

## 3. Những điểm cần chỉnh về kỹ thuật

| Điểm trong bản gốc | Vấn đề | Cách sửa |
|---|---|---|
| `x-ref` nghĩa là luôn phải read trước | Trộn ràng buộc ID hợp lệ với cách tìm ID | Chỉ read khi thiếu bằng chứng hợp lệ. Có thể dùng ID đã nhận từ kết quả tool trong mẫu, hoặc input/history nếu kit cho phép và còn hợp lệ. Không tạo ID từ suy đoán hay nhớ từ train |
| Validator tự viết là “an toàn tuyệt đối” | Schema đúng không chứng minh ID tồn tại, đủ số dư hay hành động đúng ý | Tách kiểm tra cú pháp, extension, provenance và logic nghiệp vụ. Validator chỉ giảm một lớp lỗi |
| Mọi chuỗi đều chuẩn hóa NFC | Có thể thay đổi opaque ID hoặc giá trị exact-match do tool trả | NFC cho văn bản ngôn ngữ phù hợp evaluator; giữ nguyên ID/giá trị canonical từ tool |
| Write phải đúng 100% | Mục tiêu đúng nhưng không phải tiêu chí có thể bảo đảm trước khi đo | Báo cáo hash match rate, nguyên nhân sai và điều kiện pre-call; không quảng cáo bảo đảm tuyệt đối |
| Giảm số call tối đa | Read cần thiết có thể cứu toàn bộ mẫu khỏi sai state | Giảm call thừa sau khi đã bảo đảm grounding; không bỏ read chỉ để làm trace ngắn |
| 24 GB là “khá thoải mái” | Bỏ qua KV cache, activation, workspace và nhiều runtime cùng giữ bộ nhớ | Đo peak VRAM với ASR + LLM đồng thời, prompt dài và nhiều bước; để dư bộ nhớ |
| Không quantize để tránh giảm chính xác | Đây là lựa chọn cần benchmark | So sánh score, latency, VRAM; chỉ quantize khi đo được lợi ích phù hợp |
| `temperature=0` bảo đảm tái lập | Greedy không loại trừ mọi khác biệt kernel/runtime | Pin phiên bản, seed, config; chạy lặp và so trace/hash |
| SFT / GRPO ngay từ đầu | Nguy cơ dành phần lớn thời gian cho training khi chưa hiểu lỗi | Baseline → phân loại lỗi → SFT có kiểm soát; GRPO ngoài phạm vi chính |
| Thiếu chính sách ask / stop | Dễ hỏi quá sớm hoặc gọi lặp | Chỉ hỏi sau khi kiểm tra nguồn thông tin được phép; dừng khi hoàn tất, hỏi thiếu hoặc ngoài phạm vi theo protocol |

### 3.1. Hiểu metric có điều kiện

Nếu công thức của bản gốc được evaluator xác nhận, đặt K = M + U là số predicted calls, G là số gold calls. Với G > 0, hash đúng và M > 0:

`F1 = 2M / (K + G)`.

Ví dụ: G = 3, khớp đủ 3 nhưng có thêm 1 read không khớp thì điểm là 6/7 ≈ 0,857; nếu sai hash thì về 0. Vì vậy ưu tiên trạng thái đúng nhưng không bỏ qua call precision. “Thêm read” không tự động sai: nó có thể cần để tìm đúng thực thể.

Không suy ra thứ tự thực thi không quan trọng từ việc matching bỏ qua thứ tự. Read → write, phụ thuộc ID và trạng thái sau từng write vẫn cần đúng thứ tự.

Nếu phải tái hiện evaluator, kiểm tra matching một-một với call trùng và nhiều ứng viên fuzzy; không tự dùng greedy matching rồi coi tương đương. Xác minh cả `> 0.6` hay `>= 0.6`, tokenization, trường tùy chọn, `null`, kiểu số, thứ tự mảng và cách xử lý call lỗi. Ưu tiên dùng evaluator chính thức làm nguồn chuẩn.

## 4. Phạm vi và cách chọn kiến trúc

| Hướng | Ưu điểm | Chi phí/rủi ro | Quyết định đề xuất |
|---|---|---|---|
| Prompt baseline + vòng lặp tool tuần tự + validator | Dễ debug, nhanh có số đo | Có thể chưa giỏi chuỗi dài | Làm trước |
| Cùng kiến trúc, thêm SFT trajectory | Cải thiện format và hành vi đặc thù | Cần dữ liệu sạch, compute và chống leakage | Chỉ làm khi baseline cho thấy cần |
| RL/GRPO, nhiều agent, search nhiều nhánh | Có thể tối ưu policy sâu hơn | Phức tạp reward, latency, state, tham số và đóng gói | Hoãn sau baseline; không là hạng mục bắt buộc |

Chưa chốt backbone cụ thể. B chọn tối đa 2 checkpoint ASR, C chọn tối đa 2 checkpoint LLM cho vòng sàng lọc đầu. A xác minh license, ngày phát hành, số tham số thực và tính hợp lệ trước khi chạy tốn compute. Các tên trong bản gốc là ứng viên, không phải kết luận model tốt nhất.

### 4.1. Luồng suy luận đề xuất

1. Loader đọc input được phép; tạo đúng một ToolEnv cho mẫu theo kit.
2. B chuyển audio thành transcript; bảo toàn lượt nói và trường history theo định dạng thật.
3. A lọc tool bằng installed apps và schema chính thức; giữ ngoại lệ `sys.ask_missing_info` nếu protocol chính thức cho phép. Không làm classifier riêng ở baseline.
4. C tạo hành động kế tiếp từ request, context, history và tool observations đã có.
5. A kiểm tra proposal trước khi gọi thật. Nếu sai format, trả lỗi nội bộ cho C sửa trong ngân sách; không gọi ToolEnv để thử schema.
6. A thực thi một call; trả observation thật cho C, cập nhật kho bằng chứng và ngân sách.
7. Lặp đến khi hoàn thành hoặc cần hỏi. Runner xuất trace/hash trực tiếp từ cùng instance ToolEnv theo format chính thức.

Không sinh toàn bộ chuỗi có ID phụ thuộc read rồi thực thi mù. Có thể lập kế hoạch nội bộ, nhưng bind tham số phụ thuộc sau khi có observation. Không cần xuất chain-of-thought dài để chạy hệ thống.

### 4.2. Grounding, missing information và mutation

- Lưu nguồn của tham số quan trọng: transcript, context, history hay tool result. ID trả từ tool phải giữ nguyên.
- Thời gian tương đối dựa vào `current_time` và `timezone` của mẫu, không lấy đồng hồ máy. Phân biệt giờ trong ngày, ngày mai và ngày cụ thể.
- Số tiền, số điện thoại, địa chỉ, phủ định và số lượng là nhóm cần kiểm tra riêng; không sửa ASR bằng phỏng đoán.
- Nếu nhiều thực thể cùng tên, dùng quy tắc/nguồn thông tin mà kit cho phép; không chọn đại kết quả đầu.
- `sys.ask_missing_info` chỉ chứa tên tool và params theo schema. Phải kiểm tra cách biểu diễn mơ hồ và thiếu thông tin trong train/kit; không tự mở rộng schema.
- Không coi lỗi runtime, model mất tự tin hoặc hết giờ là bằng chứng ngoài phạm vi. `[]` là một quyết định ngữ nghĩa, không là fallback bảo đảm điểm.
- Không tự đặt chính sách “hỏi trước mọi write”: đề có thể yêu cầu hoàn tất phần độc lập trước khi hỏi phần còn thiếu. Xác nhận hành vi qua train/kit.
- Cache observation trong mẫu; sau write phải đánh dấu dữ liệu liên quan có thể cũ. Không tái dùng state giữa các mẫu.
- Không retry write chỉ vì chưa thấy phản hồi; xác định call đã được thực thi/ghi log hay chưa. Không assume rollback, idempotency hay dry-run của ToolEnv tồn tại.

### 4.3. Ngân sách runtime sơ bộ

Nếu giới hạn 30 giây/mẫu được xác nhận, đặt mục tiêu nội bộ p95 ≤ 25 giây và đo cả max/timeout. p95 tốt không thay cho việc kiểm tra các mẫu dài.

| Phần | Ngân sách khởi đầu | Ghi chú |
|---|---:|---|
| ASR | 5 giây | Đo theo độ dài audio, không áp dụng cứng cho mọi mẫu |
| Tổng các lượt LLM | 16 giây | Giới hạn output token và lượt sửa format |
| Tool + validation + serialization | 2 giây | Đo thực tế |
| Dự phòng | 7 giây | Không dùng hết cho decoding mặc định |

Đây là phân bổ mục tiêu chưa benchmark. Xác minh warm-up, model loading và thời điểm bắt đầu tính giờ. Giới hạn bước Agent phải dựa vào phân phối độ dài trajectory train; không tùy tiện cắt mọi mẫu ở một con số nhỏ. Khi vượt ngân sách, lưu lỗi chẩn đoán, xuất đúng protocol nếu còn có thể; không giả trace/hash thành công.

## 5. Dữ liệu và đánh giá

### 5.1. Split và ngăn leakage

- A tạo train/dev/holdout khoảng 70/15/15 nếu số lượng thực tế cho phép. Tỷ lệ là đề xuất; ưu tiên đủ nhóm hiếm hơn cố giữ số tuyệt đối.
- Group theo session/conversation và các mẫu gần trùng/template nếu xác định được. Không random từng dòng khi chúng dùng chung kịch bản.
- Cố gắng cân bằng app, execute/ask/OOS, độ dài trace, history và code-switching; chỉ phân tầng giọng/vùng nếu có metadata đáng tin.
- Khóa danh sách ID và seed. Dev dùng chọn cấu hình; holdout dùng một lần để đánh giá ứng viên cuối. Nếu dùng lỗi holdout để sửa model/prompt, đánh dấu bộ đó đã trở thành dev và không báo kết quả tiếp theo như đánh giá độc lập. Báo số mẫu theo lát cắt để tránh diễn giải quá mức nhóm nhỏ.
- Ví dụ few-shot, từ điển học từ dữ liệu, SFT và synthetic chỉ lấy từ training portion. Được dùng điểm và phân tích lỗi dev để chọn prompt/cấu hình; không chép đáp án hoặc trajectory dev thành ví dụ trong prompt, dữ liệu huấn luyện hay luật theo ID mẫu. Không tạo paraphrase/synthetic từ dev/holdout rồi quay lại train.
- Public/private chỉ dùng theo quyền mà BTC cho phép; không gán nhãn hay đưa sample test qua dịch vụ ngoài để tạo dữ liệu. Giới hạn lượt nộp không phải lý do để thử đủ mọi lượt mỗi ngày.

### 5.2. Ba chế độ benchmark cần có

| Chế độ | Đầu vào | Mục đích |
|---|---|---|
| Agent với transcript chuẩn | `audio_text` của validation + context/history | Chẩn đoán lỗi Agent khi giảm ảnh hưởng ASR |
| ASR độc lập | Audio validation | WER/CER, lỗi thực thể/số/thời gian/phủ định, latency |
| End-to-end | Audio → ASR → Agent → ToolEnv | Chỉ số chính để chọn cấu hình cuối |

Chênh lệch giữa transcript chuẩn và ASR là tín hiệu định hướng, không là phân rã nhân quả tuyệt đối; kiểm tra từng cặp lỗi. Agent tốt trên transcript chuẩn nhưng kém end-to-end → ưu tiên B; cả hai đều kém → ưu tiên C hoặc lỗi môi trường/evaluation của A.

### 5.3. Báo cáo mỗi experiment

Ghi: run ID, commit, split hash, model revision, prompt/config hash, seed, runtime, score chính thức, hash match rate, call F1, tỷ lệ OOS/ask đúng, lỗi schema/tool, số call thừa, latency p50/p95/max, timeout, peak VRAM. Lát cắt theo app, multi-step và history.

Phân loại lỗi: ASR → hiểu ý → chọn tool → thiếu tham số/ask → grounding ID → thời gian/số → thứ tự/stale state → validator/runtime → format submission. Một mẫu có thể có nhiều nhãn; ghi nguyên nhân đầu tiên và hậu quả, tránh cộng chồng thành tỷ lệ giả.

### 5.4. Huấn luyện nếu thực sự cần

- SFT theo từng bước: input chỉ gồm thông tin hợp lệ có trước bước đó, target là call kế tiếp hoặc kết thúc theo protocol.
- Không đưa future tool results, gold final state/hash hay tool call tương lai vào input model.
- Synthetic chỉ phát sinh từ phần train được phép; lưu nguồn sinh, license và lineage. Lọc schema và kiểm tra thực thi bằng cơ chế được phép trước khi dùng.
- Không tạo nhiều biến thể của một trajectory rồi phân tán sang validation.
- Adapter chỉ được chọn khi cải thiện trên validation, không phá runtime/VRAM và không gây regression rõ ở ask/OOS. Không mặc định SFT luôn thắng prompting.

## 6. Phân công cụ thể cho 3 thành viên

Người phù hợp A: Python/testing/Docker; B: speech/data analysis; C: LLM/tool use/training. Có thể đổi người nhưng giữ trách nhiệm rõ. Không mặc định người dùng phải là A hay C.

| Thành viên | Chịu trách nhiệm chính | Deliverable | Người review |
|---|---|---|---|
| **A — Evaluation, execution, integration** | Xác minh kit/rules; split; evaluator; schema validator; ToolEnv adapter; runner; Docker và release | Rules checklist, split IDs, evaluation report, executor API, container chạy offline | C review semantics validator; B review input và đóng gói |
| **B — ASR, input, error analysis** | Audio loader; ASR shortlist; normalization có bảo toàn; ASR metrics; thống kê data và regression cases | ASR module, transcript cache cho local experiment, benchmark, danh sách lỗi quan trọng | A review split/cache; C review tác động tool arguments |
| **C — Agent, reasoning policy, SFT** | LLM shortlist; prompt; bước kế tiếp; ask/OOS; grounding policy; SFT có điều kiện | Agent module, cấu hình model/prompt, benchmark gold-text và end-to-end, checkpoint nếu có | A review tool contract; B review lỗi từ transcript |

A có nhiều việc tích hợp nên **không kiêm toàn bộ EDA và chuẩn bị release**: B sở hữu thống kê, regression set và tài liệu model/data; C chịu phần model packaging/config và golden Agent tests. Cả ba tự viết kiểm thử cho module của mình, A chỉ tổ chức chạy chung.

### 6.1. Giao diện phải chốt ngay ngày đầu

Các tên dưới đây là API nội bộ đề xuất, không phải API được xác nhận của BTC:

| Hợp đồng | Producer → Consumer | Nội dung |
|---|---|---|
| `SampleInput` | A/B → C | ID, audio references, context, history, allowed tool catalog; không chứa nhãn chỉ dành cho training |
| `ASRResult` | B → C | raw text, normalized text, thời gian chạy, lỗi; confidence là optional nếu model hỗ trợ |
| `AgentAction` | C → A | Một proposal `{name, arguments}` hoặc tín hiệu nội bộ `stop`; stop không là tool call |
| `ValidationResult` | A → C | accepted hoặc danh sách lỗi local; không mutate env |
| `Observation` | A → C | Kết quả tool thật và trạng thái call; không bịa observation |
| `SampleOutput` | A → evaluator | ID, tool calls và state hash lấy từ env, theo schema nộp chính thức |

C làm việc được với transcript chuẩn từ đầu; B không phải đợi C để benchmark ASR. A cung cấp mock observation cho development và adapter thật sớm; mock tuyệt đối không dùng để chấm kết quả cuối.

### 6.2. Quy tắc phối hợp

- Mỗi ngày 15 phút: baseline hiện tại, blocker, 3 nhóm lỗi lớn và việc ngày tiếp theo.
- Một cấu hình baseline được tag cố định để quay lại; mỗi thay đổi đo trên cùng split và ghi run ID.
- PR nhỏ theo chức năng; thay đổi API cần cập nhật contract và báo hai người còn lại trước khi merge.
- Tích hợp hàng ngày. Người phát hiện bug sở hữu repro; owner module sở hữu bản sửa.
- Với một GPU: chia ca benchmark và training; ưu tiên chạy baseline/release, không để training chiếm hết thời gian kiểm chứng.

## 7. Lịch triển khai đề xuất

Lịch dựa trên hạn 16/11/2026; đóng băng nội bộ 14/11 để có khoảng đệm. Nếu đọc tài liệu sau các ngày này, dời theo số ngày còn lại và cắt experiment trước, không cắt kiểm tra submission.

| Mốc | A | B | C | Điều kiện nghiệm thu |
|---|---|---|---|---|
| **01–02/11: xác minh và nền móng** | Đối chiếu kit, khóa split, chốt input/output, tạo runner tối thiểu | Kiểm kê audio, benchmark 1 ASR, thống kê train | Chạy 1 LLM trên transcript chuẩn, thiết kế action contract | Có checklist nguồn, contract chung và vài mẫu chạy thật qua ToolEnv |
| **03/11: baseline xuyên suốt** | Nối pipeline, evaluator, JSONL; Docker bản đầu | Nối ASR module; đo lỗi thực thể | Hoàn thiện execute/ask/OOS cơ bản | Có báo cáo toàn dev end-to-end; mọi mẫu có trạng thái hoàn thành/lỗi rõ; không cần điểm cao ngay |
| **04–05/11: sửa lỗi nền tảng** | Validator, đối chiếu metric, test trace/hash | So sánh tối đa 2 ASR; regression số/tên/giờ | So sánh tối đa 2 LLM; sửa grounding, history, stop | Chọn cấu hình baseline theo score + tài nguyên; có taxonomy lỗi |
| **06–08/11: cải thiện có mục tiêu** | Profiling, kiểm thử extension và cache | Tối ưu normalization; hỗ trợ dữ liệu SFT, đánh giá từng cặp lỗi | Một thử nghiệm SFT nhỏ nếu cần; nếu không, tiếp tục sửa Agent | Ablation rõ thay đổi nào có ích; không nhận model chỉ vì train loss giảm |
| **09–11/11: hội tụ** | Kiểm thử timeout, p95/max, chạy container offline | Kiểm tra regression và lineage, benchmark audio dài | Chốt checkpoint/prompt ứng viên; bỏ thử nghiệm không hiệu quả | Chạy full dev ổn định, chọn ứng viên bằng dev; có cấu hình fallback |
| **12–13/11: diễn tập nộp** | Rebuild và chạy đúng entrypoint, validate output, checksum | Kiểm tra weights/tokenizer/data cần thiết và hướng dẫn chạy | Kiểm tra generation config, giới hạn context, model packaging | Container sạch không cần mạng; đúng schema/ID; tài nguyên đạt; đánh giá holdout một lần sau khi khóa ứng viên |
| **14/11: đóng băng nội bộ** | Tag release, đóng gói, lưu checksum | Hoàn tất model/data manifest và báo cáo ASR | Hoàn tất tài liệu Agent và kết quả cuối | Chạy lặp, rà cả lỗi hiếm, khóa cấu hình; cả ba review |
| **15–16/11: đệm và nộp** | Owner thao tác nộp, kiểm tra trạng thái nhận | Kiểm tra độc lập file/hash | Kiểm tra độc lập version/config | Nộp sớm; chỉ sửa blocker; lưu bằng chứng nộp |
| **20–23/11: private** | Chạy bản đóng băng đúng hướng dẫn BTC | Giám sát input/runtime | Kiểm tra tính đầy đủ output | Không tinh chỉnh model/prompt từ private; nộp trước hạn |

### 7.1. Đường găng và phương án khi trễ

Đường găng: xác minh protocol → runner/evaluator → baseline thật → benchmark end-to-end → container offline → đóng băng/nộp.

- Nếu đến 03/11 chưa có baseline: cả đội dừng thử model mới, tập trung kết nối pipeline.
- Nếu đến 08/11 SFT chưa tạo lợi ích đáng tin: giữ prompting baseline, không chuyển sang GRPO.
- Nếu đến 11/11 latency/VRAM chưa đạt: giảm prompt và output, xử lý history có chọn lọc nhưng giữ bằng chứng cần thiết, thử runtime/quantization hoặc model nhỏ hơn và đánh giá lại.
- Nếu chỉ còn 7 ngày: dành 2 ngày baseline, 2 ngày sửa lỗi lớn nhất, 1 ngày đo đầy đủ, 2 ngày đóng gói/đệm; bỏ fine-tuning nếu chưa sẵn dữ liệu.

## 8. Checklist nghiệm thu trước khi build và trước khi nộp

### Trước khi đầu tư training

- [ ] A xác minh schema, matching, hash, custom extensions và giới hạn thực thi.
- [ ] A xác minh quyền đọc file/mã nguồn trong kit; code giải pháp không trực tiếp mở state ẩn.
- [ ] Có split ID cố định, group leakage được rà và data lineage.
- [ ] B/C có baseline gold-transcript và end-to-end trên cùng dev.
- [ ] Có baseline tag, hợp đồng module và owner rõ.
- [ ] Có tài nguyên thực tế để chạy thử; model shortlist có license/revision/ngày phát hành và số tham số.

### Test case bắt buộc của hệ thống

- [ ] Ngoài phạm vi: không có call thật; ask: đúng schema/params và trạng thái.
- [ ] Thiếu ID → read → chọn thực thể → write; ID đã có không bị read lặp vô ích.
- [ ] Nhiều thực thể cùng tên; history có tham chiếu; ngày mai vượt ngày/tháng; câu phủ định; số tiền/số điện thoại.
- [ ] `x-one-of`, required, enum, kiểu dữ liệu, `null` và ID không có bằng chứng.
- [ ] Proposal lỗi bị chặn trước env; runtime tool lỗi được ghi nhận đúng, không giả kết quả.
- [ ] Nhiều write phụ thuộc nhau; cache được làm mới hợp lý; không lặp side effect.
- [ ] Không có truy cập env/state chéo mẫu; không đọc gold fields trong inference.
- [ ] Trace/hash xuất từ cùng env; đủ ID, không trùng và không thiếu dòng submission.

### Trước khi nộp

- [ ] Container chạy offline từ môi trường sạch với đúng lệnh BTC quy định.
- [ ] Weight, tokenizer, adapter, dependency và config đều có sẵn, pin phiên bản; không tải ngầm lúc chạy.
- [ ] Đo peak VRAM, latency max/timeout và audio/context dài; không chỉ báo trung bình.
- [ ] Chạy lại ít nhất 3 lần trên regression set và kiểm tra full run cuối; khác biệt trace/hash được ghi rõ.
- [ ] Có code huấn luyện, inference, manifest nguồn dữ liệu/model và checksum theo yêu cầu đã xác minh.
- [ ] Ba người review release; A nộp, B/C đối chiếu version và trạng thái tiếp nhận.

## 9. Việc mỗi người làm ngay trong buổi đầu

**A:** lấy kit/rules chính thức, đánh dấu các khẳng định chưa xác minh; chốt format sample/output, split và runner tối thiểu. Ưu tiên biết mình đang chấm đúng trước khi tối ưu.

**B:** đọc cấu trúc audio/history; chạy ASR trên một tập nhỏ đa dạng lấy từ phần train/dev đúng mục đích; liệt kê lỗi tên riêng, số, giờ và phủ định. Chưa xây post-processing phức tạp khi chưa biết lỗi phổ biến.

**C:** chạy Agent với transcript chuẩn trên các mẫu có execute, ask, OOS và nhiều bước; ghi lỗi chọn tool/arguments, thống nhất action contract với A. Chưa bắt đầu GRPO.

**Cả đội:** chốt người nhận A/B/C, số giờ thực tế, GPU hiện có và cách họp/merge. Đầu ra cần đạt trước tiên là một lần chạy hoàn chỉnh có thể đo và tái hiện, sau đó mới là một model mạnh hơn.


## 10. Kết quả review lần hai và điều kiện sẵn sàng

**Kết luận:** tài liệu đủ làm kế hoạch hành động cho giai đoạn xác minh và baseline. Chưa thể coi là đặc tả triển khai đã được kiểm chứng hoặc bảo đảm hoàn thành đúng lịch. Các đầu việc chưa tick là việc phải làm, không phải kết quả đã đạt. Chưa có benchmark nào được thực hiện trong quá trình viết tài liệu.

### 10.1. Ba cổng quyết định

| Cổng | Phải có | Nếu chưa có |
|---|---|---|
| Bắt đầu baseline | Kit chính thức truy cập được; xác định input/output và API env; 3 người nhận vai trò; máy đủ chạy ít nhất một ứng viên | Làm loader/mock và đọc kit; không tuyên bố end-to-end thật hoặc chốt model |
| Bắt đầu training | Evaluator đáng tin, split cố định, baseline đo được, lỗi có thể cải thiện bằng training, compute và quyền sử dụng dữ liệu/model đã xác minh | Tiếp tục sửa pipeline/prompt và đo; không mở SFT/GRPO chỉ để theo lịch |
| Đóng băng/nộp | Rules đã kiểm chứng, container offline chạy thật, đầu ra đủ/đúng format, tài nguyên đạt, artifacts và checksum đúng | Dùng baseline hợp lệ đã kiểm chứng; không nộp cấu hình mới chưa đo chỉ vì điểm một lần cao |

Không cần đợi chốt mọi chi tiết để viết loader và baseline. Nhưng các thông tin về luật, dữ liệu, hardware không thể được thay bằng giả định khi chọn giải pháp cuối.

### 10.2. Bổ sung kiểm tra evaluator — A chủ trì, C đối chiếu

- Dùng evaluator chính thức nếu có. Chỉ viết adapter/report trước; tự viết metric khi thật sự cần và đối chiếu kết quả từng mẫu.
- Nếu kit cho phép, replay gold trajectory của một tập train đa dạng qua env mới cho mỗi lần đánh giá; so trace và hash với nhãn. Khác biệt phải được điều tra trước khi kết luận model sai.
- “Một env/mẫu” áp dụng cho một lần inference: các experiment độc lập cần trạng thái khởi đầu sạch theo kit, không tái dùng env đã bị mutate từ run trước. Không chạy nhiều rollout trên cùng state rồi chọn trace đẹp nhất.
- Tạo fixture kiểm tra: gold rỗng, thiếu call, thừa read, call trùng, sai hash, fuzzy sát ngưỡng và call thất bại. Kết quả phải khớp evaluator chính thức trên từng fixture được hỗ trợ.
- Nếu không có evaluator chính thức hoặc hash không thể tái hiện bằng API được phép, ghi rõ local score là proxy; không gọi đó là điểm chính thức chính xác 100%.

### 10.3. Quy tắc chọn cấu hình — chốt trước khi xem holdout

1. Loại cấu hình vi phạm quy định, lỗi format, thiếu model offline, hoặc vượt tài nguyên đã xác nhận.
2. So baseline và ứng viên trên cùng toàn bộ dev, cùng preprocessing, cùng runtime mục tiêu; không so điểm từ hai subset khác nhau.
3. Ưu tiên điểm end-to-end theo evaluator; báo số mẫu cải thiện và thoái lui, đặc biệt state, ask/OOS. WER hoặc gold-transcript score không thay cho end-to-end.
4. Với chênh lệch nhỏ, kiểm tra lỗi theo cặp và có thể dùng bootstrap theo conversation/group để mô tả độ bất định; không bootstrap từng dòng như độc lập nếu cùng session. Không đặt một ngưỡng tăng điểm tùy ý rồi coi chắc chắn tổng quát hóa.
5. Nếu không có bằng chứng cải thiện đủ rõ, giữ baseline đơn giản/nhanh hơn. Chạy lặp đo tính ổn định, không dùng ba lần chạy cùng dữ liệu như ba tập validation độc lập.
6. Khóa ứng viên bằng dev, sau đó chạy holdout một lần. Nếu dùng holdout để chọn giữa nhiều ứng viên hoặc sửa giải pháp, phải công khai rằng nó đã tham gia lựa chọn; không còn là đánh giá độc lập.
7. Bản kế hoạch mặc định giữ checkpoint đã đánh giá để nộp. Nếu muốn retrain trên toàn train chính thức, đó là ứng viên mới cần kế hoạch kiểm chứng riêng; điểm holdout cũ không chứng minh checkpoint mới có cùng chất lượng.

### 10.4. Audio, history và prompt — B/C đồng sở hữu

- Loader xác nhận số file audio mỗi mẫu, sample rate, số kênh, độ dài, file thiếu/hỏng và history là text hay audio. Resample đúng một lần khi cần; không cắt âm thầm phần đuôi để đạt latency.
- Nếu history có audio, tính cả chi phí ASR history theo protocol. Benchmark gold-transcript phải thay thế những trường audio nào được xác định rõ và giữ các đầu vào khác tương đương.
- Cache transcript local phải khóa theo audio checksum, model revision và preprocessing config; không dùng ID mẫu làm khóa duy nhất. Inference đánh giá thật không đọc transcript chuẩn hoặc cache đáp án test.
- Giữ câu phủ định, quan hệ trước/sau, đơn vị và ranh giới lượt nói. Normalization không biến câu mơ hồ thành một yêu cầu chắc chắn.
- Xác nhận chat template và format tool call của checkpoint; giới hạn output token và lịch sử dựa trên tokenizer thật. Context cần tính cả schemas, observations, history và phần output dự kiến.
- Khi phải rút gọn context: giữ yêu cầu hiện tại, tham chiếu chưa giải quyết và kết quả tool cần cho bước kế tiếp. Không chỉ cắt N token đầu/cuối tùy tiện.
- Nội dung email/ghi chú/tool result là dữ liệu, không phải chỉ dẫn được phép thay đổi policy hoặc gọi tool ngoài yêu cầu. Cho vào prompt dưới vai trò/khối dữ liệu phù hợp và có regression case chứa câu giống mệnh lệnh.

### 10.5. Kết thúc, lỗi và deadline — A/C đồng sở hữu

| Tình huống | Hành vi đề xuất |
|---|---|
| Proposal sai JSON/schema | Chặn local; cho sửa có giới hạn, khởi đầu tối đa một lần mỗi bước và điều chỉnh bằng dev; không gọi env bằng dữ liệu lỗi |
| Lặp lại proposal bị validator từ chối | Kết thúc sửa khi hết giới hạn; ghi lỗi nội bộ; không dùng ask giả để che lỗi runtime |
| Read lỗi | Chỉ thử lại khi protocol cho phép, có lý do lỗi tạm thời và còn ngân sách; tính cả call lỗi trong trace thật |
| Write lỗi/không rõ kết quả | Không retry mù; dùng trạng thái/log API được phép để xác định kết quả; không giả định rollback |
| Hoàn thành hoặc đã gọi ask | Dừng theo protocol được xác minh; không phát sinh read kiểm tra không cần thiết |
| Audio hỏng, model lỗi, sắp hết giờ | Không suy ra OOS; ghi nguyên nhân, giữ trace/hash thực nếu protocol cho phép xuất, đánh dấu sample thất bại trong báo cáo local |

Deadline được runner quản lý bằng đồng hồ monotonic và ngân sách còn lại truyền cho inference. Không chỉ kiểm tra thời gian sau khi LLM đã chạy quá hạn. Xác minh khả năng hủy của runtime; dành thời gian serialize output. Nếu phải hủy process trong lúc write, không tự khởi động lại mẫu và lặp side effect khi chưa xác định protocol phục hồi.

### 10.6. Tải công việc và bàn giao còn thiếu

Giả định 4–6 giờ/người/ngày trong 13 ngày tương đương 156–234 giờ-người tổng. Đề xuất chỉ cam kết khoảng 75% cho việc đã lên lịch, phần còn lại dành cho lỗi tích hợp và việc học công cụ. Đây là ngân sách lập kế hoạch, không phải ước lượng đã kiểm chứng cho từng task. Nếu năng lực thực thấp hơn, dùng lịch rút gọn ở mục 7.1.

| Quyết định cần điền | Owner | Hạn nội bộ | Bằng chứng hoàn tất |
|---|---|---|---|
| Tên người nhận A/B/C, giờ rảnh, người dự phòng | Cả đội | Trước buổi làm đầu tiên | Bảng người/vai trò và lịch làm |
| GPU dev, RAM, disk, môi trường gần A30, ngân sách thuê nếu cần | A tổng hợp, B/C cung cấp nhu cầu | 02/11 | Cấu hình thật và slot chạy, không dựa vào cấu hình giả định |
| Rules/kit version và các giới hạn chưa xác minh | A | Trước chốt model | Link/file chính thức + ngày truy cập + kết luận |
| Checkpoint ASR và LLM cụ thể | B và C; A kiểm tra hợp lệ | 05/11 hoặc sớm hơn | Revision, tham số, license, benchmark, peak VRAM |
| Training có cần thiết không | C đề xuất, cả đội quyết định | 06/11 | Nhóm lỗi mục tiêu, compute, tiêu chí dừng |
| Entrypoint, output protocol và release artifacts | A; B làm người chạy lại độc lập | Bản đầu 03/11, hoàn tất 13/11 | Lệnh chạy đã thử và kết quả từ container sạch |

B là người dự phòng chạy pipeline/nộp khi A vắng; C bàn giao model config và cách tái hiện inference để B/A không phải đoán. Mỗi deliverable phải có đường dẫn trong repo, lệnh tái hiện, run ID hoặc kết quả nghiệm thu. Việc “đã code xong” chưa phải “đã bàn giao”.

**Các mục chưa thể điền thay đội:** tên và sở trường thành viên, giờ làm, GPU/ngân sách, kit/evaluator thực tế và quy định đầy đủ. Khi có các đầu vào này, cần hiệu chỉnh lịch và model shortlist; không cần viết lại toàn bộ hướng kiến trúc.
