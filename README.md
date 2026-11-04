# plan_C — Triển khai LifeMate Phần C với Qwen3-4B-Instruct-2507

Ngày lập: **04/11/2026**, múi giờ **Asia/Saigon (UTC+7)**. Mốc bàn giao: **16/11/2026**.

Repository đã audit: `D:\Zalo AIC`, origin `https://github.com/Platypus27-coder/zalo-ai-challenge.git`, commit **`1f9186ffb8bb167e2c6de8a3091f717cef099229`**. Working tree ban đầu sạch. Phạm vi lần lập kế hoạch này: chỉ thêm `plan_C.md`; các thay đổi code bên dưới là công việc tương lai.

## 1. Quyết định và cách đọc tài liệu

Chọn **`Qwen/Qwen3-4B-Instruct-2507`** làm LLM duy nhất. Triển khai một agent tuần tự, một hành động mỗi lượt, dùng công cụ để lấy dữ liệu thật trước khi quyết định hành động tiếp theo. Giữ ASR **faster-whisper large-v3-turbo** làm ứng viên tích hợp với B; chưa tuyên bố đạt tốc độ hoặc chất lượng trên A30.

Ký hiệu:

- **[R] Verified repo fact**: đã đọc trực tiếp file tại commit trên; không có nghĩa đã chạy thành công.
- **[S] Verified source**: đã đối chiếu nguồn model chính thức/Hugging Face ngày 04/11/2026.
- **[P] Proposed**: thiết kế, ngân sách hoặc tiêu chí nội bộ cần triển khai/đo.
- **[V] Verify next**: chưa có SDK, trọng số, phần cứng hoặc bằng chứng để kết luận.

Ưu tiên quy định LifeMate được người dùng cung cấp hơn gợi ý cũ trong tài liệu repo. Không kế thừa đề xuất 7B AWQ: quantization giảm dung lượng, **không giảm số tham số kiến trúc**. Không coi các con số hiệu quả trong kế hoạch cũ là kết quả thực nghiệm.

## 2. Ràng buộc bắt buộc và cổng tuân thủ

| Ràng buộc | Cách triển khai/kiểm chứng [P] | Điều kiện loại ứng viên |
|---|---|---|
| Tổng hệ thống **<6.000.000.000** tham số; LLM **<5.000.000.000** | Manifest đếm unique parameters, gồm ASR, LLM, VAD và mọi model phụ, adapter chưa merge | Thiếu thành phần trong manifest hoặc chạm/vượt giới hạn |
| Context **<=16.384 tokens** | Kiểm tra mỗi lượt: toàn prompt sau chat template + giới hạn output <=16.384; giới hạn runtime 16.384, không dùng native 256K | Vượt ở bất kỳ lượt nào |
| **<=30s/sample**, **1×A30**, CPU 16 core, RAM 64GB | Đo end-to-end cả audio/history, env, decode, validation, serialization; timeout thực sự | Timeout, OOM, crash hoặc chỉ có số đo T4/2 GPU |
| Offline Docker, không API bên ngoài | Trọng số/tokenizer/VAD/runtime nằm local; chạy với `--network none` và cache trống | Runtime cần tải thêm hoặc lặng lẽ đổi model |
| Model phát hành **trước 26/10/2026**, license phù hợp | Lưu model ID, revision, metadata ngày, LICENSE/NOTICE và checksum | Revision/giấy phép chưa xác minh |
| Không đọc/copy/parse file bên trong `<split>_environment/`, kể cả `state.json` | Agent chỉ nhận input được phép và observation từ SDK; audit cả các dependency trên đường inference | Agent/runner truy cập trực tiếp environment |
| **Exactly one ToolEnv/sample/inference run** | Runner tạo duy nhất một instance, truyền vào C; không tạo lại khi lỗi/OOS/timeout | Retry sample bằng env mới, clone hoặc speculative execution |
| Chỉ `validate`, `call`, `get_tool_calls`, `get_state_hash` trên env khi inference | Adapter hẹp; spy test chặn mọi method/attribute khác | `get_state`, `states`, Executor, Simulator, deepcopy/clone env |
| Trace đầy đủ | Xuất nguyên thứ tự trace từ `get_tool_calls()` của chính env; hash từ `get_state_hash()` | Lọc bỏ read, call lỗi, trùng lặp hoặc dựng trace bằng proposal |
| Deterministic | Greedy ASR/LLM, khóa seed/backend/runtime/model/template và kiểm tra nhiều lần | Trace/hash khác trên cùng release và input |
| Không train/tune bằng nhãn public/private | Train/dev/holdout đều là partition của official train; public/private chỉ inference theo quy chế | Gán nhãn test, external labeling test, few-shot/test-derived tuning |

Quy định ở bảng được lấy từ yêu cầu hiện tại và nội dung thể lệ người dùng đã cung cấp trong chat trước. Trang [LifeMate](https://challenge.zalo.ai/portal/lifemate) không trả nội dung đọc được qua công cụ web trong lần audit này. **[V] A lưu bản thể lệ/hướng dẫn Docker chính thức, xác nhận giờ cutoff và layout mount trước release.** Repo ghi 16/11/2026 23:59 GMT+7; dùng mốc đó để lập lịch, không coi là vừa được xác minh lại từ portal.

## 3. Model, license và số tham số thực

### 3.1. Qwen được chọn

**[S]** Model card ghi non-thinking-only, native context 262.144; Transformers trước 4.51.0 không nhận `qwen3`. Dùng native function calling qua chat template, không thêm chain-of-thought. Card có ví dụ sampling, nhưng cuộc thi yêu cầu greedy. Nguồn: [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

**[S]** Metadata trả revision **`cdbee75f17c01a7cc42f958dc650907174af0554`**, `createdAt=2025-08-05`, `lastModified=2025-09-17`; đều trước cutoff. Đây là bằng chứng tồn tại của đúng repo, không suy ngày phát hành từ hậu tố `2507`. API cũng trả **4.022.468.096** tham số BF16. Nguồn: [metadata Qwen](https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507).

**[S]** LICENSE thực là **Apache-2.0**. Khi đóng gói: kèm bản license, giữ attribution/copyright, giữ NOTICE nếu artifact có, ghi nhận các file sửa đổi nếu phân phối bản sửa. Không tự đổi tên model thành license khác. Nguồn: [LICENSE](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/LICENSE).

### 3.2. Large-v3-turbo + 4B có vượt 6B không?

**Không vượt khi chỉ cộng hai model này.** Đếm unique model parameters, gồm embedding và parameter không trainable được artifact khai báo; không đếm tied output projection lần thứ hai. Không giảm số đếm chỉ vì freeze weights.

| Thành phần | Số tham số chính xác [S] | Nguồn |
|---|---:|---|
| Qwen3-4B-Instruct-2507 | **4.022.468.096** | API safetensors metadata + tính lại từ config |
| Whisper large-v3-turbo | **808.878.080** | API safetensors metadata + tính lại từ config |
| Tổng hai model | **4.831.346.176** | Phép cộng trực tiếp |
| Khoảng cách đến 6B | **1.168.653.824** | Tổng model phụ phải **nhỏ hơn** khoảng này |
| Khoảng cách LLM đến 5B | **977.531.904** | Không được cộng model phụ dạng LLM vượt trần |

Nguồn ASR: [metadata OpenAI](https://huggingface.co/api/models/openai/whisper-large-v3-turbo), [config ASR](https://huggingface.co/openai/whisper-large-v3-turbo/blob/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/config.json). Metadata ASR: revision **`41f01f3fe87f28c78e2fbf8b568835947dd65ed9`**, tạo 01/10/2024, cập nhật 04/10/2024, license MIT. Lưu attribution/license từ nguồn phát hành khi bundle.

Tính độc lập Qwen theo [config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json): đặt `h=2560`, `i=9728`, `q=4096`, `kv=1024`, `v=151936`, `L=36`. Mỗi block có Q/K/V/O, ba projection MLP, hai RMSNorm và Q/K head norms:

```text
P_Qwen = v*h + L*(h*q + 2*h*kv + q*h + 3*h*i + 2*h + 2*128) + h
       = 4,022,468,096
```

Whisper: `d=1280`, `f=5120`; 32 encoder blocks, 4 decoder blocks, 128 mel bins, vocabulary 51.866. Tính convolution, positional embedding và final layer norms. Bảng vị trí encoder sinusoidal cố định vẫn được tính trong số parameter artifact, không bỏ nó chỉ vì không trainable:

```text
E = 4*d*d + 8*d + 2*d*f + f = 19,676,160
D = 8*d*d + 13*d + 2*d*f + f = 26,236,160
P_Whisper = (128*3*d+d) + (d*3*d+d) + 1500*d + 32*E + 2*d
            + 51866*d + 448*d + 4*D + 2*d
          = 808,878,080
```

Số **808.878.080** ở trên đã bao gồm `1500*d=1.920.000` phần tử bảng vị trí encoder; không cộng bảng đó lần nữa. Khi đối chiếu PyTorch và safetensors, tách trainable/non-trainable/buffers để giải thích chênh lệch, nhưng dùng tổng parameter đầy đủ làm gate. Không lấy kích thước file/2 hoặc tên “4B/800M” làm phép đếm.

**[R/V] Chưa được tuyên bố toàn pipeline có đúng 4,831B:** `asr_module.py` bật `vad_filter=True`; bản VAD và số tham số chưa có local. `faster-whisper` tải bản CTranslate2 theo alias, không pin revision. Trước M1, B phải xác nhận upstream/decoder layers/conversion, license, checksum và đếm VAD ONNX theo initializer graph, không đếm constant/buffer hay tensor chia sẻ hai lần. Phương án dễ audit: tự chuyển đúng upstream Whisper đã pin sang CTranslate2; hoặc xác minh provenance bản chuyển sẵn. Nếu tắt VAD để không có model phụ, phải benchmark lại và ghi đó là cấu hình khác baseline.

**[P] Manifest bắt buộc:** component, upstream ID/SHA/date/license, file hashes, format/dtype, unique parameter count (trainable và non-trainable), buffer count riêng, conversion script/version, adapter status. Cross-check bằng `sum(p.numel() for p in model.parameters())` trên model chưa quantize (meta/init-empty để tránh tải weights nếu chỉ kiểm kiến trúc), và tensor headers/metadata. Sau LoRA merge đếm lại artifact cuối; adapter không merge phải tính phần tăng thêm. Không cộng hai bản lưu trữ của cùng một model như hai mạng khác nhau; phải khai báo mọi mạng thực sự được dùng khi inference.

### 3.3. Bộ nhớ và thời gian chỉ là ước lượng trước đo

**[P]** Qwen BF16/FP16 weights khoảng 8,045GB thập phân; ASR weights khoảng 1,618GB. Đây chưa gồm KV cache, activation, CUDA/CT2 workspaces, allocator, VAD và audio. Qwen KV ở 16.384 token, 36 lớp, 8 KV heads, head_dim 128, 2 byte: `2*36*8*128*16384*2 = 2.415.919.104 byte` cho batch=1 nếu không nén. Không suy từ các phép tính này rằng pipeline chắc chắn vừa 24GB hoặc đạt 30s.

## 4. Audit repository và những việc cần sửa trước benchmark

Đã kiểm kê source A/B/C, scripts, ba notebook, Docker/config/dependencies, tài liệu và split/report đã commit. Không tìm thấy `AGENTS.md` trong repo hoặc ở gốc `D:\`. **[R]** Checkout không có `train+public_test/`, `zaic_tools/` hay `models/`; vì vậy không audit được triển khai SDK, catalog thực, dữ liệu audio hay chạy benchmark mới.

| File/interface [R] | Hiện trạng quan sát được | Việc triển khai/kiểm chứng [P/V] |
|---|---|---|
| `src/part_c/agent.py` | Entry thực là `BaseAgent.run(transcript, sample, env)->None`, factory rule/transformers/vllm; gọi env.validate rồi call | Giữ tương thích entry này; chốt adapter deadline/diagnostic, không thay bằng `LifeMateAgent.run_sample` trong tài liệu mà bỏ runner |
| Vòng `BaseAgent.run` | Không import/use Ledger hay Tracker; max_retries được lưu nhưng không áp dụng; không có deadline; không dừng riêng sau ask | Nối ledger/tracker, repair hữu hạn, terminal ask và shared deadline |
| Load/generate | Mặc định Qwen2.5-3B, `trust_remote_code=True`; generate bỏ qua tools; vLLM cap 16K nhưng Transformers không cap | Pin Qwen3/local-only/trust_remote_code=False; truyền tools/template; cap cả hai backend |
| `prompt.py` | Protocol `CALL: {...}`/`DONE`; regex có fallback rộng; history assistant chỉ giữ tên call; observation cắt theo 500 ký tự | Protocol Qwen nhất quán, parse strict; giữ arguments/result có nguồn; chọn field theo token, không cắt JSON/ID giữa chừng |
| `tool_filter.py` | Lấy danh sách tool chính xác từ installed_apps và thêm ask; có converter function schema | Dùng cùng allowlist tại prompt, grammar, routing và validator; validate catalog thiếu/duplicate |
| `evidence_ledger.py` | Có class nhưng chưa nối agent; extractor chỉ một số domain; đọc result top-level; lịch sử args được seed như entity | Kiểm chứng result envelope; provenance/ambiguity; không mặc định args trong lịch sử là trạng thái hiện tại |
| Ledger resolve/has_entity | Entity current được cộng điểm dù không khớp query; tie lấy phần tử đầu; `unknown` vẫn có thể qua | Không resolve chỉ vì current; bắt buộc đủ ràng buộc ngữ nghĩa, unique candidate, chặn stale/unknown ở thuộc tính cần freshness |
| `task_tracker.py` | Không tự phân rã yêu cầu; `mark_executed` coi thiếu ok là thành công; read có thể đóng task | Track subgoal và bằng chứng hoàn tất đúng tool/kind; không mặc định read hoàn thành write goal |
| `src/part_a/validator.py` | Hàm `validate_call(name, arguments, specs, installed_tools)` trả ok/errors; có import SDK schema_error; fallback không kiểm x-ref/business logic | C dùng local validator API đúng, env.validate là chốt cuối; parity theo SDK thực; không import helper nội bộ SDK trên đường inference nếu chỉ bốn API được phép |
| `runner.py` | Tạo một env rồi gọi agent_fn 3 args; docstring nói thêm validator; max_time chỉ lưu, không enforce; không ASR thì dùng audio_text | Chốt signature; deadline thực; gold mode chỉ train-derived; sanitize sample bỏ label trước C |
| `scripts/run_offline.py` | Gọi transcribe(..., context=...) nhưng ASR nhận sample; đọc raw/normalized_transcript thay vì raw/normalized_text | Sửa contract trước mọi đo end-to-end; test bằng fake ASR trả đúng schema |
| Offline recovery | Thiếu local model thì rơi về HF alias; load LLM lỗi thì dùng rule agent; exception sample ghi [] và hash rỗng | Fail startup khi thiếu artifact; không đổi model ngầm; lỗi sau call vẫn xuất trace/hash thật của env cũ |
| `generate_submission.py` | Có private_test nhưng mặc định ASR gold và agent rule | Entry chính thức phải require whisper + Qwen local; từ chối gold mode trên public/private |
| `submission_linter.py` | Đường dữ liệu hard-code train/public; private rơi về train; CLI chỉ train/public; hash rỗng chưa luôn critical; nhiều schema lỗi chỉ warning | Inject input/catalog/split paths; đủ sample, hash hợp lệ, schema lỗi thành release failure; exit nonzero khi lint fail |
| `evaluator.py` | Local metric dựa hash, call matching greedy một-một, fuzzy >0.6; denominator theo prediction nhận vào | Parity với official scorer, case overlapping fuzzy matches, thiếu/duplicate ID; không tin điểm khi coverage thiếu |
| `state_diff.py`, `evaluate.py` | State diff tạo hai env và gọi get_state; evaluate tự chạy diff; keys consumer/producer diff không khớp | Chỉ train-only dev diagnostics, tuyệt đối không trên inference/public/private; sửa report “100%” giả do keys lệch |
| `loss_attribution.py` | Heuristic có recoverable rates cố định, mặc định lỗi non-ID là ASR | Chỉ giả thuyết ưu tiên; dùng paired gold-vs-ASR để kết luận, không báo expected_gain là số đo |
| `data_splitter.py` và splits | Counts lưu 1157/265/123; code stratify nhưng chưa thấy grouping session như docstring | Freeze IDs/checksum, kiểm tra trùng/disjoint và session leakage với A; không tự regenerate từ notebook |
| `Dockerfile`, `.dockerignore`, `config.py` | Offline env flags có; COPY models; CMD `/data/public_test` -> `/output/submission.jsonl`; config vẫn trỏ train+public_test | Mount/output theo BTC (`result/submission.jsonl`); không phụ thuộc train hay outputs/splits trong final image |
| Dependencies/download | `transformers>=4.45`, `vllm>=0.6`, cài cả flash-attn; downloader mặc định model cũ, không pin, bắt lỗi rồi báo hoàn tất | Lock stack hỗ trợ Qwen3 + constrained decode trên A30; artifact thiếu phải fail; chỉ cài backend cần dùng |
| Notebooks | Có code riêng khác module, Qwen cũ/7B và device_map=auto; demo generate chưa chạy ToolEnv | Chuyển thành harness gọi module chuẩn, một GPU, đúng split; T4 chỉ smoke, A30 mới acceptance |

**[R] Báo cáo cũ:** `outputs/baseline_evaluation.json` ghi 100/100 trên 1545 mẫu; `diagnostic_dev.json` ghi 265/265 hoàn hảo. `run_baseline.py` thực hiện gold-call replay. Các file này thiếu bằng chứng Qwen3 mới, ASR mới và cấu hình đo; **không dùng để khẳng định agent đạt 100 hoặc pipeline đã sẵn sàng**. EDA/split đã commit là artifact lịch sử, không phải dữ liệu gốc được kiểm lại trong lần này.

## 5. Chốt interfaces với A/B và SDK — P0

**[P] Một đường inference duy nhất:** runner load model một lần, loop sample, tạo env một lần, gọi C, thu kết quả. C không tự tạo env. Giữ `create_agent(...)` và `agent.run(transcript, sample, env)` để không gãy caller; có thể thêm keyword-only `deadline` sau khi tất cả caller đã cập nhật. Return diagnostics nội bộ nếu cần; submission luôn do runner lấy từ env.

Hợp đồng cần khóa bằng fixtures:

1. **Input C:** whitelist `id`, transcript hiện tại, context, history được phép và transcript history nếu có; không `label`, gold calls/hash, đường environment hoặc đáp án cache. Current gold `audio_text` chỉ được adapter benchmark train-derived lấy rồi chuyển thành transcript; không truyền label vào model. Giữ nguyên tên trường theo SDK đã xác minh.
2. **ASR B:** dùng thực `transcribe(audio_path, sample=sample)` -> `{raw_text, normalized_text, inference_time, segments,...}`; hoặc thống nhất adapter có tên mới, nhưng sửa tất cả caller cùng lượt. Empty audio/text là lỗi/thiếu thông tin để policy xử lý, không tự suy OOS.
3. **Tool catalog:** snapshot input catalog, giữ `kind`, required, enum, numeric bounds, arrays, format, x-ref, x-one-of. Không suy kind từ hậu tố tên.
4. **Validation:** local API như source; error/messages khác env cần normalize riêng. Kiểm tra bool/int, unknown field, nested object, required rỗng và x-one-of. Không đồng nhất x-one-of “ít nhất một nhóm” với JSON Schema oneOf “đúng một nhánh”.
5. **ToolEnv [V]:** đọc README/source SDK ở `zaic_tools` khi có, không mở environment files. Xác minh constructor/root layout, validate return/error key/side effects, call result envelope, trace chứa call lỗi/thứ tự/default args, ask schema/terminal semantics, hash. Lưu version/hash SDK và fixtures đã rút từ train qua API hợp lệ.
6. **Output:** đúng `{id, tool_calls, state_hash}`, nguyên trace; diagnostics/timing/token counts ở file riêng không chứa state dump. Không replay trên env thứ hai để xác nhận hash trong inference.

**M0 đạt** khi các hợp đồng có fixture và smoke pass với SDK thực. Chưa có SDK thì có thể làm fake interface tests nhưng không đánh dấu M0 hoàn thành.

## 6. Model loading, chat template và function calling

### 6.1. Backend

**[P]** Transformers là reference để kiểm template/parser; vLLM là ứng viên tối ưu batch=1 nếu benchmark thực chứng minh cần và đạt deterministic. Không chạy cả hai LLM trong final inference. Pin một stack Linux/CUDA/PyTorch/CT2/Transformers/backend/grammar đã test; model card nêu tối thiểu Transformers 4.51.0 và vLLM 0.8.5, nhưng phiên bản tối thiểu không thay cho kiểm tương thích structured output.

- Model/tokenizer load từ thư mục tuyệt đối local, `local_files_only=True`, `trust_remote_code=False`, GPU0; eval + inference_mode. BF16/FP16 chốt sau thử cùng ASR trên A30. Không `device_map=auto` làm spill CPU hoặc dùng GPU thứ hai ngoài dự tính.
- Preflight kiểm checksum, architecture `qwen3`, tied embeddings, số tham số, template, EOS/pad, license. Không tải latest hoặc fallback rule khi fail.
- Load và warm-up trước loop nếu hướng dẫn BTC cho phép; báo riêng cold start và warm runtime, đo cả first-sample. Không dùng model-loading exclusion để che timeout.

### 6.2. Protocol thống nhất

**[S]** [tokenizer_config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer_config.json) có template `tools`, assistant tool_calls và tool-role result; function output bọc `<tool_call>...</tool_call>`. Đây là khả năng format, không phải cơ chế tự execute.

**[P]** Truyền function schema từ allowlist vào `apply_chat_template(messages, tools=..., add_generation_prompt=True)`; không nhồi thêm toàn schema lần hai trong user prompt. Assistant history lưu call theo cấu trúc mà template hỗ trợ, observation đưa role tool; mỗi vòng chỉ sinh **một call** rồi quan sát thật. Không đưa read và write phụ thuộc nhau vào một batch trước observation.

Parser nhận đúng một frame JSON hợp lệ, name thuộc allowlist, arguments là object; chặn trailing garbage, duplicate keys, NaN/Infinity, nhiều call trong một lượt, JSON cắt dở. Không regex nhặt đại một object từ câu trả lời. Có nhánh terminal **DONE** nội bộ phân biệt hoàn tất/OOS; grammar cho phép terminal này rõ ràng, không xuất DONE như tool. Không stop-string `DONE` giữa nội dung một argument. `sys.ask_missing_info` là tool thật, không phải terminal giả.

Không cần Qwen-Agent framework tự dispatch tool; giữ executor duy nhất trong loop của repo. Không trộn protocol CALL cũ với native frame trong cùng release. Nếu reference CALL chỉ dùng làm đối chứng, tách config/parser rõ ràng và không dùng kết quả của hai protocol thay thế nhau ngầm.

### 6.3. Schema-constrained generation

**[P]** Ưu tiên grammar/token mask từ schema theo từng mẫu: union của `{name: const(tool), arguments: schema(tool)}` bọc native frame, cộng nhánh terminal. Compile/cache theo catalog hash + sorted allowlist; CPU grammar compiler không phải model phụ. Với Transformers có thể dùng thư viện constrained decoder đã pin; với vLLM chọn structured output backend của đúng phiên bản. **[V]** Spike ngày 05/11 xác nhận native wrapper + terminal + Unicode + nested schema đều hỗ trợ; không đoán tên API của một version khác.

- Dịch x-one-of giữ đúng nghĩa, không làm mất x-ref/x-match metadata cần policy. Grammar xử lý cú pháp/type/enum/range trong mức engine hỗ trợ; validator xử lý lại toàn bộ, đặc biệt format và business logic.
- Không ép toàn bộ required args xuất hiện bằng cách bịa giá trị: nhánh ask luôn khả dụng khi dữ liệu không đủ. Không biến schema thành mọi trường bắt buộc hoặc thêm default không có căn cứ.
- Grammar dead-end/context đầy/token hết: reject proposal, tối đa một lần sửa local nếu còn budget; không gửi JSON hỏng vào env. Nếu backend grammar không ổn, reference strict-parse + validation + bounded repair là phương án dự phòng phải có báo cáo lỗi riêng, không gọi đó là constrained decoding.
- Acceptance: tên tool bị cấm không sinh thành proposal được chấp nhận; malformed/schema-invalid call không đến env.call trong bộ kiểm thử. Không tuyên bố grammar bảo đảm đúng ý định hoặc ID tồn tại.

## 7. Tool routing, grounding và evidence ledger

### 7.1. Routing

**[P]** Allowlist = catalog intersect các **tool cụ thể** trong installed_apps, cộng ask đúng catalog. Không chỉ kiểm tên app. Lọc deterministic bằng sort; reject tool ngoài allowlist trước env.call dù model đã thấy nó trong history. Không dùng neural router/reranker riêng. Chỉ thêm intent pruning khi ablation cho thấy không làm mất read tool cần cho chuỗi cross-app; không mặc định số tool nhỏ hơn bảo đảm OOS đúng.

### 7.2. Ledger có nguồn và freshness

**[P]** Mỗi sample tạo ledger mới độc lập: `{entity_type, canonical_id, attributes, source_kind, source_ref/json_path, observed_step, validity, disambiguating_fields}`. Nguồn hợp lệ: context/history thật được cấp, explicit user reference, observation thành công từ env.call. Không lấy từ label, file môi trường hay ID mẫu train hard-code. Nguồn model đề xuất không tự trở thành bằng chứng.

- Extractor theo catalog x-ref + schema result thực từ train, hỗ trợ envelope/nested/list và domain cần dùng. Không giả result top-level giống class hiện tại; result lỗi không thêm current entity. Lưu raw observation riêng trong bộ nhớ để không mất nguồn khi rút gọn prompt.
- Resolver bắt buộc khớp tên/đặc tính/ngữ cảnh tham chiếu và loại entity. Zero candidates -> read tối thiểu thích hợp; nhiều ứng viên không phân biệt được -> ask; không tie-break bằng ID hoặc thứ tự xuất hiện để đoán ý người dùng.
- History chỉ có call arguments chứng minh đã nhắc/đã dùng một ID, không chứng minh call thành công hay thuộc tính còn mới. Phân biệt referenced/current/stale/deleted/unknown. Explicit user ID được bảo toàn nhưng vẫn kiểm điều kiện x-ref/freshness cần thiết.
- Write precondition: target ID có provenance, đúng entity, đúng intent, không dùng thuộc tính stale. Không bắt buộc read thừa nếu context/history/observation đã đủ chứng cứ; create không có target ID đầu vào thì không bịa read trước nó.
- Sau write thành công, invalidate thuộc tính/truy vấn chịu tác động; giữ identity nếu không bị xóa/đổi. Nếu write trả authoritative entity mới, ingest nó. Không “invalidate mọi ID rồi bắt read lại” sau mỗi write. Nếu kết quả write không rõ thành công, đánh dấu unknown và không retry mù.
- Không NFC/lowercase/trim opaque IDs, kể cả ID trong context/history và object/array lồng nhau. NFC áp dụng văn bản tự do theo field, số điện thoại giữ string/0 đầu, số tiền không làm tròn, ngày giờ dựa current_time/timezone của sample.

### 7.3. Grounding ngoài ID

**[P]** Duy trì slot provenance cho số tiền, người nhận, thời gian, phủ định, vị trí, field cập nhật. Chuyển thời gian tương đối chỉ khi ý nghĩa rõ và theo catalog format; không dùng đồng hồ hệ thống. Không tự thêm `09:00` khi người dùng chưa cung cấp giờ. Observation/tool-returned text được coi là dữ liệu, không cho nó đổi system policy hay cho phép tool mới.

## 8. Loop, validation, recovery, clarification và stop

**[P] Luồng mỗi sample:** init ledger/tracker -> build bounded prompt -> generate một proposal -> strict parse -> allowlist/schema -> grounding/intent -> env.validate -> env.call -> ghi observation/ledger/task status -> kiểm terminal/budget -> lặp. Proposal bị reject chỉ nằm diagnostic, không giả thành observation đã thực thi.

### 8.1. Validation và recovery

- Local validator và grounding trước env.validate; env.validate là chốt cuối theo SDK. Nếu kết quả validation không rõ hoặc schema đổi, không execute.
- Sửa local tối đa **1 lần/proposal**, tối đa **2 lần/sample** ban đầu; khác với số tool calls. Lỗi parse/type có thể sửa; thiếu giá trị thật không sửa bằng đoán.
- Call trả `ok=false` hoặc exception: giữ nguyên call nếu đã vào trace. Không xóa lỗi để tăng F1. Không dùng env mới, rollback hay replay. Chỉ gọi tiếp khi có hành động phục hồi có căn cứ, trong allowlist và còn deadline.
- Write có thể đã commit dù wrapper lỗi: không retry cùng write khi trạng thái thành công chưa rõ. Dùng observation/contract SDK để quyết định read xác minh nếu cần và hợp lệ; nếu không đủ thời gian thì stop an toàn với trace thật.
- Chuẩn hóa lỗi env.error/validator.errors thành channel riêng; không để model nhầm “validation chưa call” thành “write thất bại đã call”.

### 8.2. Clarification/OOS

- Missing required user slot không lấy được từ context/history/read có ích -> call `sys.ask_missing_info` với đúng `tool_name`, `params` của catalog; một lần rồi kết thúc. Hỏi đúng slot còn thiếu, không toàn bộ required list đã có.
- Cần ID có thể tra cứu thì read trước thay vì hỏi người dùng ID kỹ thuật. Ambiguous entity không còn cách phân biệt -> ask theo semantics SDK được kiểm trên train.
- Yêu cầu hoàn toàn ngoài tool cài -> không call, hash vẫn lấy từ env duy nhất. Không đọc thử để phân loại OOS. Mixed supported/unsupported cần audit policy theo train/catalog; không tự đặt “một phần OOS thì bỏ toàn bộ” hay mặc định thực thi mọi phần.
- Transcript rỗng, model lỗi, hết budget là failure category riêng, không đánh nhãn OOS để che lỗi.

### 8.3. Completion và task tracker

Không dừng sau write đầu tiên nếu còn subgoal. Track dependencies pending/blocked/done/failed; chỉ done khi observation chứng minh đúng subgoal. Read-only request có thể hoàn tất sau read; read-for-write không được kết thúc write goal. `ok` thiếu không mặc định True nếu SDK chưa quy định.

Chặn loop theo fingerprint `(tool, canonical args, relevant evidence version, subgoal)`; không chặn mù mọi call giống nhau nếu người dùng yêu cầu lặp hoặc state thay đổi. Ban đầu max **10 tool calls** là giới hạn nội bộ, **không phải quy định BTC đã xác minh**. Deadline có ưu tiên cao hơn max_steps. Stop reasons phải phân biệt completed, clarification, oos, exhausted_budget, parse_failure, execution_failure, loop_guard.

## 9. Determinism, context và ngân sách runtime

**[P]** Seed Python/NumPy/Torch/CUDA=42; PYTHONHASHSEED đặt trước process; fixed threads/order/tool schemas/sample order; eval/dropout off; torch deterministic algorithms, cuDNN benchmark off, TF32 policy và attention kernels cố định. Nếu kernel không deterministic, chọn kernel khác hoặc fail gate. Greedy decoding đơn thuần chưa chứng minh tái lập bitwise.

ASR: pin CT2/faster-whisper, language=vi, beam_size=1 cho greedy candidate, temperature=0 dạng scalar, tắt temperature fallback ngẫu nhiên; mọi thay đổi so baseline beam=5 phải benchmark. VAD config/model/hash cố định. LLM: do_sample=False/temperature=0 theo backend, không sampling từ generation_config, fixed token budget, EOS đúng tokenizer, không speculative model. Batch=1 và không đổi scheduling theo tải.

**Context:** cap 16.384 ở runtime; mỗi lượt đếm lại rendered messages/tools/evidence + output reserve. Ban đầu reserve 256 output token/call, tối đa 512 khi schema cần; số này là cấu hình đề xuất. Cắt boilerplate/observation field không liên quan trước, giữ ID/slot/precondition và thứ tự history. Không cắt tùy tiện JSON hoặc bỏ các slot quyết định; nếu không thể fit thì stop với diagnostic, không vượt trần. Persist full trace không phụ thuộc prompt bị rút gọn.

**Deadline chung:** runner tạo monotonic deadline trước xử lý sample; C nhận thời gian còn lại, không tự reset 30s. Mục tiêu nội bộ ASR <=5s, C <=16s, tool+validator <=2s, reserve 7s; **chưa có số đo**. P95 toàn pipeline <=25s là target, nhưng max <=30s mới là hard gate.

Kiểm tra trước/sau mỗi step và trong generation qua stopping criteria/cancellation backend đã test. Chỉ kiểm giữa hai tool calls không ngắt được một generation quá dài. Dành khoảng kết thúc ~1s để thu trace/hash/serialize, dừng phát proposal mới trước deadline. Nếu tool call đang chạy thì không serialize hash đua với thread đó; phải xác minh cancellation/timeout behavior SDK. Worker isolation nếu cần vẫn một env/sample, không restart để thử lại sample. Phải test cả slow call và partial write.

## 10. Kiểm thử tích hợp và tiêu chí kiểm tra

**[P]** Viết test có ý nghĩa cho hợp đồng/tính đúng, dùng fake SDK ở unit test và official train SDK ở integration; không gọi public/private để gán nhãn. Tạo `tests/` là việc tương lai, chưa có trong checkout.

| Nhóm | Ca bắt buộc | Acceptance |
|---|---|---|
| Contract ASR/runner | keyword sample, keys normalized_text, path audio/history | Không TypeError, transcript không bị rỗng do đọc sai key |
| Allowed env API | spy đếm constructor, trap thuộc tính/method cấm | Mỗi sample/run đúng 1 env, chỉ bốn method hợp lệ; không clone |
| Environment boundary | instrument agent/runner I/O; tách SDK I/O hợp lệ | Không đọc/copy/parse environment trực tiếp; SDK được quyền thực thi theo chuẩn |
| Trace integrity | read/write/error/duplicate/ask và exception sau write | Submission == env.get_tool_calls nguyên vẹn và hash thật; không [] giả |
| Function calling | escaped quotes, Vietnamese, nested object/array, multi-frame, truncation, duplicate JSON keys | Parse strict, mọi malformed proposal không đến call |
| Schema parity | required, bool vs int, enum, ranges, x-one-of, optional empty, datetime | Kết quả local/env phù hợp; khác biệt được xử lý trước call |
| Routing | tool app không cài, tool không được cấp trong app đã cài, catalog thiếu ask | Chặn tool trái phép ở executor, không chỉ trong prompt |
| Ledger | query không khớp nhưng có entity current; hai người cùng tên; result lỗi/nested; stale/unknown | Không tự chọn ID; nguồn rõ, không ledger contamination qua sample |
| History/Unicode | ID NFD/leading zero/argument lồng nhau, pronoun ở history, failed historical call | ID byte-preserved, không coi historical proposal là current state |
| Task/stop | read-only, hai writes, cross-app dependency, failed write, ask, OOS, repeat hợp lệ | Không dừng sớm hoặc lặp write; ask terminal; OOS không call |
| Timeout/context | long audio/history, long observations, slow generate/tool, 16K boundary | Không vượt context; timeout path giữ trace đã có, không env thứ hai |
| Data isolation | sample có label/gold hash; gold ASR mode ở test split | C không thấy đáp án; official entry từ chối gold/replay mode |
| Offline image | cache trống, network none, artifact thiếu, mount private-like | Không download/fallback, output đúng, không cần train files |
| Determinism | ít nhất 3 lần chạy mới cùng image, weights, input trên A30 | Tool trace/hash/JSONL giống byte-for-byte; timing ở log riêng |

Release không chỉ dựa `verify_all.py`: script hiện chủ yếu import/smoke rule agent, không kiểm Qwen3 hay inference deadline. Linter pass cũng chưa chứng minh compliance, semantic accuracy hoặc full trace.

## 11. Benchmark gold transcript vs ASR và ablations

### 11.1. Thiết kế benchmark

**[P]** Dùng split đã lưu (train 1157/dev 265/holdout 123), kiểm IDs/coverage/session leakage trước khi tin counts. Không tune trên holdout; mở một lần ngày 13/11 cho ứng viên đã khóa. Chỉ sửa lỗi compliance/blocker sau đó, không chọn prompt dựa nhãn holdout.

Ba đường chạy **riêng biệt**, mỗi sample trong mỗi inference run có một env, không dùng hai env trong cùng sample để chọn kết quả tốt hơn:

1. **Gold-call replay train-only**: kiểm SDK/scorer/hash fixture, không đánh giá suy luận agent.
2. **Qwen + gold current/history transcripts từ official train-derived dev**: đo lỗi C với đầu vào sạch; label bị tách khỏi C. Không gọi đây là upper bound tuyệt đối.
3. **Cùng Qwen/config + ASR thực cả current/history**: end-to-end; không gold fallback history. So sánh paired theo ID với (2), không thay đổi prompt giữa hai nhánh.

B xuất transcript cache dev kèm model revision/config/audio hashes; chỉ dùng cache để tái lập phân tích. Báo runtime end-to-end từ audio thực, không lấy thời gian agent trên cache làm kết quả pipeline. Không giữ sample data/ledger qua các sample; nếu cache history audio hợp lệ, key bằng content hash + ASR config và khai báo rõ cold/warm.

Đầu ra benchmark cần: score hash-gated, call precision/recall/F1 bỏ hash, raw hash-match trên tất cả và non-OOS riêng (evaluator hiện đánh OOS hash_match=True không có nghĩa raw hash đúng), OOS/ask/execute confusion, wrong-ID, invalid-call, extra-read, repeated-write, missing action, timeout/error rate. Thêm WER/CER và slot error số tiền/phone/negation/time/entity; regex hiện có chỉ là proxy, không phải nhãn slot đầy đủ.

Runtime: p50/p95/p99/max/mean ASR, LLM, tools, tổng; gồm failures/timeout, không chỉ mẫu thành công. Log generated/prompt tokens, steps, repair counts, peak GPU/RAM, audio/history lengths; CUDA synchronize khi đo từng phase để không thiếu thời gian async. Paired delta score/latency và số mẫu cứu/hỏng; attribution bằng gold-vs-ASR + trace, không tự gán mọi non-ID error cho ASR.

### 11.2. Ma trận thí nghiệm [P]

| ID | Thay đổi duy nhất so E0/reference | Câu hỏi |
|---|---|---|
| E0 | Qwen3 greedy + native template + allowlist + strict parser/validator + deadline | Baseline C thực tế là bao nhiêu? |
| E1 | + ledger/provenance/resolver | Giảm wrong-ID và read thừa không? |
| E2 | + schema-constrained generation | Giảm repair/invalid proposal, có tăng latency không? |
| E3 | + history/slot-preserving prompt và field-aware normalization | Cải thiện coreference/thời gian/phone không? |
| E4 | + tracker/dependency/stop guards | Giảm missing/repeated write không? |
| E5 | Full E1–E4 và leave-one-out từng thành phần | Có tương tác gây hại khi ghép không? |
| E6 | Transformers vs vLLM, cùng model/template/grammar/input | Backend nào đạt A30 latency, memory, reproducibility? |
| E7 | Cùng agent, thay ASR beam/VAD/normalizer từng yếu tố với B | Gain downstream có bù cost và model phụ không? |
| E8 | Optional LoRA/SFT train-only, chỉ mở gate sau E0–E5 | Có cải thiện lỗi còn lại đủ lớn để đáng rủi ro release không? |

Không ablate bỏ ràng buộc tuân thủ (env API/trace/6B/context). Dùng cùng dev IDs/model revision/seed/evaluator, lưu config và artifact hash cho từng run. Bắt đầu 50 mẫu dev chọn theo strata và seed trước khi xem output, sau đó toàn 265. Không claim gain trước đo; không dùng số giảm token/ROI từ tài liệu cũ làm chứng cứ.

Chọn cải tiến khi score dev tăng có paired evidence, không có regression compliance, max <=30s, P95 target <=25s và memory đủ. Nếu score bằng nhau chọn ít call thừa/latency thấp hơn. Không đặt điểm tuyệt đối tưởng tượng để tuyên bố hoàn thành. Nếu không có gain, giữ E0 đã kiểm chứng.

**SFT gate:** chỉ sau baseline có report, trajectory train hợp lệ được replay qua SDK, license/provenance đủ, không ăn vào đường Docker/release. Train chỉ 1157 IDs hoặc synthetic không dựa test; dev chỉ chọn cấu hình, holdout tách. Không train từ gold dev/holdout, public/private, không self-label test để few-shot/SFT. Nếu chưa đủ điều kiện ngày 10/11 thì bỏ SFT cho release này. Không RL/multi-agent/LLM verifier thứ hai.

## 12. Offline Docker và tích hợp A/B

**[P] A phụ trách release, C cung cấp artifact/model contract và integration tests; B phụ trách ASR/CT2/VAD.** Những việc sau cần làm trước 14/11:

1. Pin base image bằng digest và dependency lock/wheel hashes; cài torch/CUDA/CT2 tương thích, grammar offline. Không bắt cài flash-attn/vLLM nếu backend final không cần. Build có thể có mạng; **runtime phải hoàn toàn offline**. Nếu BTC yêu cầu build offline, chuẩn bị wheelhouse/base image sẵn.
2. Download đúng pinned revisions trong development, lưu Qwen shards/index/config/generation config/tokenizer/template/vocab/merges/LICENSE; ASR CT2 weights/config/tokenizer và VAD asset nếu dùng. Downloader fail nonzero khi artifact thiếu, checksum sai hoặc tải dở.
3. Bundle duy nhất model cấu hình final; không vô tình giữ default Qwen2.5 rồi fallback. Tắt network/cache fetch: HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE và local-only loader; test empty user cache. Tắt telemetry.
4. Xác nhận layout BTC: `/data` là root chứa split hay chính split; hỗ trợ rõ `--data-dir` và inject paths xuống catalog/linter/ToolEnv root. Không “lấy *.jsonl đầu tiên” không deterministic khi nhiều file. Không quét/copy environment để dò layout.
5. Output theo hướng dẫn BTC là `result/submission.jsonl`; chốt absolute working path/mount cùng A, ví dụ `/workspace/result/submission.jsonl` khi WORKDIR=/workspace. CMD cũ `/output/submission.jsonl` phải đổi sau khi xác minh. Kiểm entry quyền ghi, UTF-8 JSONL, đủ ID, chính xác ba khóa, hash SHA256 thật kể cả zero-call.
6. Runtime không import/call evaluator gold replay, state_diff hoặc training helpers. `.dockerignore` hiện loại toàn data/outputs; không phụ thuộc train catalog/splits bị thiếu. SDK từ mount theo hướng dẫn BTC, không tự patch SDK/scorer.
7. Inference/sample exception sau env init: giữ trace và hash của env hiện có trong finalization; nếu env init hoặc hash collection fail thì ghi diagnostic và đánh run fail, không fake hash để qua linter. Không tạo env mới để lấy initial hash.
8. Chạy image trên A30 với `--gpus device=0 --network none`, RAM/CPU tương đương máy chấm và data read-only, chỉ output mount ghi được. Data mount read-only vẫn cần xác minh SDK không yêu cầu ghi file; không chặn file read nội bộ hợp lệ của SDK bằng test nhầm.
9. Dry-run private-like có tên split khác, không có train folder, đủ audio/history/catalog/SDK; không lấy public/private thật để gán gold. Kiểm offline, trace, linter/path, 3 repeated runs; lưu image digest, code SHA, weight hashes, SDK/catalog/config hashes và hardware/driver.

## 13. File-specific tasks và người phụ trách

Các đường dẫn dưới đây là **kế hoạch thay đổi sau này**, không phải file đã sửa trong lần tạo tài liệu.

| Ưu tiên | File | Owner | Deliverable / acceptance |
|---|---|---|---|
| P0 | `src/part_c/agent.py` | C | Qwen3 loading local, compatible run, sequential tool loop, ledger/tracker, budget, bounded recovery; không tự tạo env |
| P0 | `src/part_c/prompt.py` | C | Native template-compatible messages, strict frame parser/terminal, bounded context, preserve history/observation evidence |
| P0 | `src/part_c/tool_filter.py` | C | Sorted exact-tool allowlist, schema converter giữ constraints, tool routing không bypass |
| P0 | `src/part_c/evidence_ledger.py` | C | Result adapters có fixture, provenance/ambiguity/freshness, sửa false match và unknown acceptance |
| P1 | `src/part_c/task_tracker.py` | C | Subgoals/dependencies, completion theo evidence, loop guards không chặn repeat hợp lệ |
| P0 | `src/part_a/validator.py` | A + C | Schema parity test; x-ref check ở grounding; inference dùng API được phép |
| P0 | `src/part_a/runner.py` | A + C | One-env contract, shared deadline, sanitize sample, no gold fallback official, faithful finalization |
| P0 | `scripts/run_offline.py` | A + B + C | Sửa ASR keyword/keys; Qwen path; không fallback/download; mount/output private-like; failure reporting |
| P0 | `scripts/generate_submission.py` | A + C | Chính thức không mặc định rule/gold; dùng cùng inference core thay vì ba loop sai khác |
| P0 | `src/part_a/submission_linter.py` | A | Paths injected, private/dev hỗ trợ, empty hash/schema lỗi critical, exit status; không false pass |
| P1 | `src/part_a/evaluator.py`, `scripts/evaluate.py` | A | Coverage/ID/scorer parity, raw hash stats; sửa diff keys/report; train-only diagnostics |
| P1 | `src/part_a/state_diff.py`, `loss_attribution.py`, `diagnostic.py` | A | Không trên inference import path; attribution có paired evidence, không báo gain heuristic như đo được |
| P0 | `src/part_a/data_splitter.py`, `outputs/splits/*` | A | Audit leakage/checksum/disjoint IDs; giữ split hiện tại trừ quyết định có version rõ |
| P0 | `src/part_b/asr_module.py` | B | ASRResult contract, pinned CT2/VAD, greedy/fallback controls, history ASR budget, local-only |
| P1 | `src/part_b/normalizer.py`, `asr_benchmark.py` | B + C | Field/slot preservation, paired downstream analysis; không biến missing negation thành điều chắc chắn |
| P0 | `scripts/download_models.py` | A + C + B | Pin Qwen3/upstream CT2/VAD, complete assets, manifest/checksum/license, fail-fast |
| P0 | `config.py`, `requirements*.txt`, `environment.yml` | A + C | Single-source model/budget/path config; exact compatible stack; optional backend rõ |
| P0 | `Dockerfile`, `.dockerignore` | A | Offline deterministic image, correct result path, no train dependency, one GPU tested |
| P1 | `notebooks/02_agent_colab_kaggle.ipynb` | C | Harness gọi module chuẩn, model mới, dev IDs chuẩn, không code agent bản thứ hai |
| P1 | `notebooks/00*`, `01*`, `README.md`, `scripts/verify_all.py` | A/B/C | Đồng bộ hướng dẫn/câu lệnh; loại khuyến nghị 7B, không gọi smoke là chứng nhận benchmark |
| P0 | **Mới:** `tests/test_agent_contract.py`, `test_env_compliance.py`, `test_grounding.py`, `test_offline_pipeline.py` | C + A | Bộ test mục 10, SDK fixtures riêng train-only |
| P1 | **Mới:** `src/part_c/structured_output.py` | C | Grammar compiler/adapter tách backend, schema cache và strict parser; chỉ tạo nếu tránh làm agent.py quá tải |
| P1 | **Mới:** `scripts/benchmark_part_c.py`, `scripts/audit_model_artifacts.py` | C + A/B | Gold-vs-ASR/ablations manifest và exact parameter/asset audit; không đọc environment |

## 14. Timeline 04–16/11/2026 và milestones

Lịch mới bắt đầu từ hiện trạng audit, không đánh dấu các mốc 03/11 trong MASTER_PLAN đã xong chỉ vì có trên giấy. Các acceptance là mục tiêu phải chứng minh.

| Ngày (UTC+7) | Việc / dependency | Milestone và tiêu chí nghiệm thu |
|---|---|---|
| **04/11** | Audit, xác nhận ràng buộc, thống nhất interfaces A/B/C; lấy SDK/data hợp lệ | **M0 draft:** list blockers/fixtures; không tuyên bố SDK pass khi chưa có |
| **05/11** | SDK contract smoke train, pinned model/license/parameter manifest; template/grammar spike | **M0/M1:** đúng env API, ASR contract pass; tổng manifest <6B, LLM <5B, Qwen local smoke một GPU |
| **06/11** | Qwen sequential baseline, strict parser/validator, shared deadline; 50 dev gold | **M2a:** không dùng labels trong C, faithful trace; có report thực, không chỉ replay |
| **07/11** | Chạy 265 dev gold và ASR + history với B | **M2:** paired report đầy đủ coverage, runtime/failure/token/memory; nếu chưa có A30 ghi rõ provisional |
| **08/11** | Ledger/result adapters + resolver/clarification; E1 | **M3a:** tests false-ID/ambiguity/stale pass; báo paired delta |
| **09/11** | Constrained decode + tracker/dependencies; E2/E4 | **M3b:** schema/terminal/dependency tests pass; không invalid proposal tới call trong fixtures |
| **10/11** | E3/E5 leave-one-out, backend spike E6, quyết định có SFT hay bỏ | **M3:** chọn cấu hình bằng dev evidence; không mở SFT nếu ảnh hưởng release |
| **11/11** | Tối ưu A30 batch1, E7 với B; context/audio/history stress | **M4a:** max <=30s trên suite, P95 target <=25s, memory trong 24GB; mọi spike có giải thích |
| **12/11** | Docker sạch, private-like mount, network none, artifact missing tests, strict linter | **M4:** không download/fallback/train dependency, output đúng, 0 release-critical errors |
| **13/11** | Chốt candidate trước mở holdout; đánh giá holdout 123 một lần | **M5a:** coverage đủ, báo chất lượng thật; không tune prompt bằng holdout |
| **14/11** | 3 independent runs cùng image trên A30, compliance audit và checksum | **M5 freeze:** trace/hash/JSONL identical, params/context/time/offline gates đều pass; tag/digest/manifest |
| **15/11** | Diễn tập command BTC, kiểm artifact/license/source package, A chuẩn bị nộp | **M6a:** release checklist có bằng chứng, không thay model/backend sát hạn |
| **16/11** | Nộp trước mốc nội bộ **18:00**, dự phòng đến cutoff repo ghi **23:59** | **M6:** A xác nhận nhận bài; C bàn giao reproducer, known limits, reports |

Nếu chậm: cắt SFT trước, rồi cắt tối ưu backend chưa chứng minh, giữ Qwen3 reference đã pass. Không cắt compliance/trace/SDK tests hay offline Docker để chạy đủ thí nghiệm. Nếu Qwen3 không đạt 30s: giảm prompt/output/read thừa, test backend/precision; mọi thay đổi phải đo lại trên A30. Không tự đổi sang model khác với lựa chọn của người dùng; báo blocker nếu cuối cùng vẫn không đạt.

## 15. Definition of Done và bàn giao cuối

- [ ] M0: SDK/caller/ASR/schema/trace/hash contract đã xác minh trên train, không chỉ mock.
- [ ] M1: đúng Qwen3 ID/revision; Apache-2.0 attribution; ASR/CT2/VAD provenance/license; đếm toàn hệ thống thực <6B, LLM <5B, không dùng dung lượng quantization làm số params.
- [ ] M2/M3: 265 dev gold-vs-ASR có kết quả thực và ablations; output đủ IDs; không ngụy nhận gold replay 100 là agent 100.
- [ ] Không direct environment access, không clone/replay env trong inference; exactly one env/sample/run; chỉ bốn API được phép; mọi call đã chạy đều có trong submission.
- [ ] Không ID ảo hoặc chọn ngẫu nhiên ở fixture ambiguity; không unknown/stale precondition; args/opaque IDs được bảo toàn.
- [ ] Ask/OOS/stop đa bước có kiểm thử; không retry write mù, không dừng sau write đầu khi còn việc.
- [ ] Context từng lượt prompt+output <=16.384; full E2E suite trên **1 A30** max <=30s, P95 target <=25s; không OOM/crash; failures được tính và report.
- [ ] Offline image network none/cache trống, `/data` layout và `result/submission.jsonl` đúng hướng dẫn BTC đã xác nhận; fail khi thiếu model; linter strict 0 critical.
- [ ] Ba repeated runs cùng image/input trên A30 cho trace/hash/JSONL giống hệt; seeds/kernel/runtime/assets được khóa.
- [ ] Không nhãn public/private, không gold-test fallback, không train/tune bằng test; holdout một lần có audit.
- [ ] Bàn giao source SHA/image digest/weights-SDK-config hashes, exact reproduce command, model manifest, reports, known failures và người phụ trách.

**Trạng thái tại lúc lập plan:** hoàn thành audit tĩnh và đối chiếu nguồn/parameter arithmetic; chưa chạy Qwen, ASR, ToolEnv, Docker hay benchmark A30. Các ô trên để trống có chủ ý. Những việc cần lấy ngay để triển khai: SDK/catalog/train audio hợp lệ, pinned weights/CT2/VAD assets, máy A30 và hướng dẫn mount/entry/output chính thức.
