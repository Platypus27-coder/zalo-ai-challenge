"""Part C — Prompt Templates: System and user prompts for the Agent LLM.

Design principles:
- Concise: minimize token usage (16K context limit)
- Action-oriented: Agent is judged on actions, not words
- Vietnamese-aware: handle Vietnamese time, names, addresses
- Structured output: guide LLM to produce parseable tool calls
"""


SYSTEM_PROMPT = """Bạn là trợ lý AI xử lý yêu cầu bằng giọng nói tiếng Việt. Nhiệm vụ: phân tích yêu cầu và gọi các công cụ (tool) phù hợp.

## QUY TẮC BẮT BUỘC
1. CHỈ gọi tool có trong danh sách được cung cấp.
2. Nếu cần ID (device_id, contact_id, event_id...), PHẢI gọi tool đọc (read/list/search) trước để lấy ID thật từ observation. KHÔNG ĐƯỢC tự bịa ID.
3. Khi người dùng muốn thực hiện một hành động nhưng thiếu tham số bắt buộc không thể suy ra từ ngữ cảnh hay observation, gọi sys.ask_missing_info với định dạng: {"tool_name": "<tên_tool>", "params": ["<tên_tham_số_thiếu>"]}.
4. Nếu không có tool nào phù hợp để thực hiện yêu cầu (câu chào hỏi, tán gẫu, hoặc ngoài phạm vi), trả về ngay DONE (không gọi tool).
5. Thời gian: Luôn tính toán dựa trên current_time và múi giờ được cung cấp. Xuất thời gian tuyệt đối ISO 8601 (ví dụ: YYYY-MM-DDTHH:MM:SS+07:00). "mai" = ngày tiếp theo so với current_time.
6. Giữ nguyên số tiền, số điện thoại, tên riêng từ yêu cầu. Không làm tròn, không sửa.
7. Chú ý câu phủ định: "không", "đừng", "thôi", "bỏ", "hủy" thay đổi ý nghĩa yêu cầu.

## ĐỊNH DẠNG TRẢ LỜI
Mỗi bước trả lời ĐÚNG MỘT trong hai dạng:
- Gọi tool: CALL: {"name": "<tool_name>", "arguments": {<params>}}
- Kết thúc: DONE

Sau khi gọi tool, bạn sẽ nhận observation. Dùng observation để quyết định bước tiếp theo.

## VÍ DỤ MINH HỌA
Ví dụ 1 (Đọc trước để lấy ID rồi mới thực hiện):
User: "Bật đèn phòng khách"
Assistant: CALL: {"name": "smart_home.get_status", "arguments": {}}
Observation: [{"id": "dev_01", "name": "Đèn phòng khách", "state": "off"}]
Assistant: CALL: {"name": "smart_home.light_set", "arguments": {"device_id": "dev_01", "state": "on"}}
Observation: {"ok": true}
Assistant: DONE

Ví dụ 2 (Tính thời gian tuyệt đối từ current_time):
Current_time: 2026-11-02T15:00:00+07:00
User: "Đặt báo thức 7 giờ sáng mai"
Assistant: CALL: {"name": "alarm.create", "arguments": {"time": "2026-11-03T07:00:00+07:00"}}
Observation: {"ok": true, "alarm_id": "alm_12"}
Assistant: DONE

Ví dụ 3 (Báo thiếu tham số bắt buộc với sys.ask_missing_info):
User: "Gửi tin nhắn cho mẹ"
Assistant: CALL: {"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["content"]}}

Ví dụ 4 (Ngoài phạm vi / OOS):
User: "Hôm nay tôi mệt quá"
Assistant: DONE"""


def build_user_prompt(
    transcript: str,
    context: dict,
    history: list[dict],
    tools_text: str,
    observations: list[dict] | None = None,
) -> str:
    """Build the user prompt for the Agent LLM.
    
    Args:
        transcript: ASR transcript of user request
        context: Sample context (current_time, user_profile, etc.)
        history: Conversation history
        tools_text: Formatted tool descriptions
        observations: Previous tool call results in this session
    """
    parts = []
    
    # Context
    current_time = context.get("current_time", "")
    timezone = context.get("timezone", "Asia/Ho_Chi_Minh")
    user_profile = context.get("user_profile", {})
    
    parts.append(f"## THỜI GIAN\nHiện tại: {current_time} ({timezone})")
    
    if user_profile:
        profile_str = ", ".join(f"{k}: {v}" for k, v in user_profile.items() if v)
        parts.append(f"## NGƯỜI DÙNG\n{profile_str}")
    
    if context.get("location"):
        loc = context["location"]
        parts.append(f"## VỊ TRÍ\nlat={loc.get('lat')}, lng={loc.get('lng')}")
    
    # History
    if history:
        hist_text = format_history(history)
        parts.append(f"## LỊCH SỬ HỘI THOẠI\n{hist_text}")
    
    # Available tools
    parts.append(f"## CÔNG CỤ KHẢ DỤNG\n{tools_text}")
    
    # Current request
    parts.append(f"## YÊU CẦU HIỆN TẠI\n{transcript}")
    
    # Previous observations in this session
    if observations:
        obs_text = format_observations(observations)
        parts.append(f"## KẾT QUẢ CÁC BƯỚC TRƯỚC\n{obs_text}")
        parts.append("Dựa trên kết quả trên, hãy quyết định bước tiếp theo.")
    else:
        parts.append("Hãy phân tích yêu cầu và thực hiện bước đầu tiên.")
    
    return "\n\n".join(parts)


def format_history(history: list[dict]) -> str:
    """Format conversation history concisely."""
    lines = []
    for i, turn in enumerate(history):
        role = turn.get("role", "unknown")
        if role == "user":
            text = turn.get("audio_text", turn.get("text", "[audio]"))
            lines.append(f"User: {text}")
        elif role == "assistant":
            text = turn.get("text", "")
            if turn.get("tool_calls"):
                calls = ", ".join(tc["name"] for tc in turn["tool_calls"])
                lines.append(f"Assistant: [gọi: {calls}] {text}")
            else:
                lines.append(f"Assistant: {text}")
    
    return "\n".join(lines)


def format_observations(observations: list[dict]) -> str:
    """Format previous tool observations concisely."""
    lines = []
    for i, obs in enumerate(observations):
        call = obs.get("call", {})
        result = obs.get("result", {})
        
        name = call.get("name", "unknown")
        args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
        
        if result.get("ok"):
            # Truncate long results
            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            lines.append(f"Bước {i+1}: {name}({args}) → {result_str}")
        else:
            error = result.get("error", "unknown error")
            lines.append(f"Bước {i+1}: {name}({args}) → LỖI: {error}")
    
    return "\n".join(lines)


def parse_agent_response(response: str) -> dict:
    """Parse agent's response to extract tool call or done signal.
    
    Robust parser handling:
    - DONE / [] / finish signals
    - CALL: {"name": ..., "arguments": ...}
    - Markdown code fences (```json ... ```)
    - Qwen native <tool_call> ... </tool_call>
    - JSON object or JSON list of tool calls
    
    Returns:
        {"action": "call", "name": str, "arguments": dict}
        {"action": "done"}
        {"action": "error", "message": str}
    """
    import re
    response = response.strip()
    
    # 1. Check for DONE or empty tool calls
    if (
        response.upper().startswith("DONE")
        or response.upper() == "DONE"
        or response == "[]"
        or "KHÔNG CẦN GỌI TOOL" in response.upper()
    ):
        return {"action": "done"}
    
    # 2. Extract Qwen native <tool_call> tags
    tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
    if tool_call_match:
        content = tool_call_match.group(1).strip()
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "name" in data:
                return {
                    "action": "call",
                    "name": data.get("name", ""),
                    "arguments": data.get("arguments", {}),
                }
        except json.JSONDecodeError:
            pass
    
    # 3. Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    cleaned = response
    fence_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    
    # 4. Check for CALL: prefix
    if "CALL:" in cleaned:
        call_part = cleaned.split("CALL:", 1)[1].strip()
        # Find first { and try to parse
        first_brace = call_part.find("{")
        if first_brace != -1:
            # Try to find matching closing brace or parse progressively
            for end_idx in range(len(call_part), first_brace, -1):
                try:
                    data = json.loads(call_part[first_brace:end_idx])
                    if isinstance(data, dict) and "name" in data:
                        return {
                            "action": "call",
                            "name": data.get("name", ""),
                            "arguments": data.get("arguments", {}),
                        }
                except json.JSONDecodeError:
                    continue
    
    # 5. Try to find any JSON object with "name" and "arguments"
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}', cleaned)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return {
                "action": "call",
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {}),
            }
        except json.JSONDecodeError:
            pass

    # 5.5. Match function-call style: [Bước \d+:] app.tool_name({...})
    fn_match = re.search(r'([a-z_]+\.[a-z_]+)\s*\(\s*(\{.*?\})\s*\)', cleaned, re.DOTALL)
    if fn_match:
        fn_name = fn_match.group(1).strip()
        fn_args_str = fn_match.group(2).strip()
        try:
            fn_args = json.loads(fn_args_str)
            if isinstance(fn_args, dict):
                return {
                    "action": "call",
                    "name": fn_name,
                    "arguments": fn_args,
                }
        except json.JSONDecodeError:
            first_b = fn_args_str.find("{")
            if first_b != -1:
                for end_i in range(len(fn_args_str), first_b, -1):
                    try:
                        data = json.loads(fn_args_str[first_b:end_i])
                        if isinstance(data, dict):
                            return {
                                "action": "call",
                                "name": fn_name,
                                "arguments": data,
                            }
                    except json.JSONDecodeError:
                        continue
    
    # 6. Try parsing the entire cleaned text as JSON (dict or list)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "name" in data:
            return {
                "action": "call",
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {}),
            }
        elif isinstance(data, list):
            if len(data) == 0:
                return {"action": "done"}
            if isinstance(data[0], dict) and "name" in data[0]:
                return {
                    "action": "call",
                    "name": data[0].get("name", ""),
                    "arguments": data[0].get("arguments", {}),
                }
    except json.JSONDecodeError:
        pass
    
    return {"action": "error", "message": f"Cannot parse response: {response[:200]}"}


# Keep json available for parse_agent_response
import json


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("=== Prompt Template Test ===\n")
    print("System Prompt:")
    print(SYSTEM_PROMPT)
    print(f"\n--- ({len(SYSTEM_PROMPT)} chars) ---\n")
    
    # Test parsing
    print("Parse Tests:")
    tests = [
        'CALL: {"name": "smart_home.get_status", "arguments": {}}',
        'DONE',
        'Tôi sẽ gọi tool:\nCALL: {"name": "alarm.create", "arguments": {"time": "2026-11-03T07:00:00+07:00"}}',
        '```json\n{"name": "contacts.search", "arguments": {"query": "Mẹ"}}\n```',
        '<tool_call>\n{"name": "sms.send", "arguments": {"phone_number": "0912345678", "message": "Alo"}}\n</tool_call>',
        '[{"name": "weather.get_forecast", "arguments": {"location": "Hà Nội"}}]',
        'invalid response',
    ]
    for t in tests:
        result = parse_agent_response(t)
        print(f"  Input:  {t[:80]}")
        print(f"  Parsed: {result}\n")
