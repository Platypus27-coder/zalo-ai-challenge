"""Part C — Prompt Templates: System prompts, user prompts, and protocol parser for Qwen3-4B-Instruct.

Supports:
1. Native Qwen3 function calling format: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
2. Legacy text protocol: CALL: {"name": "...", "arguments": {...}} / DONE
3. Function-call notation: [Bước X:] tool_name({...})
4. Strict parsing & terminal action identification with nested JSON support.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


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
    """Build the user prompt for the Agent LLM."""
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
        if hist_text:
            parts.append(f"## LỊCH SỬ HỘI THOẠI\n{hist_text}")
    
    # Available tools
    parts.append(f"## CÔNG CỤ KHẢ DỤNG\n{tools_text}")
    
    # Current request
    parts.append(f"## YÊU CẦU HIỆN TẠI\n{transcript if transcript else '[Yêu cầu trống]'}")
    
    # Previous observations in this session
    if observations:
        obs_text = format_observations(observations)
        parts.append(f"## KẾT QUẢ CÁC BƯỚC TRƯỚC\n{obs_text}")
        parts.append("Dựa trên kết quả trên, hãy quyết định bước tiếp theo hoặc trả về DONE nếu đã xong.")
    else:
        parts.append("Hãy phân tích yêu cầu và thực hiện bước đầu tiên.")
    
    return "\n\n".join(parts)


def format_history(history: list[dict]) -> str:
    """Format conversation history concisely."""
    lines = []
    for turn in history:
        role = turn.get("role", "unknown")
        if role == "user":
            text = turn.get("audio_text", turn.get("text", "[audio]"))
            lines.append(f"User: {text}")
        elif role == "assistant":
            text = turn.get("text", "")
            if turn.get("tool_calls"):
                calls = ", ".join(tc["name"] for tc in turn["tool_calls"] if isinstance(tc, dict) and "name" in tc)
                lines.append(f"Assistant: [gọi: {calls}] {text}".strip())
            elif text:
                lines.append(f"Assistant: {text}")
    return "\n".join(lines)


def format_observations(observations: list[dict], max_obs_len: int = 600) -> str:
    """Format previous tool observations concisely."""
    lines = []
    for i, obs in enumerate(observations):
        call = obs.get("call", {})
        result = obs.get("result", {})
        
        name = call.get("name", "unknown")
        args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
        
        if result.get("ok"):
            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > max_obs_len:
                result_str = result_str[:max_obs_len] + "... [dữ liệu dài đã rút gọn]"
            lines.append(f"Bước {i+1}: {name}({args}) → {result_str}")
        else:
            error = result.get("error", "unknown error")
            lines.append(f"Bước {i+1}: {name}({args}) → LỖI: {error}")
    
    return "\n".join(lines)


def _extract_first_json_object(text: str, start_pos: int = 0) -> Optional[dict]:
    """Robustly extract the first valid JSON object handling arbitrary nesting and escaped quotes."""
    idx = text.find("{", start_pos)
    if idx == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    start_idx = idx
    for i in range(idx, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start_idx:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def parse_agent_response(response: str) -> dict[str, Any]:
    """Parse agent's response to extract tool call or done signal.
    
    Robust parser handling:
    - DONE / [] / finish signals / <done>
    - CALL: {"name": ..., "arguments": ...}
    - Markdown code fences (```json ... ```)
    - Qwen native <tool_call> ... </tool_call>
    - Function call style: [Bước X:] tool_name({...})
    - JSON object or JSON list of tool calls
    
    Returns:
        {"action": "call", "name": str, "arguments": dict}
        {"action": "done"}
        {"action": "error", "message": str}
    """
    if not response:
        return {"action": "done"}

    cleaned = response.strip()
    
    # 1. Check for DONE or empty tool calls
    if (
        cleaned.upper().startswith("DONE")
        or cleaned.upper() == "DONE"
        or cleaned == "[]"
        or cleaned == "<done>"
        or "KHÔNG CẦN GỌI TOOL" in cleaned.upper()
    ):
        return {"action": "done"}
    
    # 2. Extract Qwen native <tool_call> tags
    tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', cleaned, re.DOTALL)
    if tool_call_match:
        content = tool_call_match.group(1).strip()
        obj = _extract_first_json_object(content)
        if isinstance(obj, dict) and "name" in obj:
            return {
                "action": "call",
                "name": str(obj["name"]).strip(),
                "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
            }
    
    # 3. Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    fence_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned, re.DOTALL)
    if fence_match:
        fenced_content = fence_match.group(1).strip()
        obj = _extract_first_json_object(fenced_content)
        if isinstance(obj, dict) and "name" in obj:
            return {
                "action": "call",
                "name": str(obj["name"]).strip(),
                "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
            }
        cleaned = fenced_content
    
    # 4. Check for CALL: prefix
    if "CALL:" in cleaned:
        call_pos = cleaned.find("CALL:")
        obj = _extract_first_json_object(cleaned, call_pos)
        if isinstance(obj, dict) and "name" in obj:
            return {
                "action": "call",
                "name": str(obj["name"]).strip(),
                "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
            }
    
    # 5. Try to find function-call style: [Bước \d+:] app.tool_name({...})
    fn_match = re.search(r'([a-z_]+\.[a-z_]+)\s*\(\s*(\{.*?\})\s*\)', cleaned, re.DOTALL)
    if fn_match:
        fn_name = fn_match.group(1).strip()
        fn_args_str = fn_match.group(2).strip()
        obj = _extract_first_json_object(fn_args_str)
        if isinstance(obj, dict):
            return {
                "action": "call",
                "name": fn_name,
                "arguments": obj,
            }
    
    # 6. Try finding any JSON object with "name" and "arguments"
    obj = _extract_first_json_object(cleaned)
    if isinstance(obj, dict) and "name" in obj:
        return {
            "action": "call",
            "name": str(obj["name"]).strip(),
            "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
        }
    
    # 7. Check for JSON list
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            if len(data) == 0:
                return {"action": "done"}
            if isinstance(data[0], dict) and "name" in data[0]:
                return {
                    "action": "call",
                    "name": str(data[0]["name"]).strip(),
                    "arguments": data[0].get("arguments") if isinstance(data[0].get("arguments"), dict) else {},
                }
    except json.JSONDecodeError:
        pass
    
    # 8. Standalone DONE word anywhere in the response
    if re.search(r'\bDONE\b', cleaned):
        return {"action": "done"}
    
    return {"action": "error", "message": f"Cannot parse response: {cleaned[:200]}"}


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("=== Prompt Template Test ===\n")
    print(f"System Prompt Length: {len(SYSTEM_PROMPT)} chars\n")
    
    tests = [
        'CALL: {"name": "smart_home.get_status", "arguments": {}}',
        'DONE',
        '<done>',
        'Bước 4: smart_home.climate_set({"device_id": "ac_living", "mode": "heat", "temperature": 25})',
        'Tôi sẽ gọi: alarm.create({"time": "2026-11-03T07:00:00+07:00"})',
        '```json\n{"name": "contacts.search", "arguments": {"query": "Mẹ"}}\n```',
        '<tool_call>\n{"name": "sms.send", "arguments": {"phone_number": "0912345678", "message": "Alo"}}\n</tool_call>',
        '[{"name": "weather.get_forecast", "arguments": {"location": "Hà Nội"}}]',
        'invalid response',
    ]
    for t in tests:
        result = parse_agent_response(t)
        print(f"  Input:  {t[:75]}")
        print(f"  Parsed: {result}\n")
