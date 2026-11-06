"""Part C — Prompt Templates: System prompts, user prompts, and protocol parser for Qwen3-4B-Instruct.

Supports:
1. Native Qwen3 function calling format: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
2. Legacy text protocol: CALL: {"name": "...", "arguments": {...}} / DONE
3. Strict parsing & terminal action identification with nested JSON support.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


SYSTEM_PROMPT = """Bạn là trợ lý AI xử lý yêu cầu bằng giọng nói tiếng Việt cho hệ thống LifeMate.
Nhiệm vụ: phân tích yêu cầu của người dùng và gọi các công cụ (tool) phù hợp một cách tuần tự.

## QUY TẮC BẮT BUỘC:
1. CHỈ gọi các tool có trong danh sách công cụ được cung cấp (thuộc app đã cài đặt).
2. THỨ TỰ THỰC THI (Read -> Write): Nếu cần ID (device_id, contact_id, event_id, alarm_id, v.v.), BẮT BUỘC phải gọi tool đọc (read/get/search/list) để lấy ID thật trước. TUYỆT ĐỐI KHÔNG tự bịa ID.
3. HỎI THÔNG TIN THIẾU: Nếu người dùng chưa cung cấp đủ tham số bắt buộc và không thể tra cứu từ hệ thống, hãy gọi tool `sys.ask_missing_info` với `tool_name` dự định và danh sách `params` bị thiếu. Sau khi gọi tool này, dừng lại ngay.
4. NGOÀI PHẠM VI (Out-of-Scope): Nếu yêu cầu thuộc về ứng dụng chưa được cài đặt, hoặc không có tool nào phù hợp, trả về `DONE` ngay lập tức mà không gọi tool nào.
5. THỜI GIAN & ĐỊA ĐIỂM:
   - Căn cứ vào `current_time` và `timezone` được cung cấp trong ngữ cảnh.
   - Giữ nguyên số tiền, số điện thoại, tên riêng, không làm tròn số và không sửa chính tả tên riêng.
6. CÂU PHỦ ĐỊNH: Chú ý từ ngữ phủ định hoặc hủy lệnh ("không", "đừng", "thôi", "hủy", "bỏ qua") làm thay đổi hoàn toàn ý định.

## GIAO THỨC TRẢ LỜI:
Mỗi lượt suy luận, bạn CHỈ ĐƯỢC thực hiện ĐÚNG MỘT hành động:
- Gọi tool:
<tool_call>
{"name": "<tool_name>", "arguments": {<tham_số>}}
</tool_call>
- Hoặc kết thúc khi đã hoàn thành hoặc ngoài phạm vi:
DONE
"""


def build_user_prompt(
    transcript: str,
    context: dict,
    history: list[dict],
    tools_text: str,
    observations: list[dict] | None = None,
) -> str:
    """Build a unified user prompt incorporating context, history, tools, and observations."""
    parts = []
    
    # 1. Temporal & Environmental Context
    current_time = context.get("current_time", "")
    timezone = context.get("timezone", "Asia/Ho_Chi_Minh")
    user_profile = context.get("user_profile", {})
    
    parts.append(f"## THỜI GIAN\nHiện tại: {current_time} ({timezone})")
    
    if user_profile:
        profile_str = ", ".join(f"{k}: {v}" for k, v in user_profile.items() if v is not None)
        if profile_str:
            parts.append(f"## THÔNG TIN NGƯỜI DÙNG\n{profile_str}")
    
    if context.get("location"):
        loc = context["location"]
        parts.append(f"## VỊ TRÍ\nlat={loc.get('lat')}, lng={loc.get('lng')}")
    
    # 2. History
    if history:
        hist_text = format_history(history)
        if hist_text:
            parts.append(f"## LỊCH SỬ HỘI THOẠI\n{hist_text}")
    
    # 3. Available Tools
    if tools_text:
        parts.append(f"## CÔNG CỤ ĐƯỢC PHÉP DÙNG\n{tools_text}")
    
    # 4. Current Request
    parts.append(f"## YÊU CẦU CẦN XỬ LÝ\n{transcript if transcript else '[Yêu cầu trống]'}")
    
    # 5. Previous observations in this session
    if observations:
        obs_text = format_observations(observations)
        parts.append(f"## QUAN SÁT TỪ CÁC BƯỚC TRƯỚC\n{obs_text}")
        parts.append("Dựa trên các quan sát trên, hãy thực hiện bước tiếp theo hoặc trả về DONE nếu đã xong.")
    else:
        parts.append("Hãy phân tích yêu cầu và gọi tool đầu tiên (hoặc DONE nếu ngoài phạm vi).")
    
    return "\n\n".join(parts)


def format_history(history: list[dict]) -> str:
    """Format conversation history concisely without leaking unrelated internal data."""
    lines = []
    for turn in history:
        role = turn.get("role", "unknown")
        if role == "user":
            text = turn.get("audio_text") or turn.get("text") or "[audio]"
            lines.append(f"User: {text}")
        elif role == "assistant":
            text = turn.get("text", "")
            if turn.get("tool_calls"):
                calls_str = ", ".join(tc.get("name", "") for tc in turn["tool_calls"] if isinstance(tc, dict))
                lines.append(f"Assistant: [Đã gọi: {calls_str}] {text}".strip())
            elif text:
                lines.append(f"Assistant: {text}")
    return "\n".join(lines)


def format_observations(observations: list[dict], max_obs_len: int = 800) -> str:
    """Format previous tool observations safely, preserving IDs and error details."""
    lines = []
    for i, obs in enumerate(observations):
        call = obs.get("call", {})
        result = obs.get("result", {})
        name = call.get("name", "unknown")
        args_str = json.dumps(call.get("arguments", {}), ensure_ascii=False)
        
        if result.get("ok"):
            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > max_obs_len:
                result_str = result_str[:max_obs_len] + "... [dữ liệu dài đã rút gọn]"
            lines.append(f"Bước {i+1}: {name}({args_str}) -> Thành công: {result_str}")
        else:
            error_msg = result.get("error", "Lỗi không xác định")
            lines.append(f"Bước {i+1}: {name}({args_str}) -> LỖI: {error_msg}")
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
    """Parse agent's LLM response strictly.
    
    Recognizes:
    - Native format: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
    - CALL protocol: CALL: {"name": "...", "arguments": {...}}
    - Terminal: DONE or <done>
    
    Returns:
        {"action": "call", "name": str, "arguments": dict}
        {"action": "done"}
        {"action": "error", "message": str}
    """
    if not response:
        return {"action": "done"}
    
    text = response.strip()
    
    # 1. Check for explicit completion / OOS signal
    if text.upper() == "DONE" or text.upper().startswith("DONE") or text == "<done>":
        return {"action": "done"}
    
    # 2. Check for native Qwen3 format: <tool_call> ... </tool_call>
    tool_call_start = text.find("<tool_call>")
    if tool_call_start != -1:
        obj = _extract_first_json_object(text, tool_call_start)
        if isinstance(obj, dict) and "name" in obj:
            return {
                "action": "call",
                "name": str(obj["name"]).strip(),
                "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
            }
            
    # 3. Check for CALL: {...} protocol
    call_pos = text.find("CALL:")
    if call_pos != -1:
        obj = _extract_first_json_object(text, call_pos)
        if isinstance(obj, dict) and "name" in obj:
            return {
                "action": "call",
                "name": str(obj["name"]).strip(),
                "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
            }

    # 4. Check for direct JSON block: ```json { ... } ``` or raw {...}
    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block:
        obj = _extract_first_json_object(json_block.group(1))
        if isinstance(obj, dict) and "name" in obj:
            return {
                "action": "call",
                "name": str(obj["name"]).strip(),
                "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
            }

    # 5. Fallback: Find any first valid JSON object in text
    obj = _extract_first_json_object(text)
    if isinstance(obj, dict) and "name" in obj:
        return {
            "action": "call",
            "name": str(obj["name"]).strip(),
            "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
        }

    # Standalone DONE word
    if re.search(r"\bDONE\b", text):
        return {"action": "done"}

    return {
        "action": "error",
        "message": f"Không thể trích xuất tool call hợp lệ từ response: {text[:200]}"
    }
