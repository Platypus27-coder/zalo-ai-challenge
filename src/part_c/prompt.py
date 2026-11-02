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
2. Nếu cần ID (device_id, contact_id, event_id...), PHẢI gọi tool đọc (read) trước để lấy ID thật. KHÔNG ĐƯỢC tự bịa ID.
3. Nếu thiếu thông tin bắt buộc mà không thể suy luận từ ngữ cảnh, gọi sys.ask_missing_info.
4. Nếu không có tool nào phù hợp, trả về DONE (không gọi tool nào).
5. Thời gian "mai" = ngày tiếp theo so với current_time. Dùng timezone của mẫu.
6. Giữ nguyên số tiền, số điện thoại, tên riêng từ yêu cầu. Không làm tròn, không sửa.
7. Chú ý câu phủ định: "không", "đừng", "thôi", "bỏ", "hủy" thay đổi ý nghĩa yêu cầu.

## ĐỊNH DẠNG TRẢ LỜI
Mỗi bước trả lời ĐÚNG MỘT trong hai dạng:
- Gọi tool: CALL: {"name": "<tool_name>", "arguments": {<params>}}
- Kết thúc: DONE

Sau khi gọi tool, bạn sẽ nhận observation. Dùng observation để quyết định bước tiếp theo."""


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
    
    Expected formats:
    - CALL: {"name": "tool.name", "arguments": {...}}
    - DONE
    
    Returns:
        {"action": "call", "name": str, "arguments": dict}
        {"action": "done"}
        {"action": "error", "message": str}
    """
    response = response.strip()
    
    # Check for DONE
    if response.upper().startswith("DONE") or response.upper() == "DONE":
        return {"action": "done"}
    
    # Try to extract CALL: {...}
    import re
    
    # Pattern: CALL: followed by JSON
    call_match = re.search(r'CALL:\s*(\{.*\})', response, re.DOTALL)
    if call_match:
        try:
            call_data = json.loads(call_match.group(1))
            return {
                "action": "call",
                "name": call_data.get("name", ""),
                "arguments": call_data.get("arguments", {}),
            }
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON with "name" and "arguments"
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}', response)
    if json_match:
        try:
            call_data = json.loads(json_match.group(0))
            return {
                "action": "call",
                "name": call_data.get("name", ""),
                "arguments": call_data.get("arguments", {}),
            }
        except json.JSONDecodeError:
            pass
    
    # Try to parse entire response as JSON
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "name" in data:
            return {
                "action": "call",
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {}),
            }
    except json.JSONDecodeError:
        pass
    
    return {"action": "error", "message": f"Cannot parse response: {response[:200]}"}


# Keep json available for parse_agent_response
import json


if __name__ == "__main__":
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
        'invalid response',
    ]
    for t in tests:
        result = parse_agent_response(t)
        print(f"  Input:  {t[:80]}")
        print(f"  Parsed: {result}\n")
