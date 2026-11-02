"""Part C — Tool Filter: Filter and format tool schemas for the Agent.

Responsibilities:
- Filter catalog tools by sample's installed_apps
- Format tool schemas into LLM-friendly descriptions
- Always include sys.ask_missing_info
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TRAIN_CATALOG, PUBLIC_TEST_CATALOG


def load_catalog(catalog_path: Path) -> dict:
    """Load tools catalog."""
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def get_all_tool_specs(catalog: dict) -> dict[str, dict]:
    """Extract {tool_name: tool_spec} from catalog."""
    specs = {}
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            specs[tool["name"]] = tool
    return specs


def filter_tools_for_sample(
    sample: dict, 
    all_tool_specs: dict[str, dict]
) -> dict[str, dict]:
    """Filter tools to only those installed in the sample.
    
    Returns {tool_name: tool_spec} for allowed tools.
    Always includes sys.ask_missing_info.
    """
    installed_apps = sample.get("context", {}).get("installed_apps", {})
    allowed_tools = set()
    
    for app_name, app_data in installed_apps.items():
        for tool_name in app_data.get("tools", []):
            allowed_tools.add(tool_name)
    
    allowed_tools.add("sys.ask_missing_info")
    
    filtered = {
        name: spec 
        for name, spec in all_tool_specs.items() 
        if name in allowed_tools
    }
    
    return filtered


def format_tool_for_prompt(tool_spec: dict) -> str:
    """Format a single tool spec into a concise LLM-readable description."""
    name = tool_spec["name"]
    kind = tool_spec.get("kind", "unknown")
    desc = tool_spec.get("description", "")
    
    params_spec = tool_spec.get("parameters", {})
    properties = params_spec.get("properties", {})
    required = params_spec.get("required", [])
    x_one_of = params_spec.get("x-one-of", [])
    
    # Format parameters
    param_lines = []
    for pname, pspec in properties.items():
        ptype = pspec.get("type", "any")
        pdesc = pspec.get("description", "")
        is_req = pname in required
        
        parts = [f"  - {pname}: {ptype}"]
        if is_req:
            parts.append("(BẮT BUỘC)")
        if pspec.get("enum"):
            parts.append(f"[{', '.join(str(e) for e in pspec['enum'])}]")
        if pspec.get("format"):
            parts.append(f"format: {pspec['format']}")
        if pspec.get("x-ref"):
            parts.append(f"(ID từ {pspec['x-ref']})")
        if pspec.get("x-match") == "fuzzy":
            parts.append("(văn bản tự do)")
        if pdesc:
            parts.append(f"— {pdesc}")
        if pspec.get("minimum") is not None:
            parts.append(f"min={pspec['minimum']}")
        if pspec.get("maximum") is not None:
            parts.append(f"max={pspec['maximum']}")
        
        param_lines.append(" ".join(parts))
    
    # Format x-one-of
    oneof_str = ""
    if x_one_of:
        groups = " HOẶC ".join(" + ".join(g) for g in x_one_of)
        oneof_str = f"\n  Cần một trong: {groups}"
    
    # Build output
    kind_label = {"read": "ĐỌC", "write": "GHI", "meta": "HỆ THỐNG"}.get(kind, kind)
    
    result = f"[{kind_label}] {name}: {desc}"
    if param_lines:
        result += "\n" + "\n".join(param_lines)
    if oneof_str:
        result += oneof_str
    
    return result


def format_tools_for_prompt(
    tool_specs: dict[str, dict],
    include_kind: bool = True,
    group_by_app: bool = True
) -> str:
    """Format all tools into a prompt-ready string.
    
    Args:
        tool_specs: {tool_name: tool_spec}
        include_kind: Include read/write/meta labels
        group_by_app: Group tools by app
    """
    if group_by_app:
        # Group by app
        apps = {}
        for name, spec in tool_specs.items():
            app = spec.get("app", name.split(".")[0])
            apps.setdefault(app, []).append(spec)
        
        sections = []
        for app_name in sorted(apps.keys()):
            tools = apps[app_name]
            tool_strs = [format_tool_for_prompt(t) for t in sorted(tools, key=lambda t: t["name"])]
            sections.append(f"## {app_name}\n" + "\n\n".join(tool_strs))
        
        return "\n\n".join(sections)
    else:
        tool_strs = [format_tool_for_prompt(spec) for spec in sorted(tool_specs.values(), key=lambda t: t["name"])]
        return "\n\n".join(tool_strs)


def format_tools_as_json_schema(tool_specs: dict[str, dict]) -> list[dict]:
    """Format tools as JSON function-calling schema (for models supporting it)."""
    functions = []
    for name, spec in sorted(tool_specs.items()):
        func = {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
            }
        }
        functions.append(func)
    return functions


if __name__ == "__main__":
    # Test tool filtering
    print("=== Tool Filter Test ===\n")
    
    catalog = load_catalog(TRAIN_CATALOG)
    all_specs = get_all_tool_specs(catalog)
    print(f"Total tools in catalog: {len(all_specs)}")
    
    # Simulate a sample with limited apps
    sample = {
        "context": {
            "installed_apps": {
                "smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set"]},
                "alarm": {"tools": ["alarm.list", "alarm.create"]},
            }
        }
    }
    
    filtered = filter_tools_for_sample(sample, all_specs)
    print(f"Filtered tools for sample: {len(filtered)}")
    
    print(f"\nFormatted tools for prompt:\n")
    print(format_tools_for_prompt(filtered))
