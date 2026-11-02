"""Part A — Pre-call Schema Validator: Validate tool calls before executing via ToolEnv.

Validates:
1. Tool name exists in catalog AND is in installed_apps
2. Required params are present
3. Types match schema (string, integer, number, boolean, array)
4. Enum values are valid
5. Format constraints (date, date-time, time)
6. x-one-of groups: at least one complete group must be present
7. Number ranges (minimum, maximum, exclusiveMinimum, exclusiveMaximum)

This does NOT validate:
- x-ref (ID existence) — that requires actual data from ToolEnv reads
- Business logic (enough balance, valid schedule, etc.)
"""
import json
import re
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TRAIN_CATALOG


def load_catalog(catalog_path: Path = TRAIN_CATALOG) -> dict:
    """Load tools catalog."""
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def get_tool_specs(catalog: dict) -> dict[str, dict]:
    """Extract {tool_name: tool_spec} from catalog."""
    specs = {}
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            specs[tool["name"]] = tool
    return specs


def validate_call(
    name: str,
    arguments: dict,
    tool_specs: dict[str, dict],
    installed_tools: set[str] | None = None
) -> dict:
    """Validate a single tool call against the catalog.
    
    Returns: {"ok": True/False, "errors": [...]}
    """
    errors = []
    
    # 1. Tool name exists
    if name not in tool_specs:
        return {"ok": False, "errors": [f"Tool '{name}' not in catalog"]}
    
    # 2. Tool is installed
    if installed_tools is not None and name not in installed_tools and name != "sys.ask_missing_info":
        return {"ok": False, "errors": [f"Tool '{name}' not in installed_apps"]}
    
    spec = tool_specs[name]
    params_spec = spec.get("parameters", {})
    properties = params_spec.get("properties", {})
    required = params_spec.get("required", [])
    x_one_of = params_spec.get("x-one-of", [])
    
    EMPTY = (None, "", [], {})
    
    # 3. Required params present
    for req_key in required:
        if arguments.get(req_key) in EMPTY:
            errors.append(f"Missing required param: {req_key}")
    
    # 4. x-one-of: at least one group fully present
    if x_one_of:
        has_group = any(
            all(arguments.get(k) not in EMPTY for k in group)
            for group in x_one_of
        )
        if not has_group:
            groups_str = " / ".join(" + ".join(g) for g in x_one_of)
            errors.append(f"Must provide one of: {groups_str}")
    
    # 5. Type/value validation for provided params
    for key, value in arguments.items():
        if key not in properties:
            continue  # Extra params — not necessarily an error per schema
        
        if value in EMPTY:
            continue  # Empty optional value
        
        prop_spec = properties[key]
        expected_type = prop_spec.get("type")
        
        # Type check
        type_ok = _check_type(value, expected_type)
        if not type_ok:
            errors.append(f"Param '{key}' must be {expected_type}, got {type(value).__name__}")
            continue
        
        # Enum check
        if "enum" in prop_spec:
            if value not in prop_spec["enum"]:
                errors.append(f"Param '{key}' must be one of {prop_spec['enum']}, got '{value}'")
        
        # Number range checks
        if expected_type in ("integer", "number"):
            if "minimum" in prop_spec and value < prop_spec["minimum"]:
                errors.append(f"Param '{key}' must be >= {prop_spec['minimum']}")
            if "maximum" in prop_spec and value > prop_spec["maximum"]:
                errors.append(f"Param '{key}' must be <= {prop_spec['maximum']}")
            if "exclusiveMinimum" in prop_spec and value <= prop_spec["exclusiveMinimum"]:
                errors.append(f"Param '{key}' must be > {prop_spec['exclusiveMinimum']}")
            if "exclusiveMaximum" in prop_spec and value >= prop_spec["exclusiveMaximum"]:
                errors.append(f"Param '{key}' must be < {prop_spec['exclusiveMaximum']}")
        
        # Format check
        if "format" in prop_spec:
            fmt = prop_spec["format"]
            if not _check_format(value, fmt):
                errors.append(f"Param '{key}' must be format '{fmt}'")
        
        # Array item checks
        if expected_type == "array" and isinstance(value, list) and "items" in prop_spec:
            items_spec = prop_spec["items"]
            for i, item in enumerate(value):
                if not _check_type(item, items_spec.get("type")):
                    errors.append(f"Param '{key}[{i}]' must be {items_spec.get('type')}")
                if "enum" in items_spec and item not in items_spec["enum"]:
                    errors.append(f"Param '{key}[{i}]' must be one of {items_spec['enum']}")
    
    return {"ok": len(errors) == 0, "errors": errors}


def _check_type(value: Any, expected_type: str | None) -> bool:
    """Check if value matches JSON Schema type."""
    if expected_type is None:
        return True
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    return checks.get(expected_type, lambda v: True)(value)


# Format validators matching zaic_tools/validate.py exactly
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATETIME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def _check_format(value: str, fmt: str) -> bool:
    """Check format constraints (date, date-time, time)."""
    if not isinstance(value, str):
        return False
    
    from datetime import date
    
    if fmt == "date":
        m = _DATE_RE.match(value)
        if not m:
            return False
        try:
            date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return True
        except ValueError:
            return False
    
    elif fmt == "date-time":
        m = _DATETIME_RE.match(value)
        if not m:
            return False
        try:
            date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            hh, mm = int(value[11:13]), int(value[14:16])
            return hh < 24 and mm < 60
        except ValueError:
            return False
    
    elif fmt == "time":
        if not _TIME_RE.match(value):
            return False
        parts = value.split(":")
        return int(parts[0]) < 24 and int(parts[1]) < 60
    
    return True


def get_installed_tools(sample: dict) -> set[str]:
    """Get set of tool names from sample's installed_apps."""
    tools = set()
    for app_data in sample.get("context", {}).get("installed_apps", {}).values():
        tools.update(app_data.get("tools", []))
    tools.add("sys.ask_missing_info")  # Always available
    return tools


if __name__ == "__main__":
    print("Loading catalog...")
    catalog = load_catalog()
    tool_specs = get_tool_specs(catalog)
    print(f"Found {len(tool_specs)} tools")
    
    # Test validation
    print("\n--- Validation Tests ---")
    tests = [
        ("smart_home.light_set", {"device_id": "light_1", "power": "on"}, "valid"),
        ("smart_home.light_set", {"power": "on"}, "missing device_id"),
        ("smart_home.light_set", {"device_id": "light_1", "power": "maybe"}, "bad enum"),
        ("nonexistent.tool", {}, "bad tool name"),
        ("alarm.create", {"time": "2026-13-01T10:00:00+07:00"}, "bad date"),
        ("alarm.create", {"time": "2026-11-01T10:00:00+07:00"}, "valid alarm"),
    ]
    
    for name, args, label in tests:
        result = validate_call(name, args, tool_specs)
        status = "✓" if result["ok"] else "✗"
        print(f"  {status} {label}: {name}({args})")
        if result["errors"]:
            for err in result["errors"]:
                print(f"      → {err}")
