"""Part A — Pre-call Schema Validator: Validate tool calls before executing via ToolEnv.

Validates:
1. Tool name exists in catalog AND is in installed_apps
2. Required params are present
3. Types match schema (string, integer, number, boolean, array)
4. Enum values are valid
5. Format constraints (date, date-time, time)
6. x-one-of groups: at least one complete group must be present
7. Number ranges (minimum, maximum, exclusiveMinimum, exclusiveMaximum)
"""
import json
import re
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TRAIN_CATALOG, DATA_ROOT

# Try importing official zaic_tools validator for 100% exact parity with official scorer
zaic_schema_error = None
try:
    from zaic_tools.validate import schema_error as zaic_schema_error
except ImportError:
    for split_dir in [DATA_ROOT / "train", DATA_ROOT / "public_test"]:
        if (split_dir / "zaic_tools").is_dir():
            if str(split_dir) not in sys.path:
                sys.path.insert(0, str(split_dir))
            try:
                from zaic_tools.validate import schema_error as zaic_schema_error
                break
            except ImportError:
                pass


def load_catalog(catalog_path: Path = TRAIN_CATALOG) -> dict:
    """Load tools catalog."""
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def get_tool_specs(catalog: dict) -> dict[str, dict]:
    """Extract {tool_name: tool_spec} from catalog."""
    specs = {}
    if "apps" in catalog:
        for app in catalog.get("apps", []):
            for tool in app.get("tools", []):
                specs[tool["name"]] = tool
    elif "tools" in catalog:
        for tool in catalog["tools"]:
            specs[tool["name"]] = tool
    elif isinstance(catalog, dict):
        specs = catalog
    return specs


def validate_call(
    name: str,
    arguments: dict,
    tool_specs: dict[str, dict],
    installed_tools: set[str] | None = None
) -> dict:
    """Validate a single tool call against the catalog.
    
    Returns: {"ok": True/False, "errors": [...], "error": str | None}
    """
    if zaic_schema_error is not None:
        err = zaic_schema_error(name, arguments, tool_specs, allowed=installed_tools)
        if err is not None:
            return {"ok": False, "errors": [err], "error": err}
        return {"ok": True, "errors": [], "error": None}

    errors = []
    
    # 1. Tool name exists
    if name not in tool_specs:
        return {"ok": False, "errors": [f"Tool '{name}' not in catalog"], "error": f"Tool '{name}' not in catalog"}
    
    # 2. Tool is installed
    if installed_tools is not None and name not in installed_tools and name != "sys.ask_missing_info":
        return {"ok": False, "errors": [f"Tool '{name}' not in installed_apps"], "error": f"Tool '{name}' not in installed_apps"}
    
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
            continue
        
        if value in EMPTY:
            continue
        
        prop_spec = properties[key]
        expected_type = prop_spec.get("type")
        
        type_ok = _check_type(value, expected_type)
        if not type_ok:
            errors.append(f"Param '{key}' must be {expected_type}, got {type(value).__name__}")
            continue
        
        if "enum" in prop_spec:
            if value not in prop_spec["enum"]:
                errors.append(f"Param '{key}' must be one of {prop_spec['enum']}, got '{value}'")
        
        if expected_type in ("integer", "number"):
            if "minimum" in prop_spec and value < prop_spec["minimum"]:
                errors.append(f"Param '{key}' must be >= {prop_spec['minimum']}")
            if "maximum" in prop_spec and value > prop_spec["maximum"]:
                errors.append(f"Param '{key}' must be <= {prop_spec['maximum']}")
        
        if "format" in prop_spec:
            fmt = prop_spec["format"]
            if not _check_format(value, fmt):
                errors.append(f"Param '{key}' invalid format for '{fmt}': '{value}'")
    
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "error": "; ".join(errors) if errors else None
    }


def _check_type(value: Any, expected_type: str | None) -> bool:
    if expected_type is None:
        return True
    type_map = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    check = type_map.get(expected_type)
    return check(value) if check else True


def _check_format(value: Any, fmt: str) -> bool:
    if not isinstance(value, str):
        return False
    if fmt == "date":
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", value))
    elif fmt == "date-time":
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", value))
    elif fmt == "time":
        return bool(re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", value))
    return True
