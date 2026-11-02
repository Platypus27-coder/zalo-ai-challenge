"""Part A — Exploratory Data Analysis: Statistics on training data.

Produces a comprehensive report covering:
- Total samples, output type distribution (execute/ask/OOS)
- App/domain frequency
- Tool call chain patterns (lengths, common sequences)
- History presence and complexity
- Audio characteristics
- x-ref and x-one-of usage patterns
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TRAIN_JSONL, TRAIN_CATALOG


def load_data(jsonl_path: Path = TRAIN_JSONL) -> list[dict]:
    """Load JSONL data."""
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_catalog(catalog_path: Path = TRAIN_CATALOG) -> dict:
    """Load tools catalog."""
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def analyze_samples(samples: list[dict]) -> dict:
    """Comprehensive analysis of training samples."""
    stats = {}
    
    # === Basic counts ===
    stats["total_samples"] = len(samples)
    
    # === Output type distribution ===
    output_types = Counter()
    for s in samples:
        tc = s.get("label", {}).get("tool_calls", [])
        if len(tc) == 0:
            output_types["oos"] += 1
        elif any(c["name"] == "sys.ask_missing_info" for c in tc):
            output_types["ask_missing_info"] += 1
        else:
            output_types["execute"] += 1
    stats["output_types"] = dict(output_types)
    stats["output_type_pct"] = {k: f"{100*v/len(samples):.1f}%" for k, v in output_types.items()}
    
    # === App frequency ===
    installed_app_counts = Counter()
    used_app_counts = Counter()
    tool_counts = Counter()
    
    for s in samples:
        # Installed apps
        for app in s.get("context", {}).get("installed_apps", {}):
            installed_app_counts[app] += 1
        
        # Used apps (in gold calls)
        for tc in s.get("label", {}).get("tool_calls", []):
            tool_name = tc["name"]
            tool_counts[tool_name] += 1
            if "." in tool_name:
                used_app_counts[tool_name.split(".")[0]] += 1
    
    stats["installed_apps_top20"] = dict(installed_app_counts.most_common(20))
    stats["used_apps_top20"] = dict(used_app_counts.most_common(20))
    stats["tool_calls_top30"] = dict(tool_counts.most_common(30))
    
    # === Trace length distribution ===
    trace_lengths = []
    for s in samples:
        tc = s.get("label", {}).get("tool_calls", [])
        trace_lengths.append(len(tc))
    
    trace_len_dist = Counter(trace_lengths)
    stats["trace_length_distribution"] = dict(sorted(trace_len_dist.items()))
    stats["trace_length_stats"] = {
        "min": min(trace_lengths),
        "max": max(trace_lengths),
        "avg": sum(trace_lengths) / len(trace_lengths),
        "median": sorted(trace_lengths)[len(trace_lengths) // 2]
    }
    
    # === Common tool call chains ===
    chain_counter = Counter()
    for s in samples:
        tc = s.get("label", {}).get("tool_calls", [])
        if tc:
            chain = " → ".join(c["name"] for c in tc)
            chain_counter[chain] += 1
    stats["common_chains_top20"] = dict(chain_counter.most_common(20))
    
    # === History analysis ===
    history_counts = Counter()
    history_lengths = []
    for s in samples:
        history = s.get("history", [])
        history_counts["has_history" if history else "no_history"] += 1
        history_lengths.append(len(history))
    
    stats["history"] = {
        "distribution": dict(history_counts),
        "length_distribution": dict(Counter(history_lengths)),
        "max_length": max(history_lengths),
    }
    
    # === Read/Write pattern analysis ===
    read_write_patterns = Counter()
    for s in samples:
        tc = s.get("label", {}).get("tool_calls", [])
        if tc:
            # Load tool kinds from catalog if available
            pattern = []
            for c in tc:
                name = c["name"]
                if name == "sys.ask_missing_info":
                    pattern.append("meta")
                elif name.endswith(("search", "list", "get", "get_status", "get_balance",
                                     "list_events", "list_tracks", "list_bills", "get_price",
                                     "get_forecast", "get_current", "get_menu", "get_quotes",
                                     "recent_calls", "read_recent", "read_chat", "list_chats",
                                     "guide", "order_history", "list_restaurants",
                                     "list_transactions", "list_beneficiaries")):
                    pattern.append("R")
                else:
                    pattern.append("W")
            read_write_patterns["".join(pattern)] += 1
    
    stats["read_write_patterns_top20"] = dict(read_write_patterns.most_common(20))
    
    # === Multi-app requests ===
    multi_app = Counter()
    for s in samples:
        tc = s.get("label", {}).get("tool_calls", [])
        apps_in_trace = set()
        for c in tc:
            if "." in c["name"]:
                apps_in_trace.add(c["name"].split(".")[0])
        apps_in_trace.discard("sys")
        multi_app[len(apps_in_trace)] += 1
    stats["apps_per_request"] = dict(sorted(multi_app.items()))
    
    # === Audio text length ===
    text_lengths = [len(s.get("audio_text", "")) for s in samples if s.get("audio_text")]
    if text_lengths:
        stats["audio_text_length"] = {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "avg": sum(text_lengths) / len(text_lengths),
        }
    
    # === Code-switching detection ===
    import re
    english_words_count = 0
    english_samples = 0
    for s in samples:
        text = s.get("audio_text", "")
        english_words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        # Filter out common Vietnamese abbreviations
        english_words = [w for w in english_words if w.lower() not in 
                        {"và", "là", "có", "cho", "của", "với", "từ", "này", "đó", "hay"}]
        if english_words:
            english_samples += 1
            english_words_count += len(english_words)
    
    stats["code_switching"] = {
        "samples_with_english": english_samples,
        "pct": f"{100*english_samples/len(samples):.1f}%",
    }
    
    return stats


def analyze_catalog(catalog: dict) -> dict:
    """Analyze the tools catalog."""
    stats = {}
    
    apps = catalog.get("apps", [])
    stats["total_apps"] = len(apps)
    
    # App groups
    groups = Counter(app.get("group", "unknown") for app in apps)
    stats["app_groups"] = dict(groups)
    
    # Tool counts per app
    tools_per_app = {}
    total_tools = 0
    xref_tools = []
    fuzzy_tools = []
    xoneof_tools = []
    
    for app in apps:
        app_id = app["id"]
        tools = app.get("tools", [])
        tools_per_app[app_id] = len(tools)
        total_tools += len(tools)
        
        for tool in tools:
            params = tool.get("parameters", {})
            properties = params.get("properties", {})
            
            # Check x-ref
            for pname, pspec in properties.items():
                if "x-ref" in pspec:
                    xref_tools.append({"tool": tool["name"], "param": pname, "ref": pspec["x-ref"]})
                if pspec.get("x-match") == "fuzzy":
                    fuzzy_tools.append({"tool": tool["name"], "param": pname})
            
            # Check x-one-of
            if params.get("x-one-of"):
                xoneof_tools.append({"tool": tool["name"], "groups": params["x-one-of"]})
    
    stats["total_tools"] = total_tools
    stats["tools_per_app"] = tools_per_app
    
    # Kind distribution
    kind_counts = Counter()
    for app in apps:
        for tool in app.get("tools", []):
            kind_counts[tool.get("kind", "unknown")] += 1
    stats["tool_kinds"] = dict(kind_counts)
    
    stats["xref_params"] = xref_tools
    stats["fuzzy_params"] = fuzzy_tools
    stats["xoneof_params"] = xoneof_tools
    
    return stats


def print_report(sample_stats: dict, catalog_stats: dict):
    """Print a formatted analysis report."""
    print("=" * 80)
    print("LIFEMATE TRAINING DATA ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\n📊 Total Samples: {sample_stats['total_samples']}")
    print(f"\n📋 Output Types:")
    for k, v in sample_stats["output_types"].items():
        pct = sample_stats["output_type_pct"][k]
        print(f"   {k:20s}: {v:5d} ({pct})")
    
    print(f"\n🔧 Trace Length:")
    ts = sample_stats["trace_length_stats"]
    print(f"   Min: {ts['min']}, Max: {ts['max']}, Avg: {ts['avg']:.1f}, Median: {ts['median']}")
    print(f"   Distribution: {sample_stats['trace_length_distribution']}")
    
    print(f"\n📱 Apps per Request:")
    for k, v in sample_stats["apps_per_request"].items():
        print(f"   {k} app(s): {v}")
    
    print(f"\n🏠 Top Installed Apps:")
    for app, count in list(sample_stats["installed_apps_top20"].items())[:15]:
        print(f"   {app:25s}: {count:5d}")
    
    print(f"\n🎯 Top Used Apps (in gold calls):")
    for app, count in list(sample_stats["used_apps_top20"].items())[:15]:
        print(f"   {app:25s}: {count:5d}")
    
    print(f"\n🔗 Top Tool Calls:")
    for tool, count in list(sample_stats["tool_calls_top30"].items())[:20]:
        print(f"   {tool:40s}: {count:5d}")
    
    print(f"\n📜 History: {sample_stats['history']['distribution']}")
    
    print(f"\n🔀 R/W Patterns (top 15):")
    for pattern, count in list(sample_stats["read_write_patterns_top20"].items())[:15]:
        print(f"   {pattern:15s}: {count:5d}")
    
    print(f"\n🔤 Code-Switching: {sample_stats['code_switching']}")
    
    print(f"\n📝 Audio Text Length: {sample_stats.get('audio_text_length', 'N/A')}")
    
    # Catalog stats
    print(f"\n{'=' * 80}")
    print("TOOLS CATALOG ANALYSIS")
    print(f"{'=' * 80}")
    print(f"\n📦 Total Apps: {catalog_stats['total_apps']}")
    print(f"🔧 Total Tools: {catalog_stats['total_tools']}")
    print(f"   Kinds: {catalog_stats['tool_kinds']}")
    print(f"   Groups: {catalog_stats['app_groups']}")
    
    print(f"\n🔗 x-ref Params ({len(catalog_stats['xref_params'])}):")
    for xr in catalog_stats["xref_params"][:15]:
        print(f"   {xr['tool']}.{xr['param']} → {xr['ref']}")
    
    print(f"\n🔤 x-match: fuzzy Params ({len(catalog_stats['fuzzy_params'])}):")
    for fp in catalog_stats["fuzzy_params"][:15]:
        print(f"   {fp['tool']}.{fp['param']}")
    
    print(f"\n📋 x-one-of Groups ({len(catalog_stats['xoneof_params'])}):")
    for xo in catalog_stats["xoneof_params"][:10]:
        groups_str = " / ".join(" + ".join(g) for g in xo["groups"])
        print(f"   {xo['tool']}: {groups_str}")
    
    # Top chains
    print(f"\n🔗 Top Tool Call Chains:")
    for chain, count in list(sample_stats["common_chains_top20"].items())[:15]:
        print(f"   [{count:3d}] {chain}")
    
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    print("Loading data...")
    samples = load_data()
    catalog = load_catalog()
    
    print("Analyzing samples...")
    sample_stats = analyze_samples(samples)
    
    print("Analyzing catalog...")
    catalog_stats = analyze_catalog(catalog)
    
    print_report(sample_stats, catalog_stats)
    
    # Save report
    from config import OUTPUTS_DIR
    report_path = OUTPUTS_DIR / "eda_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"samples": sample_stats, "catalog": catalog_stats}, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {report_path}")
