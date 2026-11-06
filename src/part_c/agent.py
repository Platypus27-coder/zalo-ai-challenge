"""Part C — Agent Module: Sequential tool-calling agent with Evidence Ledger & Task Tracker.

Features:
- Sequential tool-calling loop with pre-call validation.
- Evidence Ledger for entity grounding and ID provenance.
- Task Tracker for loop prevention and multi-action tracking.
- Monotonic deadline compliance (leaves 1s buffer for finalization).
- Terminal handling for sys.ask_missing_info and DONE.
- Strict Qwen3-4B-Instruct-2507 support (greedy decoding, context limit 16384).
"""
import copy
import json
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    TRAIN_CATALOG, PUBLIC_TEST_CATALOG, DATA_ROOT,
    DEFAULT_LLM_MODEL, LOCAL_LLM_DIR, MAX_CONTEXT_TOKENS,
    MAX_TIME_PER_SAMPLE
)

from .prompt import (
    SYSTEM_PROMPT, build_user_prompt, parse_agent_response,
)
from .tool_filter import (
    load_catalog, get_all_tool_specs, filter_tools_for_sample,
    format_tools_for_prompt, format_tools_as_json_schema,
)
from .evidence_ledger import EvidenceLedger
from .task_tracker import TaskTracker


class BaseAgent:
    """Base agent class defining the sequential tool-execution loop."""
    
    def __init__(
        self,
        catalog_path: Path = TRAIN_CATALOG,
        max_steps: int = 10,
        max_repairs: int = 2,
        verbose: bool = False,
    ):
        self.catalog = load_catalog(catalog_path)
        self.all_tool_specs = get_all_tool_specs(self.catalog)
        self.max_steps = max_steps
        self.max_repairs = max_repairs
        self.verbose = verbose
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Generate response from LLM backend. Subclasses must implement."""
        raise NotImplementedError("Subclass must implement generate()")
    
    def run(
        self,
        transcript: str,
        sample: dict,
        env,
        deadline: Optional[float] = None,
    ) -> None:
        """Run the agent loop on a single sample.
        
        Args:
            transcript: ASR transcript (or gold text for debugging)
            sample: Full sample dict with context, history, etc.
            env: ToolEnv session for this sample (only 1 instance per run)
            deadline: Optional monotonic timestamp deadline (start_time + max_time)
        """
        # Filter allowed tools for this sample (installed_apps + sys.ask_missing_info)
        sample_tools = filter_tools_for_sample(sample, self.all_tool_specs)
        tools_text = format_tools_for_prompt(sample_tools)
        
        context = sample.get("context", {})
        history = sample.get("history", [])
        
        # Initialize Evidence Ledger and Task Tracker
        ledger = EvidenceLedger(sample)
        tracker = TaskTracker(raw_transcript=transcript)
        
        observations = []
        repairs_used = 0
        
        for step in range(self.max_steps):
            # 1. Check monotonic deadline (leave 1.0s buffer for trace collection and hash calculation)
            if deadline is not None and time.monotonic() >= (deadline - 1.0):
                if self.verbose:
                    print(f"    [Step {step}] Approaching deadline, finishing early.")
                break
            
            # 2. Build user prompt with bounded observations
            user_prompt = build_user_prompt(
                transcript=transcript,
                context=context,
                history=history,
                tools_text=tools_text,
                observations=observations,
            )
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            
            # 3. Generate response from LLM
            try:
                response = self.generate(messages)
            except Exception as e:
                if self.verbose:
                    print(f"    [Step {step}] LLM generation error: {e}")
                break
            
            # 4. Parse response
            parsed = parse_agent_response(response)
            action = parsed.get("action")
            
            if action == "done":
                if self.verbose:
                    print(f"    [Step {step}] Agent signaled DONE.")
                break
            
            elif action == "call":
                name = parsed.get("name", "")
                raw_arguments = parsed.get("arguments", {})
                
                # Check tool allowlist
                if name not in sample_tools:
                    if self.verbose:
                        print(f"    [Step {step}] Tool '{name}' not in allowed tools for this sample. Skipping.")
                    if repairs_used < self.max_repairs:
                        repairs_used += 1
                        observations.append({
                            "call": {"name": name, "arguments": raw_arguments},
                            "result": {"ok": False, "error": f"Tool '{name}' không thuộc ứng dụng đã cài đặt."},
                        })
                        continue
                    else:
                        break
                
                # Normalize arguments (preserve exact opaque IDs from ledger/observations)
                arguments = self._normalize_arguments(name, raw_arguments, ledger)
                
                # Loop guard check
                if tracker.is_looping(name, arguments):
                    if self.verbose:
                        print(f"    [Step {step}] Loop detected for {name}({arguments}). Terminating loop.")
                    break
                
                # Pre-call validation via env.validate (do NOT call env.call on schema error)
                validation = env.validate(name, arguments)
                
                if validation.get("ok"):
                    # Execute tool call on environment
                    result = env.call(name, arguments)
                    tool_kind = sample_tools.get(name, {}).get("kind", "write")
                    
                    # Update ledger and tracker
                    ledger.update_from_tool_result(name, arguments, result)
                    if tool_kind == "write":
                        ledger.invalidate_on_write(name, arguments)
                    tracker.record_call(name, arguments, result, tool_kind)
                    
                    observations.append({
                        "call": {"name": name, "arguments": arguments},
                        "result": result,
                    })
                    
                    if self.verbose:
                        ok_symbol = "✓" if result.get("ok") else "✗"
                        print(f"    [Step {step}] {ok_symbol} {name}({json.dumps(arguments, ensure_ascii=False)[:80]})")
                    
                    # Terminal rule: sys.ask_missing_info is terminal
                    if name == "sys.ask_missing_info":
                        if self.verbose:
                            print(f"    [Step {step}] sys.ask_missing_info called. Ending session.")
                        break
                else:
                    # Validation failed — DO NOT call env.call()!
                    error_msg = validation.get("error", "validation failed")
                    if self.verbose:
                        print(f"    [Step {step}] ✗ VALIDATION ERROR: {error_msg}")
                    
                    if repairs_used < self.max_repairs:
                        repairs_used += 1
                        observations.append({
                            "call": {"name": name, "arguments": arguments},
                            "result": {"ok": False, "error": f"[VALIDATION REJECTED] {error_msg}"},
                        })
                    else:
                        break
            
            elif action == "error":
                if self.verbose:
                    print(f"    [Step {step}] PARSE ERROR: {parsed.get('message', '')[:100]}")
                if repairs_used < self.max_repairs:
                    repairs_used += 1
                    observations.append({
                        "call": {"name": "unknown", "arguments": {}},
                        "result": {"ok": False, "error": "Định dạng câu trả lời không đúng. Hãy trả về <tool_call>... JSON ...</tool_call> hoặc DONE."},
                    })
                else:
                    break
    
    def _normalize_arguments(self, tool_name: str, arguments: dict, ledger: EvidenceLedger) -> dict:
        """NFC normalize string values, but preserve IDs from Evidence Ledger."""
        known_ids = ledger.get_known_ids()
        normalized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                # If key looks like an ID or value is an exact known ID, keep unchanged
                if key.endswith("_id") or key == "id" or value in known_ids:
                    normalized[key] = value
                else:
                    normalized[key] = unicodedata.normalize("NFC", value)
            elif isinstance(value, dict):
                normalized[key] = self._normalize_arguments(tool_name, value, ledger)
            elif isinstance(value, list):
                normalized[key] = [
                    unicodedata.normalize("NFC", x) if isinstance(x, str) and x not in known_ids else x
                    for x in value
                ]
            else:
                normalized[key] = value
        return normalized


class RuleBasedAgent(BaseAgent):
    """Simple rule-based baseline agent (returns DONE, suitable for baseline/OOS verification)."""
    
    def __init__(self, catalog_path: Path = TRAIN_CATALOG, verbose: bool = False):
        super().__init__(catalog_path=catalog_path, max_steps=8, verbose=verbose)
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        return "DONE"
    
    def run(self, transcript: str, sample: dict, env, deadline: Optional[float] = None) -> None:
        """Baseline run: performs no calls (pure OOS baseline)."""
        pass


class TransformersAgent(BaseAgent):
    """Agent using HuggingFace Transformers for offline LLM inference.
    
    Default model: Qwen/Qwen3-4B-Instruct-2507 (or local path in models/llm/).
    Greedy decoding (do_sample=False, temperature=0.0) for 100% bitwise determinism.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        catalog_path: Path = TRAIN_CATALOG,
        device: str = "cuda",
        max_new_tokens: int = 512,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        super().__init__(catalog_path=catalog_path, max_steps=max_steps, verbose=verbose)
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
    
    def load(self):
        """Load model and tokenizer from local files or specified path."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch required. pip install transformers torch")
        
        # Check local path first
        load_path = self.model_name
        if LOCAL_LLM_DIR.is_dir():
            load_path = str(LOCAL_LLM_DIR)
        
        if self.verbose:
            print(f"Loading LLM: {load_path}...")
        start = time.monotonic()
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            local_files_only=Path(load_path).is_dir(),
            trust_remote_code=False,
        )
        
        torch_device = self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
        
        self._model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=dtype,
            device_map=torch_device,
            local_files_only=Path(load_path).is_dir(),
            trust_remote_code=False,
        )
        self._model.eval()
        
        elapsed = time.monotonic() - start
        if self.verbose:
            print(f"LLM loaded in {elapsed:.1f}s on {torch_device}")
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Generate response using greedy decoding for deterministic results."""
        if self._model is None:
            self.load()
        
        import torch
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        
        # Cap input tokens to prevent context overflow
        if inputs["input_ids"].shape[1] + self.max_new_tokens > MAX_CONTEXT_TOKENS:
            max_input_len = MAX_CONTEXT_TOKENS - self.max_new_tokens
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_len:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_len:]
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()


class VLLMAgent(BaseAgent):
    """Agent using vLLM for high-throughput batch=1 inference on A30."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        catalog_path: Path = TRAIN_CATALOG,
        max_new_tokens: int = 512,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        super().__init__(catalog_path=catalog_path, max_steps=max_steps, verbose=verbose)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._llm = None
        self._tokenizer = None
    
    def load(self):
        if self._llm is not None:
            return
        
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm required. pip install vllm")
        
        load_path = self.model_name
        if LOCAL_LLM_DIR.is_dir():
            load_path = str(LOCAL_LLM_DIR)
        
        if self.verbose:
            print(f"Loading vLLM: {load_path}...")
        start = time.monotonic()
        
        self._llm = LLM(
            model=load_path,
            trust_remote_code=False,
            dtype="bfloat16",
            max_model_len=MAX_CONTEXT_TOKENS,
        )
        self._tokenizer = self._llm.get_tokenizer()
        
        elapsed = time.monotonic() - start
        if self.verbose:
            print(f"vLLM loaded in {elapsed:.1f}s")
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        if self._llm is None:
            self.load()
        
        from vllm import SamplingParams
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        sampling = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0,  # Greedy
            stop=["DONE", "<done>"],
        )
        
        outputs = self._llm.generate([text], sampling, use_tqdm=False)
        response = outputs[0].outputs[0].text
        return response.strip()


def create_agent(
    mode: str = "rule",
    model_name: str = DEFAULT_LLM_MODEL,
    catalog_path: Path = TRAIN_CATALOG,
    verbose: bool = False,
) -> BaseAgent:
    """Factory function to instantiate an agent.
    
    Args:
        mode: "rule" (baseline), "transformers", "vllm"
        model_name: HuggingFace model repo or local path
        catalog_path: Path to catalog JSON
        verbose: Print debug logs
    """
    if mode == "rule":
        return RuleBasedAgent(catalog_path=catalog_path, verbose=verbose)
    elif mode == "transformers":
        return TransformersAgent(model_name=model_name, catalog_path=catalog_path, verbose=verbose)
    elif mode == "vllm":
        return VLLMAgent(model_name=model_name, catalog_path=catalog_path, verbose=verbose)
    else:
        raise ValueError(f"Unknown agent mode: {mode}")
