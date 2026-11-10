from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
"""Part C — Agent Module: Sequential tool-calling agent with Evidence Ledger & Task Tracker.

Features:
- Sequential tool-calling loop with pre-call validation.
- Evidence Ledger for entity grounding and ID provenance.
- Task Tracker for loop prevention and multi-action tracking.
- Monotonic deadline compliance (leaves 1s buffer for finalization).
- Terminal handling for sys.ask_missing_info and DONE.
- Strict Qwen3-4B-Instruct-2507 support (greedy decoding, context limit 16384).
- Colab T4 optimization (SDPA attention, expandable segments, 4-bit / 8-bit support).
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

# Fallback to PUBLIC_TEST_CATALOG when running on test-only environments
DEFAULT_CATALOG = TRAIN_CATALOG if TRAIN_CATALOG.exists() else PUBLIC_TEST_CATALOG

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
        catalog_path: Path = DEFAULT_CATALOG,
        max_steps: int = 5,
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
                    print(f"    LLM error at step {step}: {e}")
                break
            
            # 4. Parse response
            parsed = parse_agent_response(response)
            
            if parsed["action"] == "done":
                if self.verbose:
                    print(f"    Step {step}: DONE")
                break
            
            elif parsed["action"] == "call":
                name = parsed["name"]
                arguments = parsed["arguments"]
                
                # Check tool exists in catalog
                if name not in self.all_tool_specs and not name.startswith("sys."):
                    observations.append({
                        "call": {"name": name, "arguments": arguments},
                        "result": {"ok": False, "error": f"Tool '{name}' không tồn tại trong danh mục hệ thống."}
                    })
                    continue

                # NFC normalize text arguments, but preserve exact IDs
                known_ids = ledger.get_known_ids()
                arguments = self._normalize_arguments(arguments, known_ids)
                
                # Check loop prevention with tool_kind
                tool_kind = self.all_tool_specs.get(name, {}).get("kind", "write")
                if tracker.is_looping(name, arguments, tool_kind=tool_kind):
                    if self.verbose:
                        print(f"    Step {step}: Loop detected on {name}({arguments}) — breaking.")
                    break

                # Intercept hallucinated IDs for write tools before calling env
                ID_PARAM_KEYS = ("device_id", "bill_id", "chat_id", "event_id", "alarm_id", "ride_id", "flight_id", "booking_id", "order_id")
                hallucinated_param = None
                if tool_kind == "write":
                    for pkey in ID_PARAM_KEYS:
                        if pkey in arguments and isinstance(arguments[pkey], str):
                            val = arguments[pkey]
                            if val and val not in known_ids and val not in transcript and val not in str(context):
                                hallucinated_param = (pkey, val)
                                break

                if hallucinated_param:
                    pkey, val = hallucinated_param
                    observations.append({
                        "call": {"name": name, "arguments": arguments},
                        "result": {"ok": False, "error": f"ID '{val}' ({pkey}) chưa từng xuất hiện. Bạn PHẢI gọi tool đọc/tìm kiếm (list/search/get_status) tương ứng để lấy ID thật trước!"}
                    })
                    continue

                # Pre-call validation via environment
                validation = env.validate(name, arguments)
                
                if validation.get("ok"):
                    # Execute tool call
                    result = env.call(name, arguments)
                    tracker.record_call(name, arguments, result, tool_kind=tool_kind)
                    
                    observations.append({
                        "call": {"name": name, "arguments": arguments},
                        "result": result,
                    })

                    # Update Evidence Ledger & Invalidate state if write
                    ledger.update_from_tool_result(name, arguments, result)
                    if not name.startswith(("sys.", "weather.", "gold_price.", "fuel_price.")):
                        ledger.invalidate_on_write(name, arguments)
                    
                    if self.verbose:
                        ok_str = "✓" if result.get("ok") else "✗"
                        print(f"    Step {step}: {ok_str} {name}({json.dumps(arguments, ensure_ascii=False)[:100]})")
                    
                    # Terminal tool: sys.ask_missing_info ends the session immediately
                    if name == "sys.ask_missing_info":
                        if self.verbose:
                            print(f"    Step {step}: sys.ask_missing_info is terminal — ending session.")
                        break
                else:
                    # Validation failed — DO NOT execute invalid call
                    error_msg = validation.get("error", "validation failed")
                    repairs_used += 1
                    
                    observations.append({
                        "call": {"name": name, "arguments": arguments},
                        "result": {"ok": False, "error": f"[VALIDATION] {error_msg}"},
                    })
                    
                    if self.verbose:
                        print(f"    Step {step}: ✗ VALIDATION ({repairs_used}/{self.max_repairs}): {error_msg}")
                    
                    if repairs_used >= self.max_repairs:
                        if self.verbose:
                            print(f"    Step {step}: Max validation repairs exceeded — ending sample.")
                        break
                
                # Loop guard: avoid repeating identical failed call twice
                if len(observations) >= 2:
                    prev_obs = observations[-2]
                    curr_obs = observations[-1]
                    if prev_obs.get("call") == curr_obs.get("call") and not curr_obs.get("result", {}).get("ok"):
                        if self.verbose:
                            print(f"    Step {step}: Repeated identical failed call — stopping loop.")
                        break
            
            elif parsed["action"] == "error":
                if self.verbose:
                    print(f"    Step {step}: PARSE ERROR: {parsed['message'][:100]}")
                if step == 0:
                    continue
                break
    
    def _normalize_arguments(self, arguments: dict, known_ids: set) -> dict:
        """NFC normalize string values, preserving exact IDs from ledger."""
        normalized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                if value in known_ids:
                    normalized[key] = value
                else:
                    normalized[key] = unicodedata.normalize("NFC", value)
            elif isinstance(value, dict):
                normalized[key] = self._normalize_arguments(value, known_ids)
            elif isinstance(value, list):
                normalized[key] = [
                    unicodedata.normalize("NFC", x) if isinstance(x, str) and x not in known_ids else x
                    for x in value
                ]
            else:
                normalized[key] = value
        return normalized


class RuleBasedAgent(BaseAgent):
    """Simple rule-based agent for baseline testing (no LLM needed)."""
    
    def __init__(self, catalog_path: Path = DEFAULT_CATALOG, verbose: bool = False):
        super().__init__(catalog_path=catalog_path, max_steps=8, verbose=verbose)
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        return "DONE"
    
    def run(self, transcript: str, sample: dict, env, deadline: Optional[float] = None) -> None:
        pass


class TransformersAgent(BaseAgent):
    """Agent using HuggingFace Transformers for LLM inference.
    
    Optimized for Colab T4 / Kaggle T4x2 / A30 with SDPA, 4-bit, and 8-bit support.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        adapter_path: Optional[str | Path] = None,
        catalog_path: Path = DEFAULT_CATALOG,
        device: str = "cuda",
        max_new_tokens: int = 256,
        max_steps: int = 5,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        verbose: bool = False,
    ):
        super().__init__(catalog_path=catalog_path, max_steps=max_steps, verbose=verbose)
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self._model = None
        self._tokenizer = None
    
    def load(self):
        """Load model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch required. pip install transformers torch")
        
        load_path = self.model_name
        if LOCAL_LLM_DIR.is_dir():
            load_path = str(LOCAL_LLM_DIR)
        
        print(f"Loading LLM: {load_path}...")
        start = time.monotonic()
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            local_files_only=Path(load_path).is_dir(),
            trust_remote_code=True,
        )
        
        device_map = "auto" if "cuda" in self.device else None
        
        model_kwargs = {
            "device_map": device_map,
            "local_files_only": Path(load_path).is_dir(),
            "trust_remote_code": True,
        }
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:
                print(f"  ⚠️ BitsAndBytesConfig unavailable ({e}), falling back to standard float16...")
                model_kwargs["torch_dtype"] = torch.float16 if "cuda" in self.device else torch.float32
        elif self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                print(f"  ⚠️ BitsAndBytesConfig unavailable ({e}), falling back to standard float16...")
                model_kwargs["torch_dtype"] = torch.float16 if "cuda" in self.device else torch.float32
        else:
            model_kwargs["torch_dtype"] = torch.float16 if "cuda" in self.device else torch.float32
        
        # Use SDPA (Scaled Dot-Product Attention) to avoid quadratic attention memory allocation
        try:
            model_kwargs["attn_implementation"] = "sdpa"
        except Exception:
            pass
        
        self._model = AutoModelForCausalLM.from_pretrained(
            load_path,
            **model_kwargs,
        )
        if device_map is None:
            self._model = self._model.to(self.device)

        # Attach LoRA adapter if provided (supports local directory or Hugging Face repo ID)
        if self.adapter_path:
            adapter_str = str(self.adapter_path)
            print(f"Loading LoRA adapter from '{adapter_str}'...")
            try:
                # Bypass incompatible torchao version check in PEFT
                try:
                    import peft.import_utils
                    peft.import_utils.is_torchao_available = lambda: False
                except Exception:
                    pass

                from peft import PeftModel
                hf_token = os.environ.get("HF_TOKEN")
                self._model = PeftModel.from_pretrained(self._model, adapter_str, token=hf_token)
                print(f"✅ LoRA adapter loaded successfully: {adapter_str}")
            except Exception as e:
                print(f"❌ Failed to load LoRA adapter from '{adapter_str}': {e}")
                raise

        self._model.eval()
        
        elapsed = time.monotonic() - start
        print(f"LLM loaded in {elapsed:.1f}s")
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Generate response using Transformers model."""
        if self._model is None:
            self.load()
        
        import torch
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        device = getattr(self._model, "device", "cuda" if torch.cuda.is_available() else "cpu")
        inputs = self._tokenizer(text, return_tensors="pt").to(device)
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id
        
        # Cap input tokens to prevent context overflow
        if inputs["input_ids"].shape[1] + self.max_new_tokens > MAX_CONTEXT_TOKENS:
            max_input_len = MAX_CONTEXT_TOKENS - self.max_new_tokens
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_len:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_len:]
        
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for reproducibility
                pad_token_id=pad_token_id,
            )
        
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        del inputs, outputs, new_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.strip()


class VLLMAgent(BaseAgent):
    """Agent using vLLM for high-throughput batch=1 inference on A30."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        catalog_path: Path = DEFAULT_CATALOG,
        max_new_tokens: int = 512,
        max_steps: int = 5,
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
            from vllm import LLM
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
            trust_remote_code=True,
            dtype="float16",
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
    adapter_path: Optional[str | Path] = None,
    catalog_path: Path = DEFAULT_CATALOG,
    max_steps: int = 5,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    verbose: bool = False,
) -> BaseAgent:
    """Factory function to instantiate an agent.
    
    Args:
        mode: "rule" (baseline), "transformers", "vllm"
        model_name: HuggingFace model repo or local path
        adapter_path: Path to trained LoRA adapter directory
        catalog_path: Path to catalog JSON
        max_steps: Maximum reasoning / tool execution steps per sample
        load_in_4bit: Quantize LLM to 4-bit (saves VRAM on Colab T4)
        load_in_8bit: Quantize LLM to 8-bit
        verbose: Print debug logs
    """
    if mode == "rule":
        return RuleBasedAgent(catalog_path=catalog_path, verbose=verbose)
    elif mode == "transformers":
        return TransformersAgent(
            model_name=model_name,
            adapter_path=adapter_path,
            catalog_path=catalog_path,
            max_steps=max_steps,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            verbose=verbose,
        )
    elif mode == "vllm":
        return VLLMAgent(model_name=model_name, catalog_path=catalog_path, max_steps=max_steps, verbose=verbose)
    else:
        raise ValueError(f"Unknown agent mode: {mode}")
