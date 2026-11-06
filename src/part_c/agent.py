"""Part C — Agent Module: LLM-based tool-calling agent.

Architecture:
- Sequential tool-calling loop with validator
- Uses filtered tool catalog per sample
- Respects runtime budget (max steps, timeout)
- Tracks observations for grounding

Three output behaviors:
1. Execute: call read → write tools
2. Ask: call sys.ask_missing_info when params are missing
3. OOS: return no calls when no tool can handle the request
"""
import json
import time
import unicodedata
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TRAIN_CATALOG, PUBLIC_TEST_CATALOG, DATA_ROOT

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


class BaseAgent:
    """Base agent class. Subclass and implement `generate()` for specific LLM backends."""
    
    def __init__(
        self,
        catalog_path: Path = DEFAULT_CATALOG,
        max_steps: int = 10,
        max_retries: int = 1,
        verbose: bool = False,
    ):
        """
        Args:
            catalog_path: Path to tools catalog JSON
            max_steps: Maximum tool calls per sample
            max_retries: Max retries on validation error before giving up
            verbose: Print debug info
        """
        self.catalog = load_catalog(catalog_path)
        self.all_tool_specs = get_all_tool_specs(self.catalog)
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.verbose = verbose
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Generate next response from LLM.
        
        Args:
            messages: Chat messages [{"role": ..., "content": ...}]
            tools: Optional tool definitions for function calling
        
        Returns: LLM response text
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement generate()")
    
    def run(self, transcript: str, sample: dict, env) -> None:
        """Run the agent loop on a single sample.
        
        This is the main entry point called by the Runner.
        The agent calls env.call() / env.validate() internally.
        
        Args:
            transcript: ASR transcript (or gold text)
            sample: Full sample dict with context, history, etc.
            env: ToolEnv session for this sample
        """
        # Filter tools for this sample
        sample_tools = filter_tools_for_sample(sample, self.all_tool_specs)
        tools_text = format_tools_for_prompt(sample_tools)
        
        context = sample.get("context", {})
        history = sample.get("history", [])
        
        # Scoped Evidence Ledger per sample (§5.1 of Master Plan)
        ledger = EvidenceLedger(sample)
        observations = []
        
        for step in range(self.max_steps):
            # Build prompt
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
            
            # Generate response
            try:
                response = self.generate(messages)
            except Exception as e:
                if self.verbose:
                    print(f"    LLM error at step {step}: {e}")
                break
            
            # Parse response
            parsed = parse_agent_response(response)
            
            if parsed["action"] == "done":
                if self.verbose:
                    print(f"    Step {step}: DONE")
                break
            
            elif parsed["action"] == "call":
                name = parsed["name"]
                arguments = parsed["arguments"]
                
                # NFC normalize string arguments (but not IDs from tool results)
                arguments = self._normalize_arguments(arguments, observations)
                
                # Validate before calling
                validation = env.validate(name, arguments)
                
                if validation.get("ok"):
                    # Execute the call
                    result = env.call(name, arguments)
                    
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
                    
                    # Terminal tool call: sys.ask_missing_info ends the session
                    if name == "sys.ask_missing_info":
                        if self.verbose:
                            print(f"    Step {step}: sys.ask_missing_info is terminal — ending session.")
                        break
                else:
                    # Validation failed — DO NOT call env.call()!
                    # Give agent a chance to retry
                    error_msg = validation.get("error", "validation failed")
                    observations.append({
                        "call": {"name": name, "arguments": arguments},
                        "result": {"ok": False, "error": f"[VALIDATION] {error_msg}"},
                    })
                    
                    if self.verbose:
                        print(f"    Step {step}: ✗ VALIDATION: {error_msg}")
                
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
                # Give one more chance
                if step == 0:
                    continue
                break
    
    def _normalize_arguments(self, arguments: dict, observations: list[dict]) -> dict:
        """NFC normalize string values, but preserve IDs from tool results."""
        # Collect IDs we've seen from tool results
        known_ids = set()
        for obs in observations:
            result = obs.get("result", {})
            self._collect_ids(result, known_ids)
        
        normalized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                if value in known_ids:
                    # Preserve ID exactly as received from tool
                    normalized[key] = value
                else:
                    # NFC normalize text from ASR/LLM
                    normalized[key] = unicodedata.normalize("NFC", value)
            else:
                normalized[key] = value
        
        return normalized
    
    def _collect_ids(self, data: Any, ids: set):
        """Recursively collect ID-like values from tool results."""
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ("id", "device_id", "contact_id", "event_id", "ride_id",
                          "bill_id", "track_id", "playlist_id", "message_id",
                          "chat_id", "alarm_id", "reminder_id", "note_id"):
                    if isinstance(value, str):
                        ids.add(value)
                self._collect_ids(value, ids)
        elif isinstance(data, list):
            for item in data:
                self._collect_ids(item, ids)


class RuleBasedAgent(BaseAgent):
    """Simple rule-based agent for baseline testing (no LLM needed).
    
    Strategy:
    - If transcript is empty or no tools match → OOS
    - Simple keyword matching to select tools
    - Always reads before writes (to get IDs)
    """
    
    def __init__(self, catalog_path: Path = DEFAULT_CATALOG, verbose: bool = False):
        super().__init__(catalog_path=catalog_path, max_steps=8, verbose=verbose)
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Rule-based 'generation' — returns DONE for baseline."""
        return "DONE"
    
    def run(self, transcript: str, sample: dict, env) -> None:
        """Simple baseline: just return empty (OOS for everything).
        
        This establishes the minimum baseline. Real agent should override this.
        """
        # For baseline, just return nothing (will get OOS correct, everything else wrong)
        pass


class TransformersAgent(BaseAgent):
    """Agent using HuggingFace Transformers for LLM inference.
    
    Suitable for Colab T4 / Kaggle T4x2 with models < 5B params.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        catalog_path: Path = DEFAULT_CATALOG,
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
        """Load model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch required. pip install transformers torch")
        
        print(f"Loading LLM: {self.model_name}...")
        start = time.monotonic()
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        # Use device_map="auto" for proper multi-layer / multi-GPU mapping on CUDA
        device_map = "auto" if "cuda" in self.device else None
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device_map is None:
            self._model = self._model.to(self.device)
        self._model.eval()
        
        elapsed = time.monotonic() - start
        print(f"LLM loaded in {elapsed:.1f}s")
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Generate response using Transformers model."""
        if self._model is None:
            self.load()
        
        import torch
        
        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        device = getattr(self._model, "device", "cuda" if torch.cuda.is_available() else "cpu")
        inputs = self._tokenizer(text, return_tensors="pt").to(device)
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for reproducibility
                temperature=1.0,  # Not used with do_sample=False
                pad_token_id=pad_token_id,
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()


class VLLMAgent(BaseAgent):
    """Agent using vLLM for fast LLM inference.
    
    Recommended for production / evaluation runs.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        catalog_path: Path = DEFAULT_CATALOG,
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
        """Load vLLM model."""
        if self._llm is not None:
            return
        
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm required. pip install vllm")
        
        print(f"Loading vLLM: {self.model_name}...")
        start = time.monotonic()
        
        self._llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=16384,
        )
        self._tokenizer = self._llm.get_tokenizer()
        
        elapsed = time.monotonic() - start
        print(f"vLLM loaded in {elapsed:.1f}s")
    
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Generate response using vLLM."""
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
            stop=["DONE", "\n\nUser:", "\n\n##"],
        )
        
        outputs = self._llm.generate([text], sampling)
        response = outputs[0].outputs[0].text
        
        return response.strip()


def create_agent(
    mode: str = "rule",
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    catalog_path: Path = DEFAULT_CATALOG,
    verbose: bool = False,
) -> BaseAgent:
    """Factory function to create an agent.
    
    Args:
        mode: "rule" (baseline), "transformers", "vllm"
        model_name: HuggingFace model name
        catalog_path: Path to tools catalog
        verbose: Print debug info
    """
    if mode == "rule":
        return RuleBasedAgent(catalog_path=catalog_path, verbose=verbose)
    elif mode == "transformers":
        return TransformersAgent(model_name=model_name, catalog_path=catalog_path, verbose=verbose)
    elif mode == "vllm":
        return VLLMAgent(model_name=model_name, catalog_path=catalog_path, verbose=verbose)
    else:
        raise ValueError(f"Unknown agent mode: {mode}")


if __name__ == "__main__":
    # Test rule-based agent (no GPU needed)
    print("=== Agent Module Test (Rule-Based Baseline) ===\n")
    
    agent = create_agent(mode="rule", verbose=True)
    
    # Simulate a sample
    sample = {
        "id": "test_001",
        "context": {
            "current_time": "2026-11-02T15:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "installed_apps": {
                "smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set"]},
            }
        },
        "history": [],
    }
    
    print("Rule-based agent returns empty (baseline OOS).")
    print("To test with LLM, use: create_agent(mode='transformers', model_name='...')")
