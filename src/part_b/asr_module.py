"""Part B — ASR Module: Speech-to-text using faster-whisper with Prompt Biasing.

Supports:
- faster-whisper (CTranslate2-based, fast inference)
- Contextual Prompt Biasing (injecting domain vocabulary & app names into initial_prompt)
- Gold transcript fallback for debugging
- Configurable model selection & offline weights loading
- API contract compliance with Part A (Runner) and Part C (Agent)

Target performance on T4 GPU:
- faster-whisper large-v3-turbo (~800M params, FP16)
- Latency P95 < 2.5s/sample (max budget 5.0s)
"""
import time
from pathlib import Path
from typing import Optional, Any

try:
    from .normalizer import normalize_transcript
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from normalizer import normalize_transcript


# Domain vocabulary for Prompt Biasing (Optimized for Token Budget <= 180 tokens)
# Priority: 1. Negation (Crucial for state_hash) -> 2. Code-switching -> 3. Phone/Money format -> 4. Common apps
DEFAULT_PROMPT_BIAS = (
    "Trợ lý LifeMate. "
    "Tuyệt đối chú ý từ phủ định: không, đừng, hủy, chưa, thôi, ngưng, tắt, khỏi cần, bỏ. "
    "Thuật ngữ: book xe, set alarm, check-out, wifi, AC, OK, cancel, Zalo, VOV, Momo, Grab. "
    "Ứng dụng: smart_home, banking, contacts, bill_payment, ride_hailing, calendar, reminders. "
    "Định dạng: số điện thoại 09, 08, 03, 07, 05, tiền nghìn, triệu, trăm, k."
)


def build_initial_prompt(
    sample: dict | None = None,
    context: dict | None = None,
    extra_terms: str | None = None
) -> str:
    """Build a contextual initial prompt to bias Whisper decoding."""
    prompts = [DEFAULT_PROMPT_BIAS]
    
    ctx = context or (sample.get("context", {}) if sample else {})
    installed_apps = ctx.get("installed_apps", {})
    
    if installed_apps:
        app_names = list(installed_apps.keys())
        prompts.append("Ứng dụng đã cài: " + ", ".join(app_names) + ".")
    
    if extra_terms:
        prompts.append(extra_terms)
        
    return " ".join(prompts)


class ASRModule:
    """ASR module using faster-whisper for Vietnamese speech recognition."""
    
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "vi",
        beam_size: int = 5,
        use_prompt_bias: bool = True,
        vad_parameters: dict | None = None,
    ):
        """
        Args:
            model_name: Whisper model name or local path (default: large-v3)
            device: "cuda" or "cpu"
            compute_type: "float16", "int8_float16", "int8"
            language: Target language code ('vi')
            beam_size: Beam search size (1=greedy, 3-5=recommended)
            use_prompt_bias: Enable domain vocabulary prompt biasing
            vad_parameters: Custom VAD settings (min_silence_duration_ms, speech_pad_ms)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.use_prompt_bias = use_prompt_bias
        self.vad_parameters = vad_parameters or dict(
            min_silence_duration_ms=500,
            speech_pad_ms=300,
        )
        self.model = None
        self._load_time = 0
    
    def load(self):
        """Load the ASR model. Call once before transcribing."""
        if self.model is not None:
            return
        
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with: pip install faster-whisper\n"
                "Or use GoldTranscriptASR for debugging without ASR model."
            )
        
        start = time.monotonic()
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._load_time = time.monotonic() - start
        print(f"ASR model loaded: {self.model_name} ({self._load_time:.1f}s)")
    
    def transcribe(
        self,
        audio_path: str | Path,
        sample: dict | None = None,
        context: dict | None = None,
        initial_prompt: str | None = None,
    ) -> dict:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to WAV file (16kHz, mono, 16-bit)
            sample: Optional sample dict (contains context, installed_apps)
            context: Optional context dict
            initial_prompt: Optional custom initial prompt override
        
        Returns:
            Dict containing normalized and raw transcripts (dual keys for compatibility).
        """
        if self.model is None:
            self.load()
        
        prompt_to_use = None
        if initial_prompt:
            prompt_to_use = initial_prompt
        elif self.use_prompt_bias:
            prompt_to_use = build_initial_prompt(sample=sample, context=context)
        
        start = time.monotonic()
        
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=prompt_to_use,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=self.vad_parameters,
        )
        
        segments = []
        text_parts = []
        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            text_parts.append(seg.text.strip())
        
        raw_text = " ".join(text_parts)
        normalized_text = normalize_transcript(raw_text)

        # Self-healing Fallback: If output is empty (e.g. YouTube hallucination stripped),
        # retry immediately without prompt bias and without VAD to recover raw speech.
        if not normalized_text.strip():
            fallback_iter, fallback_info = self.model.transcribe(
                str(audio_path),
                language=self.language,
                beam_size=3,
                initial_prompt=None,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            fb_parts = [seg.text.strip() for seg in fallback_iter]
            fb_raw = " ".join(fb_parts)
            fb_norm = normalize_transcript(fb_raw)
            
            # If still empty, try greedy decoding with slight temperature
            if not fb_norm.strip():
                fb_iter_t, fb_info_t = self.model.transcribe(
                    str(audio_path),
                    language=self.language,
                    beam_size=1,
                    temperature=0.2,
                    initial_prompt=None,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                fb_parts = [seg.text.strip() for seg in fb_iter_t]
                fb_raw = " ".join(fb_parts)
                fb_norm = normalize_transcript(fb_raw)
                if fb_norm.strip():
                    fallback_info = fb_info_t

            if fb_norm.strip():
                raw_text = fb_raw
                normalized_text = fb_norm
                info = fallback_info

        inference_time = time.monotonic() - start
        
        return {
            "raw_transcript": raw_text,
            "raw_text": raw_text,
            "normalized_transcript": normalized_text,
            "normalized_text": normalized_text,

            "language": info.language if info else self.language,
            "duration": info.duration if info else 0,
            "inference_time": inference_time,
            "segments": segments,
        }
    
    def __call__(self, audio_path: str | Path, sample: dict | None = None, context: dict | None = None) -> str:
        """Shortcut: transcribe and return normalized text."""
        result = self.transcribe(audio_path, sample=sample, context=context)
        return result["normalized_text"]



class GoldTranscriptASR:
    """Fallback ASR that returns the gold transcript from training data.
    
    Used for:
    - Debugging Agent logic without ASR errors
    - Establishing Agent-only baseline scores
    - Isolating ASR error sources
    """
    
    def __init__(self):
        self.model_name = "gold_transcript"
    
    def load(self):
        pass
    
    def transcribe(
        self,
        audio_path: str | Path,
        sample: dict | None = None,
        context: dict | None = None,
        initial_prompt: str | None = None,
    ) -> dict:
        text = ""
        if sample:
            text = sample.get("audio_text", "")
        
        normalized = normalize_transcript(text)

        return {
            "raw_transcript": text,
            "raw_text": text,
            "normalized_transcript": normalized,
            "normalized_text": normalized,

            "language": "vi",
            "duration": 0,
            "inference_time": 0,
            "segments": [{"start": 0, "end": 0, "text": text}],
        }
    
    def __call__(self, audio_path: str | Path, sample: dict | None = None, context: dict | None = None) -> str:
        result = self.transcribe(audio_path, sample=sample, context=context)
        return result["normalized_text"]


# Global default ASR instance for simple function calling
_DEFAULT_ASR_INSTANCE: Optional[ASRModule | GoldTranscriptASR] = None


def transcribe_audio(
    audio_path: str | Path,
    context: dict | None = None,
    initial_prompt: str | None = None,
    asr_instance: Optional[ASRModule | GoldTranscriptASR] = None,
) -> dict[str, str]:
    """Standard API contract function for Part B.
    
    Args:
        audio_path: Path to .wav audio file
        context: Sample context (installed_apps, user_profile, current_time)
        initial_prompt: Optional bias prompt
        asr_instance: Pre-loaded ASR instance
        
    Returns:
        {
            "raw_transcript": str,
            "normalized_transcript": str
        }
    """
    global _DEFAULT_ASR_INSTANCE
    
    instance = asr_instance or _DEFAULT_ASR_INSTANCE
    if instance is None:
        try:
            instance = ASRModule()
            instance.load()
            _DEFAULT_ASR_INSTANCE = instance
        except Exception:
            instance = GoldTranscriptASR()
            _DEFAULT_ASR_INSTANCE = instance
            
    res = instance.transcribe(
        audio_path=audio_path,
        context=context,
        initial_prompt=initial_prompt
    )
    return {
        "raw_transcript": res["raw_transcript"],
        "normalized_transcript": res["normalized_transcript"]
    }


def create_asr(
    mode: str = "gold",
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    use_prompt_bias: bool = True,
    vad_parameters: dict | None = None,
) -> ASRModule | GoldTranscriptASR:
    """Factory function to create ASR module."""
    if mode == "gold":
        return GoldTranscriptASR()
    elif mode == "whisper":
        return ASRModule(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            use_prompt_bias=use_prompt_bias,
            vad_parameters=vad_parameters,
        )
    else:
        raise ValueError(f"Unknown ASR mode: {mode}")


if __name__ == "__main__":
    print("=== ASR Module Test (Gold Transcript & Prompt Biasing) ===\n")
    
    prompt = build_initial_prompt(
        context={"installed_apps": {"smart_home": {}, "banking": {}}}
    )
    print("Generated Prompt Bias preview:")
    print(prompt[:120] + "...\n")
    
    asr = create_asr(mode="gold")
    sample = {
        "audio_text": "Trợ lý ơi, gọi cho số 912345678 và chuyển 500 nghìn cho mẹ.",
        "audio": "audio/train_0004/train_0004.wav",
    }
    
    result = asr.transcribe("dummy.wav", sample)
    print(f"Raw transcript:        {result['raw_transcript']}")
    print(f"Normalized transcript: {result['normalized_transcript']}")
    print(f"Inference time:        {result['inference_time']:.3f}s")
