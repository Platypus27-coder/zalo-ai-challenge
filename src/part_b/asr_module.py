"""Part B — ASR Module: Speech-to-text using faster-whisper.

Supports:
- faster-whisper (CTranslate2-based, fast inference)
- Gold transcript fallback for debugging
- Configurable model selection

For Colab/Kaggle with T4 GPU:
- whisper-large-v3-turbo (~800M params) — best quality/speed tradeoff
- whisper-medium (~769M params) — lighter alternative
"""
import time
from pathlib import Path
from typing import Optional

from .normalizer import normalize_transcript


class ASRModule:
    """ASR module using faster-whisper for Vietnamese speech recognition."""
    
    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "vi",
        beam_size: int = 5,
    ):
        """
        Args:
            model_name: Whisper model name (e.g. "large-v3-turbo", "medium")
            device: "cuda" or "cpu"
            compute_type: "float16", "int8_float16", "int8"
            language: Target language code
            beam_size: Beam search size (1=greedy, 5=default)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
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
    
    def transcribe(self, audio_path: str, sample: dict | None = None) -> dict:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to WAV file (16kHz, mono, 16-bit)
            sample: Optional sample dict (for context, not used in basic ASR)
        
        Returns:
            {
                "raw_text": str,           # Raw ASR output
                "normalized_text": str,    # After NFC normalization
                "language": str,           # Detected language
                "duration": float,         # Audio duration in seconds
                "inference_time": float,   # ASR inference time
                "segments": list,          # Per-segment results
            }
        """
        if self.model is None:
            self.load()
        
        start = time.monotonic()
        
        segments_iter, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
            ),
        )
        
        # Collect segments
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
        inference_time = time.monotonic() - start
        
        return {
            # Canonical names
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            # Aliases for 100% backwards and forwards compatibility
            "raw_transcript": raw_text,
            "normalized_transcript": normalized_text,
            "language": info.language if info else self.language,
            "duration": info.duration if info else 0,
            "inference_time": inference_time,
            "segments": segments,
        }
    
    def __call__(self, audio_path: str, sample: dict | None = None, context: dict | None = None) -> str:
        """Shortcut: transcribe and return normalized text."""
        result = self.transcribe(audio_path, sample=sample, context=context)
        return result["normalized_text"]


class GoldTranscriptASR:
    """Fallback ASR that returns the gold transcript from training data.
    
    Used for:
    - Debugging Agent logic without ASR errors
    - Establishing Agent-only baseline scores
    - Comparing gold-transcript vs ASR to isolate error sources
    """
    
    def __init__(self):
        self.model_name = "gold_transcript"
    
    def load(self):
        pass  # No model to load
    
    def transcribe(self, audio_path: str, sample: dict | None = None) -> dict:
        """Return gold transcript from sample."""
        text = ""
        if sample:
            text = sample.get("audio_text", "")
        
        norm_text = normalize_transcript(text)
        return {
            "raw_text": text,
            "normalized_text": norm_text,
            "raw_transcript": text,
            "normalized_transcript": norm_text,
            "language": "vi",
            "duration": 0,
            "inference_time": 0,
            "segments": [{"start": 0, "end": 0, "text": text}],
        }
    
    def __call__(self, audio_path: str, sample: dict | None = None) -> str:
        result = self.transcribe(audio_path, sample)
        return result["normalized_text"]


class HistoryASR:
    """ASR that also transcribes history audio files.
    
    Per official README: history user turns have audio at audio/<id>/<id>_u1.wav, _u2.wav...
    """
    
    def __init__(self, base_asr: ASRModule | GoldTranscriptASR):
        self.base_asr = base_asr
    
    def load(self):
        self.base_asr.load()
    
    def transcribe_with_history(
        self, 
        audio_path: str, 
        sample: dict,
        data_root: str | Path = "."
    ) -> dict:
        """Transcribe current audio + history audio files.
        
        Returns main transcript plus history transcripts.
        """
        # Transcribe main audio
        main_result = self.base_asr.transcribe(audio_path, sample)
        
        # Transcribe history audio if present
        history_transcripts = []
        for turn in sample.get("history", []):
            if turn.get("role") == "user" and turn.get("audio"):
                hist_audio = Path(data_root) / turn["audio"]
                if hist_audio.exists():
                    hist_result = self.base_asr.transcribe(str(hist_audio), sample)
                    history_transcripts.append({
                        "audio": turn["audio"],
                        "text": hist_result["normalized_text"],
                        "inference_time": hist_result["inference_time"],
                    })
                elif sample.get("audio_text"):
                    # Use gold text for history in train
                    if "audio_text" in turn:
                        history_transcripts.append({
                            "audio": turn.get("audio", ""),
                            "text": turn["audio_text"],
                            "inference_time": 0,
                        })
        
        main_result["history_transcripts"] = history_transcripts
        return main_result


def create_asr(
    mode: str = "gold",
    model_name: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
) -> ASRModule | GoldTranscriptASR:
    """Factory function to create ASR module.
    
    Args:
        mode: "gold" (use gold transcripts), "whisper" (use faster-whisper)
        model_name: Whisper model name
        device: "cuda" or "cpu"
        compute_type: Quantization type
    """
    if mode == "gold":
        return GoldTranscriptASR()
    elif mode == "whisper":
        return ASRModule(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )
    else:
        raise ValueError(f"Unknown ASR mode: {mode}")


if __name__ == "__main__":
    # Test with gold transcripts
    print("=== ASR Module Test (Gold Transcript) ===\n")
    
    asr = create_asr(mode="gold")
    
    sample = {
        "audio_text": "Trợ lý ơi, giảm độ sáng đèn phòng ngủ nhỏ xuống còn một nửa nhé.",
        "audio": "audio/train_0004/train_0004.wav",
    }
    
    result = asr.transcribe("dummy_path.wav", sample)
    print(f"Raw text:        {result['raw_text']}")
    print(f"Normalized text: {result['normalized_text']}")
    print(f"Inference time:  {result['inference_time']:.3f}s")
