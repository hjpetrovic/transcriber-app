"""Language detection using Whisper-tiny."""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# Languages supported by both Whisper and Cohere Transcribe
SUPPORTED_LANGUAGES = {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"}

_detector = None


def _load_detector(device: str):
    global _detector
    if _detector is None:
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model = model.to(device)
        _detector = (processor, model)
    return _detector


def detect_language(audio, sample_rate: int, device: str = "cpu") -> str:
    """Detect the language of an audio array.

    Uses Whisper-tiny for fast language identification (~0.2s).
    Falls back to 'en' if the detected language isn't supported by Cohere Transcribe.

    Args:
        audio: 1-D float numpy array at the given sample_rate.
        sample_rate: Sample rate of the audio.
        device: Device for inference.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'fr', 'ja').
    """
    processor, model = _load_detector(device)

    # Use first 30s max
    max_samples = sample_rate * 30
    clip = audio[:max_samples]

    inputs = processor(clip, sampling_rate=sample_rate, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        lang_ids = model.detect_language(inputs.input_features)

    lang_token = processor.tokenizer.decode(lang_ids[0])
    # Token is like "<|en|>" — extract the code
    lang_code = lang_token.strip("<|>")

    if lang_code not in SUPPORTED_LANGUAGES:
        return "en"

    return lang_code
