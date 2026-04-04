#!/usr/bin/env python3
"""Transcribe an audio file using Cohere Transcribe on Apple Silicon."""

import argparse
import time

import numpy as np
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

from vad import filter_speech


def get_device(requested: str = "auto") -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def load_model(device: str):
    model_id = "CohereLabs/cohere-transcribe-03-2026"

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device,
    )
    load_time = time.time() - t0

    return processor, model, load_time


def transcribe(audio_path: str, language: str, processor, model, use_vad: bool = True):
    # Load audio at 16kHz
    audio = load_audio(audio_path, sampling_rate=16000)
    audio_duration = len(audio) / 16000

    # VAD filtering
    if use_vad:
        t_vad = time.time()
        filtered, timestamps = filter_speech(audio, sample_rate=16000)
        vad_time = time.time() - t_vad
        if filtered is None:
            return None, audio_duration, 0.0, vad_time
        kept = len(filtered) / 16000
        print(f"VAD: kept {kept:.1f}s of {audio_duration:.1f}s ({vad_time*1000:.0f}ms)")
        audio = filtered
    else:
        vad_time = 0.0

    # Process with language
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=language)
    audio_chunk_index = inputs.get("audio_chunk_index")
    inputs.to(model.device, dtype=model.dtype)

    # Generate
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=256)
    transcription_time = time.time() - t0

    # Decode
    text = processor.decode(
        outputs, skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index, language=language,
    )
    if isinstance(text, list):
        text = text[0]

    return text.strip(), audio_duration, transcription_time, vad_time


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Cohere Transcribe")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--language", "-l", default="en", help="Language code (default: en)")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "mps", "cpu"],
                        help="Device to use (default: auto)")
    parser.add_argument("--no-vad", action="store_true", help="Skip VAD pre-filtering")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Loading model...")
    processor, model, load_time = load_model(device)
    print(f"Model loaded in {load_time:.1f}s")

    print("Transcribing...")
    text, audio_duration, transcription_time, vad_time = transcribe(
        args.audio_file, args.language, processor, model, use_vad=not args.no_vad
    )

    if text is None:
        print("\nNo speech detected.")
    else:
        print(f"\n--- Result ---")
        print(text)
        print(f"\n--- Stats ---")
        print(f"Audio duration: {audio_duration:.1f}s")
        print(f"Transcription time: {transcription_time:.1f}s")
        print(f"Real-time factor: {transcription_time / audio_duration:.2f}x")


if __name__ == "__main__":
    main()
