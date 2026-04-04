"""Silero VAD wrapper for filtering speech from audio."""

import torch
import numpy as np


_vad_model = None
_vad_utils = None


def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        _vad_model = model
        _vad_utils = utils
    return _vad_model, _vad_utils


def filter_speech(audio_array: np.ndarray, sample_rate: int, threshold: float = 0.5):
    """Run VAD on audio and return only speech segments.

    Args:
        audio_array: 1-D float numpy array of audio samples.
        sample_rate: Sample rate of the audio.
        threshold: VAD confidence threshold (0-1).

    Returns:
        Tuple of (filtered_audio, timestamps) where:
        - filtered_audio: numpy array of concatenated speech segments, or None if no speech.
        - timestamps: list of (start_sec, end_sec) tuples for each speech segment.
    """
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]

    # Silero VAD expects 16kHz torch tensor
    audio_tensor = torch.from_numpy(audio_array).float()
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0)

    # Get speech timestamps (in samples)
    speech_timestamps = get_speech_timestamps(
        audio_tensor, model,
        sampling_rate=sample_rate,
        threshold=threshold,
        return_seconds=False,
    )

    if not speech_timestamps:
        return None, []

    # Extract speech segments
    segments = []
    timestamps_sec = []
    for ts in speech_timestamps:
        start, end = ts["start"], ts["end"]
        segments.append(audio_array[start:end])
        timestamps_sec.append((start / sample_rate, end / sample_rate))

    filtered_audio = np.concatenate(segments)
    return filtered_audio, timestamps_sec
