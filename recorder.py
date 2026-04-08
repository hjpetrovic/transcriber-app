"""Microphone capture using sounddevice."""

import sys
import threading
import time

import numpy as np
import sounddevice as sd
import torch


class Recorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._frames = []
        self._stream = None
        self._start_time = 0.0
        self._lock = threading.Lock()

    def start(self):
        """Start recording from the default input device."""
        with self._lock:
            self._frames = []
            self._recording = True
            self._start_time = time.time()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return the audio as a 1-D float32 numpy array."""
        with self._lock:
            self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0).flatten()

        return audio

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed(self) -> float:
        if not self._recording:
            return 0.0
        return time.time() - self._start_time

    def record_until_silence(self, silence_threshold: float = 1.5, max_duration: float = 60.0) -> np.ndarray:
        """Record until speech stops (silence_threshold seconds of no speech).

        Uses Silero VAD in streaming fashion to detect speech/silence.
        Returns the recorded audio as a 1-D float32 numpy array.
        """
        from vad import _load_vad

        vad_model, _ = _load_vad()
        vad_model.reset_states()

        # VAD processes 512-sample chunks at 16kHz (32ms each)
        chunk_samples = 512
        silence_start = None
        speech_detected = False

        self.start()

        try:
            while self.is_recording:
                elapsed = self.elapsed

                # Safety limit
                if elapsed >= max_duration:
                    sys.stdout.write(f"\rMax duration reached ({max_duration:.0f}s)          \n")
                    sys.stdout.flush()
                    break

                # Need enough audio to analyze
                with self._lock:
                    total_samples = sum(f.shape[0] for f in self._frames)

                if total_samples < chunk_samples:
                    time.sleep(0.01)
                    continue

                # Get the latest chunk for VAD analysis — only grab enough
                # frames from the tail to cover chunk_samples (avoids O(n²) concat)
                with self._lock:
                    needed, count = [], 0
                    for f in reversed(self._frames):
                        needed.insert(0, f)
                        count += f.shape[0]
                        if count >= chunk_samples:
                            break
                    tail = np.concatenate(needed, axis=0).flatten()

                # Analyze the most recent 512 samples
                recent = tail[-chunk_samples:]
                audio_tensor = torch.from_numpy(recent).float()

                confidence = vad_model(audio_tensor, self.sample_rate).item()
                is_speech = confidence > 0.5

                if is_speech:
                    speech_detected = True
                    silence_start = None
                    sys.stdout.write(f"\rRecording... [{elapsed:.1f}s] \033[91m●\033[0m")
                    sys.stdout.flush()
                elif speech_detected:
                    # Speech was detected before, now silence
                    if silence_start is None:
                        silence_start = time.time()
                    silence_duration = time.time() - silence_start
                    sys.stdout.write(f"\rSilence detected... [{silence_duration:.1f}s/{silence_threshold:.1f}s]   ")
                    sys.stdout.flush()
                    if silence_duration >= silence_threshold:
                        sys.stdout.write("\r" + " " * 50 + "\r")
                        sys.stdout.flush()
                        break
                else:
                    # Waiting for speech to start
                    sys.stdout.write(f"\rListening... [{elapsed:.1f}s]   ")
                    sys.stdout.flush()

                time.sleep(0.03)  # ~30ms between VAD checks

        except KeyboardInterrupt:
            pass

        audio = self.stop()
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()
        return audio

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        with self._lock:
            if self._recording:
                self._frames.append(indata.copy())
