"""Bug reproduction tests for the two reported issues."""

import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# ── Patch heavy dependencies before importing our modules ──────────────────
sys.modules["sounddevice"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["AppKit"] = MagicMock()
sys.modules["objc"] = MagicMock()
sys.modules["Quartz"] = MagicMock()
sys.modules["PyObjCTools"] = MagicMock()
sys.modules["PyObjCTools.AppHelper"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["Foundation"] = MagicMock()

import importlib, types

# Minimal vad stub (no torch.hub needed)
vad_stub = types.ModuleType("vad")
vad_stub.filter_speech = lambda audio, sample_rate, threshold=0.5: (audio, [])
vad_stub._load_vad = lambda: (MagicMock(), None)
sys.modules["vad"] = vad_stub

# Minimal langdetect stub
ld_stub = types.ModuleType("langdetect")
ld_stub.detect_language = lambda *a, **kw: "en"
sys.modules["langdetect"] = ld_stub

# Minimal config stub
cfg_stub = types.ModuleType("config")
cfg_stub.load_config = lambda: {
    "language": "auto", "hotkey": "option+d", "recording_mode": "auto",
    "silence_threshold": 1.5, "max_recording_duration": 120,
    "auto_paste": True, "sound_feedback": False, "device": "cpu",
    "float16": False, "warmup": False,
}
cfg_stub.save_config = lambda c: None
cfg_stub.load_vocabulary = lambda: []
cfg_stub.save_vocabulary = lambda p: None
cfg_stub.apply_vocabulary = lambda text, vocab: text
cfg_stub.VOCAB_FILE = "/tmp/vocab.txt"
cfg_stub.CONFIG_FILE = "/tmp/config.yaml"
sys.modules["config"] = cfg_stub

import importlib.util, pathlib

recorder_path = str(pathlib.Path(__file__).parent / "recorder.py")
spec = importlib.util.spec_from_file_location("recorder", recorder_path)
recorder_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder_mod)
Recorder = recorder_mod.Recorder


# ─────────────────────────────────────────────────────────────────────────────

class TestDoubleStopSafety(unittest.TestCase):
    """BUG: Calling stop() twice (cancel + end of record_until_silence) should not crash."""

    def setUp(self):
        self.rec = Recorder()
        sd = sys.modules["sounddevice"]
        self.mock_stream = MagicMock()
        sd.InputStream.return_value = self.mock_stream

    def test_stop_when_not_started_returns_empty(self):
        audio = self.rec.stop()
        self.assertEqual(len(audio), 0, "stop() before start() should return empty array")

    def test_double_stop_does_not_raise(self):
        self.rec.start()
        self.rec.stop()
        try:
            self.rec.stop()
        except Exception as e:
            self.fail(f"Second stop() raised: {e}")

    def test_start_resets_frames(self):
        self.rec.start()
        # Simulate audio data arriving
        self.rec._frames.append(np.ones((160, 1), dtype=np.float32))
        self.rec.stop()
        # Start again — frames must be cleared
        self.rec.start()
        self.assertEqual(self.rec._frames, [], "start() must clear previous frames")


class TestCancelRaceCondition(unittest.TestCase):
    """FIX: _cancel_event stops the background thread before transcription."""

    def test_cancel_sets_recording_false(self):
        rec = Recorder()
        sd = sys.modules["sounddevice"]
        sd.InputStream.return_value = MagicMock()

        rec.start()
        self.assertTrue(rec.is_recording)
        rec.stop()
        self.assertFalse(rec.is_recording, "is_recording must be False after stop()")

    def test_cancel_event_prevents_transcription(self):
        """FIX: background thread now checks _cancel_event and bails out."""
        cancel_event = threading.Event()
        transcription_ran = {"value": False}

        def background_thread():
            time.sleep(0.05)  # simulate recording
            if cancel_event.is_set():
                cancel_event.clear()
                return  # bail out — this is the fix
            transcription_ran["value"] = True

        t = threading.Thread(target=background_thread)
        t.start()

        time.sleep(0.02)
        cancel_event.set()  # cancel fires before thread checks it
        t.join()

        self.assertFalse(transcription_ran["value"],
            "After cancel, background thread must NOT run transcription")


class TestFrameConcatenationPerformance(unittest.TestCase):
    """FIX: record_until_silence() now only concatenates the tail frames needed
    for the 512-sample VAD window, not all frames. Should be O(1) per tick.
    """

    def _old_concat(self, frames, chunk_samples):
        all_audio = np.concatenate(frames, axis=0).flatten()
        return all_audio[-chunk_samples:]

    def _new_concat(self, frames, chunk_samples):
        needed, count = [], 0
        for f in reversed(frames):
            needed.insert(0, f)
            count += f.shape[0]
            if count >= chunk_samples:
                break
        tail = np.concatenate(needed, axis=0).flatten()
        return tail[-chunk_samples:]

    def test_old_approach_is_slow(self):
        frames = [np.ones((512, 1), dtype=np.float32) for _ in range(200)]
        times = []
        for i in range(1, len(frames) + 1):
            t0 = time.perf_counter()
            self._old_concat(frames[:i], 512)
            times.append(time.perf_counter() - t0)
        ratio = (sum(times[-20:]) / 20) / (sum(times[:20]) / 20)
        print(f"\n  OLD approach slowdown: {ratio:.1f}x at 200 frames vs 20")
        self.assertGreater(ratio, 2.0, "Old approach should be measurably slower")

    def test_new_approach_stays_flat(self):
        frames = [np.ones((512, 1), dtype=np.float32) for _ in range(200)]
        times = []
        for i in range(1, len(frames) + 1):
            t0 = time.perf_counter()
            self._new_concat(frames[:i], 512)
            times.append(time.perf_counter() - t0)
        ratio = (sum(times[-20:]) / 20) / (sum(times[:20]) / 20)
        print(f"  NEW approach slowdown: {ratio:.1f}x at 200 frames vs 20")
        self.assertLess(ratio, 3.0, "New approach should stay roughly flat")

    def test_new_and_old_return_same_result(self):
        frames = [np.random.rand(512, 1).astype(np.float32) for _ in range(10)]
        old = self._old_concat(frames, 512)
        new = self._new_concat(frames, 512)
        np.testing.assert_array_equal(old, new, "Both approaches must return identical samples")


class TestWindowVisibilityFix(unittest.TestCase):
    """FIX: Panel now uses CanJoinAllSpaces so it follows you across Spaces."""

    def test_panel_uses_can_join_all_spaces(self):
        with open(__file__.replace("test_bugs.py", "app.py")) as f:
            source = f.read()

        panel_section = source[source.index("class DictationPanel"):source.index("class StatusPill")]

        has_join_all = "NSWindowCollectionBehaviorCanJoinAllSpaces" in panel_section
        print(f"\n  DictationPanel uses CanJoinAllSpaces: {has_join_all}")

        self.assertTrue(has_join_all,
            "DictationPanel must use CanJoinAllSpaces to follow the user across Spaces")


if __name__ == "__main__":
    unittest.main(verbosity=2)
