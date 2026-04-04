"""Configuration management for Cohere Dictation."""

import os
from pathlib import Path

import yaml


CONFIG_DIR = Path.home() / ".config" / "cohere-dictation"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
VOCAB_FILE = CONFIG_DIR / "vocabulary.txt"

DEFAULT_CONFIG = {
    "language": "auto",
    "hotkey": "option+d",
    "recording_mode": "auto",       # auto | manual | push
    "silence_threshold": 1.5,
    "max_recording_duration": 120,
    "auto_paste": True,
    "sound_feedback": True,
    "device": "auto",
    "float16": True,
    "warmup": True,
}


def ensure_config():
    """Create default config files if they don't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

    if not VOCAB_FILE.exists():
        with open(VOCAB_FILE, "w") as f:
            f.write("# Vocabulary corrections: wrong spelling -> correct spelling\n")
            f.write("# One entry per line. Applied as post-processing after transcription.\n")
            f.write("#\n")
            f.write("# Examples:\n")
            f.write("# claude code -> Claude Code\n")
            f.write("# exponential view -> Exponential View\n")


def load_config() -> dict:
    """Load config from file, falling back to defaults for missing keys."""
    ensure_config()
    with open(CONFIG_FILE) as f:
        user_config = yaml.safe_load(f) or {}
    config = dict(DEFAULT_CONFIG)
    config.update(user_config)
    return config


def load_vocabulary() -> list[tuple[str, str]]:
    """Load vocabulary corrections as (wrong, correct) pairs."""
    ensure_config()
    pairs = []
    if not VOCAB_FILE.exists():
        return pairs
    with open(VOCAB_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " -> " in line:
                wrong, correct = line.split(" -> ", 1)
                pairs.append((wrong.strip(), correct.strip()))
    return pairs


def apply_vocabulary(text: str, vocab: list[tuple[str, str]]) -> str:
    """Apply vocabulary corrections to transcribed text."""
    for wrong, correct in vocab:
        # Case-insensitive replacement
        lower = text.lower()
        idx = 0
        result = []
        while idx < len(lower):
            pos = lower.find(wrong.lower(), idx)
            if pos == -1:
                result.append(text[idx:])
                break
            result.append(text[idx:pos])
            result.append(correct)
            idx = pos + len(wrong)
        text = "".join(result)
    return text


def save_config(config: dict):
    """Save config dict to YAML file."""
    ensure_config()
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_vocabulary(pairs: list[tuple[str, str]]):
    """Save vocabulary pairs to file."""
    ensure_config()
    with open(VOCAB_FILE, "w") as f:
        f.write("# Vocabulary corrections: wrong spelling -> correct spelling\n")
        f.write("# One entry per line.\n")
        for wrong, correct in pairs:
            f.write(f"{wrong} -> {correct}\n")


def print_config():
    """Print current configuration."""
    config = load_config()
    vocab = load_vocabulary()
    print(f"Config file: {CONFIG_FILE}")
    print(f"Vocabulary file: {VOCAB_FILE}")
    print()
    print("--- Settings ---")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    print(f"--- Vocabulary ({len(vocab)} entries) ---")
    for wrong, correct in vocab:
        print(f"  {wrong} -> {correct}")
    if not vocab:
        print("  (none)")
