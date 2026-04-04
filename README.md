# Cohere Dictation

Local, offline speech-to-text for macOS. Press a hotkey, speak, and the transcription is pasted into whatever app you're using.

Powered by [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) running on Apple Silicon (MPS) — nothing leaves your machine.

## Features

- **Offline** — all processing runs locally on your Mac
- **Auto language detection** — supports 20+ languages out of the box
- **Three recording modes** — auto-stop (silence detection), manual toggle, push-to-talk
- **Auto-paste** — transcription is pasted directly into the active app
- **Vocabulary corrections** — custom word replacements (names, jargon, etc.)
- **Configurable hotkey** — Option+D by default, 10+ alternatives available
- **Floating panel** — shows status, history, and vocabulary
- **macOS native** — Dock icon, Mission Control support, non-intrusive status pill

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4) recommended — works on Intel but much slower
- Python 3.10+
- ~4GB disk space (model weights + dependencies)
- ~2GB RAM during transcription

## Install

```bash
git clone https://github.com/hjpetrovic/transcriber-app.git
cd transcriber-app
./install.sh
```

The installer:
1. Creates a virtual environment with all dependencies
2. Builds a macOS `.app` bundle in `/Applications`
3. Optionally downloads the model (~3GB)

## First launch setup

1. Open **Cohere Dictation** from `/Applications` or Spotlight
2. **Grant Microphone access** when prompted
3. Go to **System Settings > Privacy & Security > Accessibility** and enable the toggle for **Python**
   - This is required for auto-paste (simulating Cmd+V)
   - The Python binary is at: `/opt/homebrew/Cellar/python@3.13/*/Frameworks/Python.framework/Versions/*/Resources/Python.app`
4. Wait for the model to load (first launch downloads ~3GB if not already cached)
5. Press **Option+D** to start dictating!

## Usage

| Action | Default hotkey |
|--------|---------------|
| Start/stop dictation | **Option+D** |
| Cancel (double-tap during recording) | **Option+D** quickly |

### Recording modes

- **Auto-stop** — starts recording, stops automatically when you stop speaking
- **Manual** — press hotkey to start, press again to stop
- **Push-to-talk** — hold the hotkey, release to stop

Change the mode in the Settings panel (gear icon in the app).

### Hotkey alternatives

If Option+D conflicts with your setup, open Settings and pick from:

| Hotkey | Notes |
|--------|-------|
| Option+D/F/J/K | Quick to reach, but produces a character if typed without the app running |
| F5, F6, F7, F8 | No modifier needed, no character produced |
| Ctrl+Shift+D/R | Two modifiers, very unlikely to conflict |

Hotkey changes require an app restart.

## Configuration

All config lives in `~/.config/cohere-dictation/`:

| File | Purpose |
|------|---------|
| `config.yaml` | Settings (hotkey, language, mode, auto-paste, etc.) |
| `vocabulary.txt` | Custom word corrections (`wrong -> correct`, one per line) |
| `app.log` | Debug log for troubleshooting |

### Vocabulary example

```
# ~/.config/cohere-dictation/vocabulary.txt
exponential view -> Exponential View
claude code -> Claude Code
```

## Troubleshooting

### Auto-paste not working
Go to **System Settings > Privacy & Security > Accessibility** and make sure Python is enabled. You may need to remove and re-add it. Restart the app after changing permissions.

### App crashes on launch
Check `~/.config/cohere-dictation/app.log` for errors. Common issues:
- Missing dependencies: re-run `./install.sh`
- Model download interrupted: delete `~/.cache/huggingface/` and relaunch

### Hotkey not responding
- Make sure the app is running (check Dock)
- Check that another app isn't capturing the same hotkey
- Try a different hotkey in Settings

### Slow transcription
- Ensure you're on Apple Silicon with MPS acceleration
- First transcription after launch is slower (model warmup)
- Check `config.yaml` has `float16: true` and `device: auto`

## Uninstall

```bash
# Remove the app
rm -rf "/Applications/Cohere Dictation.app"

# Remove config (optional)
rm -rf ~/.config/cohere-dictation

# Remove cached model (optional, ~3GB)
rm -rf ~/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026
```

## License

MIT
