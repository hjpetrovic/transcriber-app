#!/bin/bash
set -e

# ─── Cohere Dictation Installer ──────────────────────────────────────────
# Creates a self-contained macOS app in /Applications.
# All source + venv live inside the .app bundle.
# Usage: ./install.sh

APP_NAME="Cohere Dictation"
BUNDLE_ID="com.exponentialview.cohere-dictation"
VERSION="1.0.0"
INSTALL_DIR="/Applications/${APP_NAME}.app"
CONTENTS="${INSTALL_DIR}/Contents"
RESOURCES="${CONTENTS}/Resources"
MACOS="${CONTENTS}/MacOS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════╗"
echo "║   Cohere Dictation — Installer           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ─── Pre-checks ──────────────────────────────────────────────────────────

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This app only runs on macOS."
    exit 1
fi

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This app is optimised for Apple Silicon (M1/M2/M3/M4)."
    echo "It will still work on Intel, but transcription will be slower (CPU only)."
    echo ""
fi

# Check Python 3.10+
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required. Install via: brew install python@3.13"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    echo "Error: Python 3.10+ required (found $PY_VERSION)."
    exit 1
fi
echo "✓ Python $PY_VERSION"

# ─── Remove old installation ─────────────────────────────────────────────

if [[ -d "$INSTALL_DIR" ]]; then
    echo ""
    echo "Existing installation found. Removing..."
    rm -rf "$INSTALL_DIR"
fi

# ─── Create .app bundle structure ─────────────────────────────────────────

echo ""
echo "Creating app bundle..."
mkdir -p "$MACOS" "$RESOURCES"

# ─── Copy source files ────────────────────────────────────────────────────

echo "Copying source files..."
for f in app.py config.py recorder.py transcribe.py vad.py langdetect.py; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        cp "$SCRIPT_DIR/$f" "$RESOURCES/$f"
    else
        echo "Warning: $f not found, skipping"
    fi
done

# ─── Create virtualenv + install dependencies ────────────────────────────

echo ""
echo "Creating virtual environment..."
python3 -m venv "$RESOURCES/.venv"

echo "Installing dependencies (this may take a few minutes)..."
"$RESOURCES/.venv/bin/pip" install --upgrade pip -q
"$RESOURCES/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q

echo "✓ Dependencies installed"

# ─── Create native-named binary + launcher ──────────────────────────────
# Copy the Python binary into the bundle with our app name.
# macOS uses the executable name for the menu bar title.

echo "Setting up launcher..."
PYTHON_FRAMEWORK="$(python3 -c "import sys, os; print(os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'Resources/Python.app/Contents/MacOS/Python'))")"
if [[ ! -f "$PYTHON_FRAMEWORK" ]]; then
    # Fallback: try the Homebrew framework path
    PYTHON_FRAMEWORK="$(python3 -c "import sys; print(sys.executable)")"
fi
cp "$PYTHON_FRAMEWORK" "$MACOS/Cohere Dictation"

cat > "$MACOS/launch" << 'LAUNCHER'
#!/bin/bash
DIR="$(dirname "$(dirname "$0")")/Resources"
MACOS="$(dirname "$0")"
export PATH="$DIR/.venv/bin:$PATH"
export PYTHONPATH="$DIR/.venv/lib/python3.13/site-packages"
cd "$DIR"
exec "$MACOS/Cohere Dictation" "$DIR/app.py"
LAUNCHER
chmod +x "$MACOS/launch"

# ─── Create Info.plist ────────────────────────────────────────────────────

cat > "$CONTENTS/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleExecutable</key>
    <string>launch</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Cohere Dictation needs microphone access to transcribe your speech.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>Cohere Dictation needs automation access to paste transcribed text.</string>
</dict>
</plist>
PLIST

# ─── Generate app icon ────────────────────────────────────────────────────

echo "Generating app icon..."
python3 "$SCRIPT_DIR/generate_icon.py" "$RESOURCES/AppIcon.icns"

# ─── Download model on first install (optional) ──────────────────────────

echo ""
read -p "Download the transcription model now (~3GB)? [Y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Downloading model (this may take a while)..."
    "$RESOURCES/.venv/bin/python" -c "
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
print('Downloading processor...')
AutoProcessor.from_pretrained('CohereLabs/cohere-transcribe-03-2026')
print('Downloading model weights...')
CohereAsrForConditionalGeneration.from_pretrained('CohereLabs/cohere-transcribe-03-2026')
print('Done!')
"
    echo "✓ Model downloaded"
else
    echo "Skipping. The model will download on first launch."
fi

# ─── Done ─────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Installation complete!                  ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "The app is at: $INSTALL_DIR"
echo ""
echo "IMPORTANT — First launch setup:"
echo "  1. Open the app from /Applications (or Spotlight)"
echo "  2. Grant Microphone access when prompted"
echo "  3. Go to System Settings → Privacy & Security → Accessibility"
echo "     and enable the toggle for Python"
echo "     (needed so the app can paste transcriptions)"
echo "  4. Press Option+D to start dictating!"
echo ""
echo "Config files: ~/.config/cohere-dictation/"
echo "  config.yaml   — settings (hotkey, language, mode, etc.)"
echo "  vocabulary.txt — custom word corrections"
echo "  app.log        — debug log"
echo ""
echo "To uninstall: rm -rf \"$INSTALL_DIR\""
