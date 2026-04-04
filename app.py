#!/usr/bin/env python3
"""Cohere Dictation — macOS floating panel app."""

import logging
import subprocess
import threading
import time
from pathlib import Path

import AppKit
import numpy as np
import objc
import torch
import Quartz
from PyObjCTools import AppHelper
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

# ─── File logging (visible via: tail -f ~/.config/cohere-dictation/app.log) ──
LOG_FILE = Path.home() / ".config" / "cohere-dictation" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("dictation")
log.setLevel(logging.DEBUG)
_fh = logging.FileHandler(str(LOG_FILE), mode="w")  # fresh each launch
_fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_fh)

from config import (
    load_config, save_config, load_vocabulary, save_vocabulary,
    apply_vocabulary, VOCAB_FILE, CONFIG_FILE,
)
from langdetect import detect_language
from recorder import Recorder
from vad import filter_speech


SAMPLE_RATE = 16000

# Recording modes
MODE_AUTO = "auto"       # Auto-stop on silence
MODE_MANUAL = "manual"   # Press hotkey to start, again to stop
MODE_PUSH = "push"       # Hold hotkey, release to stop

# ─── Hotkey definitions ───────────────────────────────────────────────────
# Maps config string → (keycode, modifier_mask, display_label)
# macOS keycodes: D=2, F=3, J=38, K=40, F5=96, F6=97, F7=98, F8=100
HOTKEY_OPTIONS = {
    "option+d":     (2,   Quartz.kCGEventFlagMaskAlternate, "⌥D"),
    "option+f":     (3,   Quartz.kCGEventFlagMaskAlternate, "⌥F"),
    "option+j":     (38,  Quartz.kCGEventFlagMaskAlternate, "⌥J"),
    "option+k":     (40,  Quartz.kCGEventFlagMaskAlternate, "⌥K"),
    "f5":           (96,  0, "F5"),
    "f6":           (97,  0, "F6"),
    "f7":           (98,  0, "F7"),
    "f8":           (100, 0, "F8"),
    "ctrl+shift+d": (2,   Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskShift, "⌃⇧D"),
    "ctrl+shift+r": (15,  Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskShift, "⌃⇧R"),
}
DEFAULT_HOTKEY = "option+d"


# ─── Colors ────────────────────────────────────────────────────────────────

BG_COLOR = (0.12, 0.12, 0.14, 0.95)
ACCENT_GREEN = (0.3, 0.85, 0.4)
ACCENT_RED = (1.0, 0.25, 0.25)
ACCENT_BLUE = (0.4, 0.55, 1.0)
ACCENT_GRAY = (0.5, 0.5, 0.5)
ACCENT_ORANGE = (1.0, 0.6, 0.2)
TEXT_PRIMARY = AppKit.NSColor.whiteColor()
TEXT_SECONDARY = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.65, 0.65, 0.7, 1.0)
TEXT_DIM = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.45, 0.45, 0.5, 1.0)


def ns_color(r, g, b, a=1.0):
    return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a)


OUR_BUNDLE_ID = "com.exponentialview.cohere-dictation"


def _get_frontmost_bundle_id():
    """Get the bundle ID of the currently focused app.
    Filters out our own app so we always get the *user's* target app."""
    try:
        # Fast path: use NSWorkspace (no subprocess)
        ws = AppKit.NSWorkspace.sharedWorkspace()
        front = ws.frontmostApplication()
        if front:
            bid = front.bundleIdentifier()
            if bid and bid != OUR_BUNDLE_ID:
                log.info(f"Frontmost app: {bid}")
                return bid

        # If we are frontmost, find the most recently active other app
        running = ws.runningApplications()
        for app in running:
            bid = app.bundleIdentifier()
            if bid and bid != OUR_BUNDLE_ID and not app.isHidden() and app.activationPolicy() == AppKit.NSApplicationActivationPolicyRegular:
                log.info(f"Fallback app: {bid}")
                return bid
    except Exception as e:
        log.error(f"_get_frontmost_bundle_id error: {e}")
    log.warning("Could not determine frontmost app")
    return None


def _check_accessibility():
    """Check if we have accessibility permissions. Prompt if not."""
    trusted = Quartz.CGPreflightListenEventAccess()
    post_ok = Quartz.CGPreflightPostEventAccess()
    log.info(f"Accessibility: listen={trusted}, post={post_ok}")
    if not post_ok:
        log.warning("No post-event accessibility — requesting...")
        Quartz.CGRequestPostEventAccess()
        # Also show a dialog so the user knows
        def _alert():
            alert = AppKit.NSAlert.alloc().init()
            alert.setMessageText_("Accessibility Permission Required")
            alert.setInformativeText_(
                "Cohere Dictation needs Accessibility access to paste transcriptions.\n\n"
                "Go to: System Settings → Privacy & Security → Accessibility\n"
                "and enable the toggle for Python or Cohere Dictation.\n\n"
                "You may need to remove and re-add it if it was previously granted."
            )
            alert.addButtonWithTitle_("Open System Settings")
            alert.addButtonWithTitle_("OK")
            response = alert.runModal()
            if response == AppKit.NSAlertFirstButtonReturn:
                subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"])
        AppHelper.callAfter(_alert)


def _activate_and_paste(bundle_id):
    """Re-activate the user's app and paste clipboard contents.

    Uses CGEvent at HID level (same approach as Raycast/Alfred).
    Falls back to AppleScript if CGEvent fails.
    Both require Accessibility permission in System Settings.
    """
    if not bundle_id:
        log.warning("No bundle_id — skipping paste")
        return False

    # Activate the target app
    apps = AppKit.NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id)
    if apps:
        apps[0].activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
        log.info(f"Activated {bundle_id}")
    else:
        subprocess.Popen(["osascript", "-e", f'tell application id "{bundle_id}" to activate'])
        log.info(f"Activated {bundle_id} via osascript")

    time.sleep(0.3)  # let app fully come to front

    # Simulate Cmd+V via CGEvent at HID level
    try:
        src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        # keycode 9 = 'V'
        key_down = Quartz.CGEventCreateKeyboardEvent(src, 9, True)
        Quartz.CGEventSetFlags(key_down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_down)
        time.sleep(0.05)
        key_up = Quartz.CGEventCreateKeyboardEvent(src, 9, False)
        Quartz.CGEventSetFlags(key_up, 0)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_up)
        log.info(f"Sent Cmd+V via CGEvent into {bundle_id}")
        return True
    except Exception as e:
        log.error(f"CGEvent paste failed: {e}")

    # Fallback: AppleScript
    try:
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to keystroke "v" using command down'],
            capture_output=True, timeout=3,
        )
        log.info("Fallback: sent Cmd+V via AppleScript")
        return True
    except Exception as e:
        log.error(f"AppleScript fallback also failed: {e}")
        return False


# ─── ObjC bridge for button actions ───────────────────────────────────────

class ActionTarget(AppKit.NSObject):
    """Bridge to let NSButtons call Python callbacks."""
    _callback = None

    def initWithCallback_(self, cb):
        self = objc.super(ActionTarget, self).init()
        if self is not None:
            self._callback = cb
        return self

    def doAction_(self, sender):
        if self._callback:
            self._callback()


# ─── Custom Views ──────────────────────────────────────────────────────────

class RoundedView(AppKit.NSView):
    _r, _g, _b, _a = BG_COLOR

    def updateColor_g_b_a_(self, r, g, b, a):
        self._r, self._g, self._b, self._a = r, g, b, a
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            self.bounds(), 14, 14
        )
        ns_color(self._r, self._g, self._b, self._a).setFill()
        path.fill()


class DotView(AppKit.NSView):
    _r, _g, _b = ACCENT_GRAY

    def updateColor_g_b_(self, r, g, b):
        self._r, self._g, self._b = r, g, b
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(self.bounds())
        ns_color(self._r, self._g, self._b).setFill()
        path.fill()


class SeparatorView(AppKit.NSView):
    def drawRect_(self, rect):
        path = AppKit.NSBezierPath.bezierPathWithRect_(
            AppKit.NSMakeRect(0, 0, self.bounds().size.width, 1)
        )
        ns_color(0.3, 0.3, 0.35, 0.5).setFill()
        path.fill()


class ClickableLabel(AppKit.NSTextField):
    """A label that calls a callback when clicked."""
    _click_callback = None

    def setClickCallback_(self, cb):
        self._click_callback = cb

    def mouseDown_(self, event):
        if self._click_callback:
            self._click_callback()

    def resetCursorRects(self):
        self.addCursorRect_cursor_(self.bounds(), AppKit.NSCursor.pointingHandCursor())


# ─── Vocabulary Editor Window ──────────────────────────────────────────────

class VocabEditor:
    """A window for editing vocabulary corrections."""

    WIDTH = 360
    HEIGHT = 400

    def __init__(self, on_save=None):
        self.on_save = on_save
        self.entries = []
        self.vocab = load_vocabulary()
        self._action_targets = []  # prevent GC

        screen = AppKit.NSScreen.mainScreen().frame()
        x = (screen.size.width - self.WIDTH) / 2
        y = (screen.size.height - self.HEIGHT) / 2

        self.window = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(x, y, self.WIDTH, self.HEIGHT),
            AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("Edit Vocabulary")
        self.window.setBackgroundColor_(ns_color(0.15, 0.15, 0.17))
        self.window.setLevel_(AppKit.NSFloatingWindowLevel)

        content = self.window.contentView()

        # Scroll view for entries
        self.scroll = AppKit.NSScrollView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 50, self.WIDTH, self.HEIGHT - 80)
        )
        self.scroll.setHasVerticalScroller_(True)
        self.scroll.setDrawsBackground_(False)

        self.container = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, self.WIDTH - 20, 400)
        )
        self.scroll.setDocumentView_(self.container)
        content.addSubview_(self.scroll)

        # Header labels
        h1 = self._make_header("wrong spelling", 16)
        self.container.addSubview_(h1)
        h2 = self._make_header("→", 152)
        self.container.addSubview_(h2)
        h3 = self._make_header("correct spelling", 175)
        self.container.addSubview_(h3)

        # Populate existing + one blank
        for wrong, correct in self.vocab:
            self._add_entry_row(wrong, correct)
        self._add_entry_row("", "")
        self._relayout()

        # Bottom: + Add and Save buttons (using ClickableLabel)
        add_lbl = ClickableLabel.alloc().initWithFrame_(
            AppKit.NSMakeRect(20, 14, 60, 22)
        )
        add_lbl.setStringValue_("+ Add")
        add_lbl.setBezeled_(False)
        add_lbl.setDrawsBackground_(False)
        add_lbl.setEditable_(False)
        add_lbl.setSelectable_(False)
        add_lbl.setTextColor_(ns_color(*ACCENT_BLUE))
        add_lbl.setFont_(AppKit.NSFont.systemFontOfSize_weight_(13, AppKit.NSFontWeightMedium))
        add_lbl.setClickCallback_(self._add_blank_row)
        content.addSubview_(add_lbl)

        save_lbl = ClickableLabel.alloc().initWithFrame_(
            AppKit.NSMakeRect(self.WIDTH - 80, 14, 60, 22)
        )
        save_lbl.setStringValue_("Save")
        save_lbl.setBezeled_(False)
        save_lbl.setDrawsBackground_(False)
        save_lbl.setEditable_(False)
        save_lbl.setSelectable_(False)
        save_lbl.setTextColor_(ns_color(*ACCENT_GREEN))
        save_lbl.setFont_(AppKit.NSFont.systemFontOfSize_weight_(13, AppKit.NSFontWeightSemibold))
        save_lbl.setAlignment_(AppKit.NSTextAlignmentRight)
        save_lbl.setClickCallback_(self._save)
        content.addSubview_(save_lbl)

    def _make_header(self, text, x):
        lbl = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(x, 14, 140, 16)
        )
        lbl.setStringValue_(text)
        lbl.setBezeled_(False)
        lbl.setDrawsBackground_(False)
        lbl.setEditable_(False)
        lbl.setSelectable_(False)
        lbl.setTextColor_(TEXT_DIM)
        lbl.setFont_(AppKit.NSFont.systemFontOfSize_weight_(10, AppKit.NSFontWeightMedium))
        return lbl

    def _add_entry_row(self, wrong, correct):
        idx = len(self.entries)

        wrong_field = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(16, 0, 130, 24))
        wrong_field.setStringValue_(wrong)
        wrong_field.setPlaceholderString_("wrong")
        wrong_field.setFont_(AppKit.NSFont.systemFontOfSize_(12))
        wrong_field.setBezeled_(True)
        wrong_field.setBezelStyle_(AppKit.NSTextFieldSquareBezel)
        wrong_field.setDrawsBackground_(True)
        wrong_field.setBackgroundColor_(ns_color(0.2, 0.2, 0.22))
        wrong_field.setTextColor_(TEXT_PRIMARY)
        self.container.addSubview_(wrong_field)

        arrow = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(152, 2, 20, 20))
        arrow.setStringValue_("→")
        arrow.setBezeled_(False)
        arrow.setDrawsBackground_(False)
        arrow.setEditable_(False)
        arrow.setSelectable_(False)
        arrow.setTextColor_(TEXT_DIM)
        arrow.setFont_(AppKit.NSFont.systemFontOfSize_(13))
        self.container.addSubview_(arrow)

        correct_field = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(175, 0, 130, 24))
        correct_field.setStringValue_(correct)
        correct_field.setPlaceholderString_("correct")
        correct_field.setFont_(AppKit.NSFont.systemFontOfSize_(12))
        correct_field.setBezeled_(True)
        correct_field.setBezelStyle_(AppKit.NSTextFieldSquareBezel)
        correct_field.setDrawsBackground_(True)
        correct_field.setBackgroundColor_(ns_color(0.2, 0.2, 0.22))
        correct_field.setTextColor_(TEXT_PRIMARY)
        self.container.addSubview_(correct_field)

        del_btn = ClickableLabel.alloc().initWithFrame_(AppKit.NSMakeRect(312, 2, 20, 20))
        del_btn.setStringValue_("✕")
        del_btn.setBezeled_(False)
        del_btn.setDrawsBackground_(False)
        del_btn.setEditable_(False)
        del_btn.setSelectable_(False)
        del_btn.setTextColor_(ns_color(0.8, 0.3, 0.3))
        del_btn.setFont_(AppKit.NSFont.systemFontOfSize_(12))
        del_btn.setClickCallback_(lambda idx=idx: self._delete_row(idx))
        self.container.addSubview_(del_btn)

        self.entries.append((wrong_field, correct_field, arrow, del_btn))

    def _relayout(self):
        n = len(self.entries)
        total_height = max(40 + n * 32, self.HEIGHT - 80)
        self.container.setFrameSize_(AppKit.NSMakeSize(self.WIDTH - 20, total_height))
        for i, (wf, cf, arrow, db) in enumerate(self.entries):
            y = total_height - 40 - i * 32
            wf.setFrameOrigin_(AppKit.NSMakePoint(16, y))
            arrow.setFrameOrigin_(AppKit.NSMakePoint(152, y + 2))
            cf.setFrameOrigin_(AppKit.NSMakePoint(175, y))
            db.setFrameOrigin_(AppKit.NSMakePoint(312, y + 2))

    def _delete_row(self, idx):
        if idx < len(self.entries):
            for view in self.entries[idx]:
                view.removeFromSuperview()
            self.entries.pop(idx)
            for i, (_, _, _, db) in enumerate(self.entries):
                db.setClickCallback_(lambda idx=i: self._delete_row(idx))
            self._relayout()

    def _add_blank_row(self):
        self._add_entry_row("", "")
        self._relayout()

    def _save(self):
        pairs = []
        for wf, cf, _, _ in self.entries:
            wrong = wf.stringValue().strip()
            correct = cf.stringValue().strip()
            if wrong and correct:
                pairs.append((wrong, correct))
        save_vocabulary(pairs)
        self.window.close()
        if self.on_save:
            self.on_save()

    def show(self):
        self.window.makeKeyAndOrderFront_(None)


# ─── Settings Editor Window ───────────────────────────────────────────────

class SettingsEditor:
    """A window for editing app settings."""

    WIDTH = 340
    HEIGHT = 600

    def __init__(self, config, on_save=None):
        self.config = dict(config)
        self.on_save = on_save
        self._selected_mode = config.get("recording_mode", MODE_AUTO)
        self._selected_hotkey = config.get("hotkey", DEFAULT_HOTKEY)

        screen = AppKit.NSScreen.mainScreen().frame()
        x = (screen.size.width - self.WIDTH) / 2
        y = (screen.size.height - self.HEIGHT) / 2

        self.window = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(x, y, self.WIDTH, self.HEIGHT),
            AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("Settings")
        self.window.setBackgroundColor_(ns_color(0.15, 0.15, 0.17))
        self.window.setLevel_(AppKit.NSFloatingWindowLevel)

        content = self.window.contentView()
        y_pos = self.HEIGHT - 50

        # ── Recording mode ──
        y_pos = self._section_title(content, "RECORDING MODE", y_pos)

        mode_info = [
            (MODE_AUTO,   "Auto-stop (stops when you stop talking)"),
            (MODE_MANUAL, "Manual (⌥D to start & stop)"),
            (MODE_PUSH,   "Push-to-talk (hold ⌥D, release to stop)"),
        ]
        self._mode_labels = {}
        for mode, label in mode_info:
            y_pos -= 26
            lbl = ClickableLabel.alloc().initWithFrame_(
                AppKit.NSMakeRect(20, y_pos, self.WIDTH - 40, 20)
            )
            lbl.setBezeled_(False)
            lbl.setDrawsBackground_(False)
            lbl.setEditable_(False)
            lbl.setSelectable_(False)
            lbl.setFont_(AppKit.NSFont.systemFontOfSize_(12))
            lbl.setClickCallback_(lambda m=mode: self._select_mode(m))
            content.addSubview_(lbl)
            self._mode_labels[mode] = (lbl, label)

        self._refresh_mode_labels()

        # ── Hotkey ──
        y_pos -= 28
        y_pos = self._section_title(content, "HOTKEY (restart app to apply)", y_pos)

        # Show hotkey options in a scrollable list of clickable labels
        hotkey_display_order = [
            "option+d", "option+f", "option+j", "option+k",
            "f5", "f6", "f7", "f8",
            "ctrl+shift+d", "ctrl+shift+r",
        ]
        self._hotkey_labels = {}
        for hk_name in hotkey_display_order:
            if hk_name not in HOTKEY_OPTIONS:
                continue
            _, _, display = HOTKEY_OPTIONS[hk_name]
            y_pos -= 22
            lbl = ClickableLabel.alloc().initWithFrame_(
                AppKit.NSMakeRect(20, y_pos, self.WIDTH - 40, 18)
            )
            lbl.setBezeled_(False)
            lbl.setDrawsBackground_(False)
            lbl.setEditable_(False)
            lbl.setSelectable_(False)
            lbl.setFont_(AppKit.NSFont.systemFontOfSize_(11))
            lbl.setClickCallback_(lambda n=hk_name: self._select_hotkey(n))
            content.addSubview_(lbl)
            self._hotkey_labels[hk_name] = (lbl, display)

        self._refresh_hotkey_labels()

        # ── Language ──
        y_pos -= 28
        y_pos = self._section_title(content, "LANGUAGE", y_pos)
        y_pos -= 28

        self.lang_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(20, y_pos, 120, 24)
        )
        self.lang_field.setStringValue_(self.config.get("language", "auto"))
        self.lang_field.setFont_(AppKit.NSFont.systemFontOfSize_(12))
        self.lang_field.setBezeled_(True)
        self.lang_field.setBezelStyle_(AppKit.NSTextFieldSquareBezel)
        self.lang_field.setDrawsBackground_(True)
        self.lang_field.setBackgroundColor_(ns_color(0.2, 0.2, 0.22))
        self.lang_field.setTextColor_(TEXT_PRIMARY)
        content.addSubview_(self.lang_field)

        hint = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(150, y_pos + 3, 170, 18)
        )
        hint.setStringValue_('"auto" or ISO code (en, ja, fr...)')
        hint.setBezeled_(False)
        hint.setDrawsBackground_(False)
        hint.setEditable_(False)
        hint.setSelectable_(False)
        hint.setTextColor_(TEXT_DIM)
        hint.setFont_(AppKit.NSFont.systemFontOfSize_(10))
        content.addSubview_(hint)

        # ── Toggles ──
        y_pos -= 32
        y_pos = self._section_title(content, "BEHAVIOUR", y_pos)

        y_pos -= 26
        self._paste_on = self.config.get("auto_paste", True)
        self.paste_lbl = ClickableLabel.alloc().initWithFrame_(
            AppKit.NSMakeRect(20, y_pos, self.WIDTH - 40, 20)
        )
        self.paste_lbl.setBezeled_(False)
        self.paste_lbl.setDrawsBackground_(False)
        self.paste_lbl.setEditable_(False)
        self.paste_lbl.setSelectable_(False)
        self.paste_lbl.setFont_(AppKit.NSFont.systemFontOfSize_(12))
        self.paste_lbl.setClickCallback_(self._toggle_paste)
        content.addSubview_(self.paste_lbl)
        self._refresh_toggle(self.paste_lbl, "Auto-paste into active app", self._paste_on)

        y_pos -= 26
        self._sound_on = self.config.get("sound_feedback", True)
        self.sound_lbl = ClickableLabel.alloc().initWithFrame_(
            AppKit.NSMakeRect(20, y_pos, self.WIDTH - 40, 20)
        )
        self.sound_lbl.setBezeled_(False)
        self.sound_lbl.setDrawsBackground_(False)
        self.sound_lbl.setEditable_(False)
        self.sound_lbl.setSelectable_(False)
        self.sound_lbl.setFont_(AppKit.NSFont.systemFontOfSize_(12))
        self.sound_lbl.setClickCallback_(self._toggle_sound)
        content.addSubview_(self.sound_lbl)
        self._refresh_toggle(self.sound_lbl, "Sound feedback", self._sound_on)

        # ── Save button ──
        save_lbl = ClickableLabel.alloc().initWithFrame_(
            AppKit.NSMakeRect(self.WIDTH - 80, 14, 60, 22)
        )
        save_lbl.setStringValue_("Save")
        save_lbl.setBezeled_(False)
        save_lbl.setDrawsBackground_(False)
        save_lbl.setEditable_(False)
        save_lbl.setSelectable_(False)
        save_lbl.setTextColor_(ns_color(*ACCENT_GREEN))
        save_lbl.setFont_(AppKit.NSFont.systemFontOfSize_weight_(13, AppKit.NSFontWeightSemibold))
        save_lbl.setAlignment_(AppKit.NSTextAlignmentRight)
        save_lbl.setClickCallback_(self._save)
        content.addSubview_(save_lbl)

    def _section_title(self, content, text, y_pos):
        lbl = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(20, y_pos, self.WIDTH - 40, 18)
        )
        lbl.setStringValue_(text)
        lbl.setBezeled_(False)
        lbl.setDrawsBackground_(False)
        lbl.setEditable_(False)
        lbl.setSelectable_(False)
        lbl.setTextColor_(TEXT_DIM)
        lbl.setFont_(AppKit.NSFont.systemFontOfSize_weight_(10, AppKit.NSFontWeightSemibold))
        content.addSubview_(lbl)
        return y_pos

    def _select_mode(self, mode):
        self._selected_mode = mode
        self._refresh_mode_labels()

    def _refresh_mode_labels(self):
        for mode, (lbl, text) in self._mode_labels.items():
            if mode == self._selected_mode:
                lbl.setStringValue_(f"● {text}")
                lbl.setTextColor_(TEXT_PRIMARY)
            else:
                lbl.setStringValue_(f"○ {text}")
                lbl.setTextColor_(TEXT_SECONDARY)

    def _select_hotkey(self, name):
        self._selected_hotkey = name
        self._refresh_hotkey_labels()

    def _refresh_hotkey_labels(self):
        for name, (lbl, display) in self._hotkey_labels.items():
            if name == self._selected_hotkey:
                lbl.setStringValue_(f"● {display}  ({name})")
                lbl.setTextColor_(TEXT_PRIMARY)
            else:
                lbl.setStringValue_(f"○ {display}  ({name})")
                lbl.setTextColor_(TEXT_SECONDARY)

    def _toggle_paste(self):
        self._paste_on = not self._paste_on
        self._refresh_toggle(self.paste_lbl, "Auto-paste into active app", self._paste_on)

    def _toggle_sound(self):
        self._sound_on = not self._sound_on
        self._refresh_toggle(self.sound_lbl, "Sound feedback", self._sound_on)

    def _refresh_toggle(self, lbl, text, on):
        if on:
            lbl.setStringValue_(f"✓  {text}")
            lbl.setTextColor_(ns_color(*ACCENT_GREEN))
        else:
            lbl.setStringValue_(f"✗  {text}")
            lbl.setTextColor_(TEXT_SECONDARY)

    def _save(self):
        self.config["recording_mode"] = self._selected_mode
        self.config["hotkey"] = self._selected_hotkey
        self.config["language"] = self.lang_field.stringValue().strip() or "auto"
        self.config["auto_paste"] = self._paste_on
        self.config["sound_feedback"] = self._sound_on
        save_config(self.config)
        self.window.close()
        if self.on_save:
            self.on_save()

    def show(self):
        self.window.makeKeyAndOrderFront_(None)


# ─── Main Panel ────────────────────────────────────────────────────────────

class DictationPanel:
    """Persistent floating panel with status, history, and vocabulary."""

    PANEL_WIDTH = 300
    PANEL_HEIGHT = 440

    def __init__(self):
        screen = AppKit.NSScreen.mainScreen().frame()
        x = screen.size.width - self.PANEL_WIDTH - 20
        y = screen.size.height - self.PANEL_HEIGHT - 60

        # Use a regular NSWindow so it participates in Mission Control
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(x, y, self.PANEL_WIDTH, self.PANEL_HEIGHT),
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self.window.setLevel_(AppKit.NSNormalWindowLevel)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.window.setHasShadow_(True)
        self.window.setMovableByWindowBackground_(True)
        # Managed = participates in Mission Control / Exposé like a normal window
        self.window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorManaged
        )
        self.window.setHidesOnDeactivate_(False)

        # Background
        content = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, self.PANEL_WIDTH, self.PANEL_HEIGHT)
        )
        self.window.setContentView_(content)

        self.bg = RoundedView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, self.PANEL_WIDTH, self.PANEL_HEIGHT)
        )
        content.addSubview_(self.bg)

        y_pos = self.PANEL_HEIGHT

        # ── Title bar with close button ──
        y_pos -= 40
        title = self._make_label("Cohere Dictation", 16, AppKit.NSFontWeightBold, TEXT_PRIMARY)
        title.setFrame_(AppKit.NSMakeRect(16, y_pos, 200, 24))
        content.addSubview_(title)

        close_btn = ClickableLabel.alloc().initWithFrame_(
            AppKit.NSMakeRect(self.PANEL_WIDTH - 36, y_pos + 2, 20, 20)
        )
        close_btn.setStringValue_("✕")
        close_btn.setBezeled_(False)
        close_btn.setDrawsBackground_(False)
        close_btn.setEditable_(False)
        close_btn.setSelectable_(False)
        close_btn.setTextColor_(TEXT_DIM)
        close_btn.setFont_(AppKit.NSFont.systemFontOfSize_weight_(14, AppKit.NSFontWeightMedium))
        close_btn.setAlignment_(AppKit.NSTextAlignmentCenter)
        close_btn.setClickCallback_(self.hide)
        content.addSubview_(close_btn)

        hint = self._make_label("⌥D", 11, AppKit.NSFontWeightRegular, TEXT_DIM)
        hint.setFrame_(AppKit.NSMakeRect(self.PANEL_WIDTH - 90, y_pos + 2, 50, 18))
        hint.setAlignment_(AppKit.NSTextAlignmentRight)
        content.addSubview_(hint)

        # ── Status bar ──
        y_pos -= 36
        self.status_dot = DotView.alloc().initWithFrame_(
            AppKit.NSMakeRect(16, y_pos + 6, 12, 12)
        )
        content.addSubview_(self.status_dot)

        self.status_label = self._make_label("Loading...", 13, AppKit.NSFontWeightMedium, TEXT_PRIMARY)
        self.status_label.setFrame_(AppKit.NSMakeRect(36, y_pos, self.PANEL_WIDTH - 52, 22))
        content.addSubview_(self.status_label)

        # Recording mode hint
        y_pos -= 20
        self.mode_label = self._make_label("", 10, AppKit.NSFontWeightRegular, TEXT_DIM)
        self.mode_label.setFrame_(AppKit.NSMakeRect(36, y_pos, self.PANEL_WIDTH - 52, 16))
        content.addSubview_(self.mode_label)

        # ── Separator ──
        y_pos -= 10
        content.addSubview_(SeparatorView.alloc().initWithFrame_(
            AppKit.NSMakeRect(16, y_pos, self.PANEL_WIDTH - 32, 1)
        ))

        # ── History section ──
        y_pos -= 22
        hist_title = self._make_label("HISTORY", 10, AppKit.NSFontWeightSemibold, TEXT_DIM)
        hist_title.setFrame_(AppKit.NSMakeRect(16, y_pos, 100, 16))
        content.addSubview_(hist_title)

        hist_hint = self._make_label("click to copy", 9, AppKit.NSFontWeightRegular, TEXT_DIM)
        hist_hint.setFrame_(AppKit.NSMakeRect(self.PANEL_WIDTH - 100, y_pos, 84, 16))
        hist_hint.setAlignment_(AppKit.NSTextAlignmentRight)
        content.addSubview_(hist_hint)

        self.history_labels = []
        for i in range(5):
            y_pos -= 22
            lbl = ClickableLabel.alloc().initWithFrame_(
                AppKit.NSMakeRect(16, y_pos, self.PANEL_WIDTH - 32, 18)
            )
            lbl.setStringValue_("")
            lbl.setBezeled_(False)
            lbl.setDrawsBackground_(False)
            lbl.setEditable_(False)
            lbl.setSelectable_(False)
            lbl.setTextColor_(TEXT_SECONDARY)
            lbl.setFont_(AppKit.NSFont.systemFontOfSize_weight_(12, AppKit.NSFontWeightRegular))
            lbl.setLineBreakMode_(AppKit.NSLineBreakByTruncatingTail)
            idx = i
            lbl.setClickCallback_(lambda idx=idx: self._copy_history(idx))
            content.addSubview_(lbl)
            self.history_labels.append(lbl)

        self._update_history_empty()

        # ── Separator ──
        y_pos -= 12
        content.addSubview_(SeparatorView.alloc().initWithFrame_(
            AppKit.NSMakeRect(16, y_pos, self.PANEL_WIDTH - 32, 1)
        ))

        # ── Vocabulary section ──
        y_pos -= 22
        vocab_title = self._make_label("VOCABULARY", 10, AppKit.NSFontWeightSemibold, TEXT_DIM)
        vocab_title.setFrame_(AppKit.NSMakeRect(16, y_pos, 100, 16))
        content.addSubview_(vocab_title)

        self.vocab_labels = []
        for i in range(6):
            y_pos -= 20
            lbl = self._make_label("", 11, AppKit.NSFontWeightRegular, TEXT_DIM)
            lbl.setFrame_(AppKit.NSMakeRect(16, y_pos, self.PANEL_WIDTH - 32, 16))
            lbl.setLineBreakMode_(AppKit.NSLineBreakByTruncatingTail)
            content.addSubview_(lbl)
            self.vocab_labels.append(lbl)

        # ── Bottom buttons ──
        edit_btn = ClickableLabel.alloc().initWithFrame_(
            AppKit.NSMakeRect(16, 12, 120, 16)
        )
        edit_btn.setStringValue_("Edit vocabulary...")
        edit_btn.setBezeled_(False)
        edit_btn.setDrawsBackground_(False)
        edit_btn.setEditable_(False)
        edit_btn.setSelectable_(False)
        edit_btn.setTextColor_(ns_color(*ACCENT_BLUE))
        edit_btn.setFont_(AppKit.NSFont.systemFontOfSize_weight_(11, AppKit.NSFontWeightRegular))
        edit_btn.setClickCallback_(self._open_vocab_editor)
        content.addSubview_(edit_btn)

        config_btn = ClickableLabel.alloc().initWithFrame_(
            AppKit.NSMakeRect(self.PANEL_WIDTH - 80, 12, 64, 16)
        )
        config_btn.setStringValue_("Settings...")
        config_btn.setBezeled_(False)
        config_btn.setDrawsBackground_(False)
        config_btn.setEditable_(False)
        config_btn.setSelectable_(False)
        config_btn.setTextColor_(ns_color(*ACCENT_BLUE))
        config_btn.setFont_(AppKit.NSFont.systemFontOfSize_weight_(11, AppKit.NSFontWeightRegular))
        config_btn.setAlignment_(AppKit.NSTextAlignmentRight)
        config_btn.setClickCallback_(self._open_settings_editor)
        content.addSubview_(config_btn)

        self.history = []
        self._vocab_editor = None
        self._settings_editor = None
        self._on_settings_changed = None

    def _make_label(self, text, size, weight, color):
        label = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSZeroRect)
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setTextColor_(color)
        label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(size, weight))
        return label

    def _open_vocab_editor(self):
        self._vocab_editor = VocabEditor(on_save=self._on_vocab_saved)
        self._vocab_editor.show()

    def _on_vocab_saved(self):
        self.set_vocabulary(load_vocabulary())
        if self._on_settings_changed:
            self._on_settings_changed()

    def _open_settings_editor(self):
        self._settings_editor = SettingsEditor(load_config(), on_save=self._on_settings_saved)
        self._settings_editor.show()

    def _on_settings_saved(self):
        if self._on_settings_changed:
            self._on_settings_changed()

    def _copy_history(self, idx):
        if idx < len(self.history):
            subprocess.run(["pbcopy"], input=self.history[idx].encode(), check=True)
            self.set_status("done", "Copied to clipboard!")
            def _r():
                time.sleep(1.5)
                self.set_status("ready")
            threading.Thread(target=_r, daemon=True).start()

    def show(self):
        self.window.orderFront_(None)

    def hide(self):
        self.window.orderOut_(None)

    def toggle(self):
        if self.window.isVisible():
            self.hide()
        else:
            self.show()

    def set_mode_hint(self, mode):
        hints = {
            MODE_AUTO: "Mode: auto-stop on silence",
            MODE_MANUAL: "Mode: ⌥D to start & stop",
            MODE_PUSH: "Mode: hold ⌥D, release to stop",
        }
        def _u():
            self.mode_label.setStringValue_(hints.get(mode, ""))
        AppHelper.callAfter(_u)

    def set_status(self, state, text=None):
        def _u():
            dot_colors = {
                "loading": ACCENT_GRAY, "ready": ACCENT_GREEN,
                "recording": ACCENT_RED, "transcribing": ACCENT_BLUE,
                "done": ACCENT_GREEN, "error": ACCENT_ORANGE,
            }
            labels = {
                "loading": text or "Loading model...",
                "ready": "Ready — ⌥D to dictate",
                "recording": text or "Recording...",
                "transcribing": "Transcribing...",
                "done": text or "Done!",
                "error": text or "Error",
            }
            dr, dg, db = dot_colors.get(state, ACCENT_GRAY)
            self.status_dot.updateColor_g_b_(dr, dg, db)
            self.status_label.setStringValue_(labels.get(state, state))
        AppHelper.callAfter(_u)

    def add_history(self, text):
        self.history.insert(0, text)
        self.history = self.history[:5]
        def _u():
            for i, lbl in enumerate(self.history_labels):
                if i < len(self.history):
                    lbl.setStringValue_(self.history[i])
                    lbl.setTextColor_(TEXT_SECONDARY)
                else:
                    lbl.setStringValue_("")
        AppHelper.callAfter(_u)

    def _update_history_empty(self):
        self.history_labels[0].setStringValue_("No transcriptions yet")
        self.history_labels[0].setTextColor_(TEXT_DIM)
        for lbl in self.history_labels[1:]:
            lbl.setStringValue_("")

    def set_vocabulary(self, vocab):
        def _u():
            for i, lbl in enumerate(self.vocab_labels):
                if i < len(vocab):
                    wrong, correct = vocab[i]
                    lbl.setStringValue_(f"{wrong} → {correct}")
                elif i == 0 and len(vocab) == 0:
                    lbl.setStringValue_("No corrections set")
                else:
                    lbl.setStringValue_("")
        AppHelper.callAfter(_u)


# ─── Compact Status Pill ──────────────────────────────────────────────────

class StatusPill:
    """Small pill overlay that appears center-top during recording."""

    def __init__(self):
        self.width = 260
        self.height = 40
        screen = AppKit.NSScreen.mainScreen().frame()
        x = (screen.size.width - self.width) / 2
        y = screen.size.height - 70

        self.window = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(x, y, self.width, self.height),
            AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self.window.setLevel_(AppKit.NSFloatingWindowLevel + 1)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.window.setHasShadow_(True)
        self.window.setIgnoresMouseEvents_(True)
        self.window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
        )

        content = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, self.width, self.height)
        )
        self.window.setContentView_(content)

        self.bg = RoundedView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, self.width, self.height)
        )
        content.addSubview_(self.bg)

        self.dot = DotView.alloc().initWithFrame_(AppKit.NSMakeRect(14, 12, 14, 14))
        content.addSubview_(self.dot)

        self.label = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(36, 8, self.width - 50, 22)
        )
        self.label.setStringValue_("")
        self.label.setBezeled_(False)
        self.label.setDrawsBackground_(False)
        self.label.setEditable_(False)
        self.label.setSelectable_(False)
        self.label.setTextColor_(TEXT_PRIMARY)
        self.label.setFont_(AppKit.NSFont.systemFontOfSize_weight_(13, AppKit.NSFontWeightMedium))
        content.addSubview_(self.label)

    def show(self, state, text):
        def _u():
            colors = {
                "recording": (0.45, 0.05, 0.05, 0.93),
                "transcribing": (0.12, 0.12, 0.3, 0.93),
                "done": (0.05, 0.28, 0.1, 0.93),
                "error": (0.4, 0.15, 0.0, 0.93),
            }
            dot_colors = {
                "recording": ACCENT_RED, "transcribing": ACCENT_BLUE,
                "done": ACCENT_GREEN, "error": ACCENT_ORANGE,
            }
            r, g, b, a = colors.get(state, BG_COLOR)
            self.bg.updateColor_g_b_a_(r, g, b, a)
            dr, dg, db = dot_colors.get(state, ACCENT_GRAY)
            self.dot.updateColor_g_b_(dr, dg, db)
            self.label.setStringValue_(text)
            self.window.orderFront_(None)
        AppHelper.callAfter(_u)

    def hide(self):
        AppHelper.callAfter(lambda: self.window.orderOut_(None))

    def show_and_hide(self, state, text, delay=2.0):
        self.show(state, text)
        def _later():
            time.sleep(delay)
            self.hide()
        threading.Thread(target=_later, daemon=True).start()


# ─── Dictation Engine ──────────────────────────────────────────────────────

class DictationEngine:
    def __init__(self, panel: DictationPanel, pill: StatusPill):
        self.panel = panel
        self.pill = pill
        self._reload_config()

        self.processor = None
        self.model = None
        self.recorder = Recorder(sample_rate=SAMPLE_RATE)
        self.state = "loading"
        self._last_hotkey_time = 0.0
        self._manual_stop_event = threading.Event()

        self.panel._on_settings_changed = self._on_settings_changed

    def _reload_config(self):
        self.cfg = load_config()
        self.vocab = load_vocabulary()
        self.language = self.cfg["language"]
        self.silence_threshold = self.cfg["silence_threshold"]
        self.auto_paste = self.cfg["auto_paste"]
        self.sound_feedback = self.cfg["sound_feedback"]
        self.recording_mode = self.cfg.get("recording_mode", MODE_AUTO)
        self.panel.set_vocabulary(self.vocab)
        self.panel.set_mode_hint(self.recording_mode)

    def _on_settings_changed(self):
        self._reload_config()
        print(f"Settings reloaded. Mode: {self.recording_mode}, Language: {self.language}")

    def load_model(self):
        device = self.cfg["device"]
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        use_float16 = self.cfg.get("float16", True)
        dtype = torch.float16 if use_float16 else torch.float32
        model_id = "CohereLabs/cohere-transcribe-03-2026"

        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        print("Loading model weights...")
        try:
            self.model = CohereAsrForConditionalGeneration.from_pretrained(
                model_id, device_map=device, torch_dtype=dtype,
            )
        except Exception:
            print("float16 failed, falling back to float32...")
            self.model = CohereAsrForConditionalGeneration.from_pretrained(
                model_id, device_map=device, torch_dtype=torch.float32,
            )
        print(f"Model loaded on {device}.")

        if self.cfg.get("warmup", True):
            print("Warming up...")
            self.panel.set_status("loading", "Warming up...")
            dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
            inputs = self.processor(dummy, sampling_rate=SAMPLE_RATE, return_tensors="pt", language="en")
            inputs.to(self.model.device, dtype=self.model.dtype)
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=8)
            print("Warm-up done.")

        self.state = "ready"
        self.panel.set_status("ready")
        print(f"Ready! Mode: {self.recording_mode}. Press Option+D to dictate.")
        self._start_hotkey_listener()

    def _start_hotkey_listener(self):
        """Use Quartz CGEventTap for global hotkey — no TSM threading issues."""
        hotkey_name = self.cfg.get("hotkey", DEFAULT_HOTKEY)
        hotkey_def = HOTKEY_OPTIONS.get(hotkey_name)
        if not hotkey_def:
            print(f"Unknown hotkey '{hotkey_name}', falling back to {DEFAULT_HOTKEY}")
            hotkey_def = HOTKEY_OPTIONS[DEFAULT_HOTKEY]
            hotkey_name = DEFAULT_HOTKEY

        target_keycode, target_modmask, display_label = hotkey_def
        self._hotkey_keycode = target_keycode
        self._hotkey_modmask = target_modmask
        self._hotkey_label = display_label

        def _event_callback(_proxy, event_type, event, _refcon):
            if event_type == Quartz.kCGEventKeyDown:
                keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                flags = Quartz.CGEventGetFlags(event)
                # Check keycode matches, and required modifiers are held
                if keycode == self._hotkey_keycode:
                    if self._hotkey_modmask == 0 or (flags & self._hotkey_modmask):
                        self._on_hotkey_down()
            elif event_type == Quartz.kCGEventKeyUp:
                # Track key release for push-to-talk
                if self.recording_mode == MODE_PUSH and self.state == "recording":
                    keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                    if keycode == self._hotkey_keycode:
                        self._manual_stop_event.set()
            elif event_type == Quartz.kCGEventFlagsChanged:
                # Track modifier release for push-to-talk (only if hotkey uses modifiers)
                if self._hotkey_modmask and self.recording_mode == MODE_PUSH and self.state == "recording":
                    flags = Quartz.CGEventGetFlags(event)
                    if not (flags & self._hotkey_modmask):
                        self._manual_stop_event.set()
            return event

        event_mask = (
            (1 << Quartz.kCGEventKeyDown) |
            (1 << Quartz.kCGEventKeyUp) |
            (1 << Quartz.kCGEventFlagsChanged)
        )

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            event_mask,
            _event_callback,
            None,
        )

        if tap is None:
            print("ERROR: Could not create event tap. Grant Accessibility access in System Settings.")
            return

        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetMain(),
            source,
            Quartz.kCFRunLoopCommonModes,
        )
        Quartz.CGEventTapEnable(tap, True)
        print(f"Global hotkey {display_label} registered via CGEventTap.")

    def _on_hotkey_down(self):
        """Called from main thread (CGEventTap). Dispatch work to background."""
        now = time.time()
        log.info(f"Hotkey pressed (state={self.state}, mode={self.recording_mode})")
        # Capture frontmost app on main thread (safe, uses NSWorkspace)
        prev_app = _get_frontmost_bundle_id()

        if self.recording_mode == MODE_MANUAL:
            if self.state == "recording":
                self._manual_stop_event.set()
                return
            if self.state == "ready":
                self._last_hotkey_time = now
                threading.Thread(target=self._record_and_transcribe, args=(prev_app,), daemon=True).start()
            return

        if self.recording_mode == MODE_PUSH:
            if self.state == "ready":
                self._last_hotkey_time = now
                threading.Thread(target=self._record_and_transcribe, args=(prev_app,), daemon=True).start()
            return

        # MODE_AUTO
        if self.state == "recording" and (now - self._last_hotkey_time) < 0.5:
            self._last_hotkey_time = now
            self.recorder.stop()
            self.state = "ready"
            self.panel.set_status("ready")
            self.pill.show_and_hide("error", "Cancelled")
            print("Cancelled.")
            return

        self._last_hotkey_time = now
        if self.state == "ready":
            threading.Thread(target=self._record_and_transcribe, args=(prev_app,), daemon=True).start()

    def _record_and_transcribe(self, previous_app_id):
        self.state = "recording"
        if self.sound_feedback:
            self._play_sound()

        mode = self.recording_mode
        mode_hints = {
            MODE_AUTO: "Recording... (auto-stop)",
            MODE_MANUAL: "Recording... ⌥D to stop",
            MODE_PUSH: "Recording... release to stop",
        }
        self.panel.set_status("recording", mode_hints.get(mode))
        self.pill.show("recording", mode_hints.get(mode))

        if mode == MODE_AUTO:
            audio = self.recorder.record_until_silence(
                silence_threshold=self.silence_threshold,
            )
        else:
            self._manual_stop_event.clear()
            self.recorder.start()
            self._manual_stop_event.wait(timeout=self.cfg.get("max_recording_duration", 120))
            audio = self.recorder.stop()

        # Push mode: undo any characters typed while holding the hotkey
        if mode == MODE_PUSH and previous_app_id:
            try:
                subprocess.run([
                    "osascript",
                    "-e", f'tell application id "{previous_app_id}" to activate',
                    "-e", 'delay 0.1',
                    "-e", 'tell application "System Events" to keystroke "z" using command down',
                ], capture_output=True, timeout=3)
            except Exception:
                pass

        duration = len(audio) / SAMPLE_RATE
        if duration < 0.3:
            self.state = "ready"
            self.panel.set_status("ready")
            self.pill.show_and_hide("error", "Too short")
            return

        self.state = "transcribing"
        self.panel.set_status("transcribing")
        self.pill.show("transcribing", "Transcribing...")
        print(f"Recorded {duration:.1f}s. Transcribing...")

        filtered, _ = filter_speech(audio, sample_rate=SAMPLE_RATE)
        if filtered is None:
            self.state = "ready"
            self.panel.set_status("ready")
            self.pill.show_and_hide("error", "No speech detected")
            return

        language = self.language
        if language == "auto":
            device_str = str(self.model.device)
            language = detect_language(filtered, sample_rate=SAMPLE_RATE, device=device_str)

        inputs = self.processor(filtered, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=language)
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs.to(self.model.device, dtype=self.model.dtype)

        outputs = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(
            outputs, skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index, language=language,
        )
        if isinstance(text, list):
            text = text[0]
        text = text.strip()

        if not text:
            self.state = "ready"
            self.panel.set_status("ready")
            self.pill.show_and_hide("error", "No speech detected")
            return

        if self.vocab:
            text = apply_vocabulary(text, self.vocab)

        self.panel.add_history(text)
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
        log.info(f"Copied to clipboard: {text[:50]}")

        # Auto-paste: activate previous app + Cmd+V via CGEvent
        if self.auto_paste and previous_app_id:
            log.info(f"Auto-pasting into {previous_app_id}")
            _activate_and_paste(previous_app_id)
        elif not self.auto_paste:
            log.info("Auto-paste disabled")
        elif not previous_app_id:
            log.warning("No previous app to paste into")

        if self.sound_feedback:
            self._play_sound()

        lang_label = f"[{language}] " if self.language == "auto" else ""
        preview = text[:35] + ("..." if len(text) > 35 else "")
        self.pill.show_and_hide("done", f"{lang_label}{preview}")
        self.panel.set_status("done", f"Pasted: {preview}")
        print(f"Pasted [{language}]: {text}")

        def _reset():
            time.sleep(2)
            if self.state != "recording":
                self.panel.set_status("ready")
        threading.Thread(target=_reset, daemon=True).start()

        self.state = "ready"

    def _play_sound(self):
        subprocess.Popen(
            ["afplay", "/System/Library/Sounds/Tink.aiff"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


# ─── App Delegate ──────────────────────────────────────────────────────────

class AppDelegate(AppKit.NSObject):
    def applicationDidFinishLaunching_(self, notification):
        self.panel = DictationPanel()
        self.pill = StatusPill()

        self.panel.show()
        self.panel.set_status("loading")

        # Check accessibility (needed for auto-paste)
        _check_accessibility()

        self.engine = DictationEngine(self.panel, self.pill)
        threading.Thread(target=self.engine.load_model, daemon=True).start()

    def applicationShouldHandleReopen_hasVisibleWindows_(self, sender, flag):
        self.panel.toggle()
        return True

    def applicationShouldTerminateAfterLastWindowClosed_(self, sender):
        return False


# ─── Main ──────────────────────────────────────────────────────────────────

def _build_menu_bar():
    """Create a standard macOS menu bar: app name, File, Edit, Window, Help."""
    app = AppKit.NSApplication.sharedApplication()
    mainMenu = AppKit.NSMenu.alloc().init()

    APP_NAME = "Cohere Dictation"

    # ── Application menu (shows as the bold name top-left) ──
    appMenuItem = AppKit.NSMenuItem.alloc().init()
    mainMenu.addItem_(appMenuItem)
    appMenu = AppKit.NSMenu.alloc().init()
    appMenuItem.setSubmenu_(appMenu)

    # About
    appMenu.addItemWithTitle_action_keyEquivalent_(f"About {APP_NAME}", "orderFrontStandardAboutPanel:", "")
    appMenu.addItem_(AppKit.NSMenuItem.separatorItem())

    # Hide
    appMenu.addItemWithTitle_action_keyEquivalent_(f"Hide {APP_NAME}", "hide:", "h")
    # Hide Others
    hideOthers = appMenu.addItemWithTitle_action_keyEquivalent_("Hide Others", "hideOtherApplications:", "h")
    hideOthers.setKeyEquivalentModifierMask_(
        AppKit.NSEventModifierFlagCommand | AppKit.NSEventModifierFlagOption
    )
    # Show All
    appMenu.addItemWithTitle_action_keyEquivalent_("Show All", "unhideAllApplications:", "")
    appMenu.addItem_(AppKit.NSMenuItem.separatorItem())

    # Quit
    appMenu.addItemWithTitle_action_keyEquivalent_(f"Quit {APP_NAME}", "terminate:", "q")

    # ── Edit menu (for Cmd+C/V/X/A in text fields) ──
    editMenuItem = AppKit.NSMenuItem.alloc().init()
    mainMenu.addItem_(editMenuItem)
    editMenu = AppKit.NSMenu.alloc().initWithTitle_("Edit")
    editMenuItem.setSubmenu_(editMenu)

    editMenu.addItemWithTitle_action_keyEquivalent_("Undo", "undo:", "z")
    editMenu.addItemWithTitle_action_keyEquivalent_("Redo", "redo:", "Z")
    editMenu.addItem_(AppKit.NSMenuItem.separatorItem())
    editMenu.addItemWithTitle_action_keyEquivalent_("Cut", "cut:", "x")
    editMenu.addItemWithTitle_action_keyEquivalent_("Copy", "copy:", "c")
    editMenu.addItemWithTitle_action_keyEquivalent_("Paste", "paste:", "v")
    editMenu.addItemWithTitle_action_keyEquivalent_("Select All", "selectAll:", "a")

    # ── Window menu ──
    windowMenuItem = AppKit.NSMenuItem.alloc().init()
    mainMenu.addItem_(windowMenuItem)
    windowMenu = AppKit.NSMenu.alloc().initWithTitle_("Window")
    windowMenuItem.setSubmenu_(windowMenu)

    windowMenu.addItemWithTitle_action_keyEquivalent_("Minimize", "performMiniaturize:", "m")
    windowMenu.addItemWithTitle_action_keyEquivalent_("Close", "performClose:", "w")
    app.setWindowsMenu_(windowMenu)

    app.setMainMenu_(mainMenu)

    # Set the process name so the menu bar shows "Cohere Dictation" not "Python"
    try:
        from Foundation import NSProcessInfo
        NSProcessInfo.processInfo().setProcessName_(APP_NAME)
    except Exception:
        pass


if __name__ == "__main__":
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)

    _build_menu_bar()

    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)

    log.info("Starting Cohere Dictation")
    AppHelper.runEventLoop()
