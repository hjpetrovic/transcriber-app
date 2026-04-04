#!/usr/bin/env python3
"""Generate the Cohere Dictation app icon as .icns."""

import math
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path


def make_icon(size):
    """Return RGBA bytes for a square icon of the given size."""
    pixels = bytearray(size * size * 4)
    cx, cy = size / 2, size / 2
    radius = size * 0.42  # rounded-rect corner radius
    inset = size * 0.08   # padding from edge

    for y in range(size):
        for x in range(size):
            idx = (y * size + x) * 4

            # ── Rounded rectangle mask ──
            rx, ry = x - inset, y - inset
            rw, rh = size - 2 * inset, size - 2 * inset
            cr = rw * 0.22  # corner radius

            # Distance to rounded rect
            inside = True
            if rx < 0 or ry < 0 or rx > rw or ry > rh:
                inside = False
            elif rx < cr and ry < cr:
                inside = math.hypot(rx - cr, ry - cr) <= cr
            elif rx > rw - cr and ry < cr:
                inside = math.hypot(rx - (rw - cr), ry - cr) <= cr
            elif rx < cr and ry > rh - cr:
                inside = math.hypot(rx - cr, ry - (rh - cr)) <= cr
            elif rx > rw - cr and ry > rh - cr:
                inside = math.hypot(rx - (rw - cr), ry - (rh - cr)) <= cr

            if not inside:
                pixels[idx:idx+4] = b'\x00\x00\x00\x00'
                continue

            # ── Background gradient (dark charcoal → deep blue-black) ──
            t = (y - inset) / rh  # 0 at top, 1 at bottom
            bg_r = int(25 + t * 8)
            bg_g = int(28 + t * 12)
            bg_b = int(42 + t * 20)

            # ── Subtle radial glow in center ──
            dist_center = math.hypot(x - cx, y - cy) / (size * 0.5)
            glow = max(0, 1 - dist_center * 1.3) ** 2 * 0.15
            bg_r = min(255, int(bg_r + glow * 40))
            bg_g = min(255, int(bg_g + glow * 80))
            bg_b = min(255, int(bg_b + glow * 160))

            r, g, b = bg_r, bg_g, bg_b

            # ── Waveform bars ──
            # 7 vertical bars in the center, varying heights
            bar_count = 7
            bar_width = size * 0.055
            bar_gap = size * 0.035
            total_w = bar_count * bar_width + (bar_count - 1) * bar_gap
            bar_start_x = cx - total_w / 2

            # Heights as fraction of available space (tallest = 0.55)
            heights = [0.18, 0.32, 0.48, 0.55, 0.42, 0.28, 0.15]

            for i, h_frac in enumerate(heights):
                bx = bar_start_x + i * (bar_width + bar_gap)
                if bx <= x < bx + bar_width:
                    bar_h = rh * h_frac
                    bar_top = cy - bar_h / 2
                    bar_bot = cy + bar_h / 2

                    if bar_top <= y <= bar_bot:
                        # Gradient on each bar: green at center → blue at edges
                        bar_t = abs(y - cy) / (bar_h / 2) if bar_h > 0 else 0
                        # Green core
                        r = int(60 + (1 - bar_t) * 40)
                        g = int(200 + (1 - bar_t) * 55)
                        b = int(120 + bar_t * 100)

                        # Slight per-bar color shift (center bars more vibrant)
                        center_factor = 1 - abs(i - 3) / 3.5
                        r = min(255, int(r * (0.8 + 0.2 * center_factor)))
                        g = min(255, int(g * (0.85 + 0.15 * center_factor)))
                        b = min(255, int(b * (0.7 + 0.3 * center_factor)))

                        # Rounded bar ends
                        end_radius = bar_width / 2
                        bar_cx = bx + bar_width / 2
                        if y < bar_top + end_radius:
                            if math.hypot(x - bar_cx, y - (bar_top + end_radius)) > end_radius:
                                r, g, b = bg_r, bg_g, bg_b
                        elif y > bar_bot - end_radius:
                            if math.hypot(x - bar_cx, y - (bar_bot - end_radius)) > end_radius:
                                r, g, b = bg_r, bg_g, bg_b
                    break

            pixels[idx] = r
            pixels[idx+1] = g
            pixels[idx+2] = b
            pixels[idx+3] = 255

    return bytes(pixels)


def write_png(filepath, width, height, rgba_data):
    """Write minimal PNG file from RGBA data."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + c + struct.pack('>I', crc)

    raw_rows = b''
    for row in range(height):
        raw_rows += b'\x00'  # filter: none
        raw_rows += rgba_data[row * width * 4:(row + 1) * width * 4]

    with open(filepath, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        ihdr = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
        f.write(chunk(b'IHDR', ihdr))
        f.write(chunk(b'IDAT', zlib.compress(raw_rows, 9)))
        f.write(chunk(b'IEND', b''))


def downscale(rgba, src_size, dst_size):
    """Simple box-filter downscale."""
    ratio = src_size // dst_size
    out = bytearray(dst_size * dst_size * 4)
    for dy in range(dst_size):
        for dx in range(dst_size):
            rt = gt = bt = at = 0
            for sy in range(dy * ratio, (dy + 1) * ratio):
                for sx in range(dx * ratio, (dx + 1) * ratio):
                    si = (sy * src_size + sx) * 4
                    rt += rgba[si]
                    gt += rgba[si+1]
                    bt += rgba[si+2]
                    at += rgba[si+3]
            n = ratio * ratio
            di = (dy * dst_size + dx) * 4
            out[di] = rt // n
            out[di+1] = gt // n
            out[di+2] = bt // n
            out[di+3] = at // n
    return bytes(out)


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "AppIcon.icns"

    print("Generating icon at 1024x1024...")
    master = make_icon(1024)

    with tempfile.TemporaryDirectory() as tmpdir:
        iconset = Path(tmpdir) / "AppIcon.iconset"
        iconset.mkdir()

        # Required sizes for macOS .icns
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        for s in sizes:
            print(f"  Scaling to {s}x{s}...")
            scaled = downscale(master, 1024, s) if s < 1024 else master

            if s <= 512:
                write_png(str(iconset / f"icon_{s}x{s}.png"), s, s, scaled)
            if s >= 32:
                # The @2x variant for the size below
                half = s // 2
                if half >= 16:
                    write_png(str(iconset / f"icon_{half}x{half}@2x.png"), s, s, scaled)

        print("Building .icns...")
        result = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", output],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"iconutil error: {result.stderr}")
            sys.exit(1)

    print(f"Done: {output}")


if __name__ == "__main__":
    main()
