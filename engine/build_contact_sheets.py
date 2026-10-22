#!/usr/bin/env python3
"""
Build contact sheets: one numbered image per plate for the visual matcher.

Usage:
    python3 engine/build_contact_sheets.py

Output: illustrations/contact_sheets/<Plate>.png
Each sheet shows its crops in grid order with large figure numbers.
"""

import math
from pathlib import Path
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).parent.parent
HANDS_DIR = REPO_ROOT / "illustrations" / "cropped_hands"
OUT_DIR = REPO_ROOT / "illustrations" / "contact_sheets"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Thumb height and label bar size
THUMB_H = 300
LABEL_H = 40
PAD = 8


def collect_plates():
    plates = {}
    for img in HANDS_DIR.glob("*_Hand_*.png"):
        name = img.stem
        parts = name.rsplit("_Hand_", 1)
        if len(parts) != 2:
            continue
        plate_name, hand_num = parts[0], int(parts[1])
        plates.setdefault(plate_name, []).append((hand_num, img))
    for plate_name in plates:
        plates[plate_name].sort(key=lambda x: x[0])
    return plates


def build_sheet(plate_name, hands):
    n = len(hands)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Load thumbs at fixed height
    thumbs = []
    max_w = 0
    for hand_num, img_path in hands:
        im = Image.open(img_path).convert("RGB")
        scale = THUMB_H / im.height
        w = int(im.width * scale)
        im = im.resize((w, THUMB_H))
        thumbs.append((hand_num, im))
        max_w = max(max_w, w)

    cell_w = max_w + PAD * 2
    cell_h = THUMB_H + LABEL_H + PAD * 2
    sheet_w = cols * cell_w + PAD
    sheet_h = rows * cell_h + PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (hand_num, thumb) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x0 = PAD + c * cell_w
        y0 = PAD + r * cell_h
        sheet.paste(thumb, (x0 + PAD, y0 + PAD + LABEL_H))
        draw.text((x0 + PAD, y0 + PAD), f"Fig {hand_num}", fill="black")

    out = OUT_DIR / f"{plate_name}.png"
    sheet.save(out)
    print(f"  {plate_name}: {n} figures -> {out.name} ({sheet_w}x{sheet_h})")
    return out


def main():
    plates = collect_plates()
    print(f"Found {len(plates)} plates")
    for plate_name, hands in sorted(plates.items()):
        build_sheet(plate_name, hands)
    print(f"\nContact sheets saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
