#!/usr/bin/env python3
"""
Vision benchmark: presence test on the book's own hand figures.

For each crop (e.g. Heart_Line_Hand_3.png) the plate's own line is known, so
the only question is whether the vision model marks that line "present".

Imports the real reader's prompt, schema and model, so the benchmark
measures the same system the reader uses.

Usage:
    python3 engine/test_vision_hands.py --limit 10
    python3 engine/test_vision_hands.py --all
"""

import json
import base64
import sys
import argparse
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from palm_reader import (OLLAMA_CLIENT, VISION_MODEL, VISION_ANALYSIS_PROMPT,
                            VISION_SCHEMA, MAJOR_LINES)

HANDS_DIR = Path(__file__).parent.parent / "illustrations" / "cropped_hands"
OUT_DIR = Path(__file__).parent / "vision_test_results"
READER_OPTIONS = {"temperature": 0.2, "num_predict": 600, "num_ctx": 8192}
RESIZE_LIMIT = 1024


def plate_line_from_filename(filename):
    """Heart_Line_Hand_3.png -> heart_line"""
    stem = filename.split("_Hand_")[0].lower()
    return stem if stem in MAJOR_LINES else None


def resize_image_bytes(image_bytes):
    try:
        from PIL import Image
        with Image.open(BytesIO(image_bytes)) as img:
            img.thumbnail((RESIZE_LIMIT, RESIZE_LIMIT))
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
    except Exception:
        return image_bytes


def analyze_one(image_path):
    image_bytes = resize_image_bytes(image_path.read_bytes())
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    expected = plate_line_from_filename(image_path.name)
    result = {"file": image_path.name, "expected_line": expected}
    start = time.time()
    try:
        response = OLLAMA_CLIENT.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": VISION_ANALYSIS_PROMPT,
                       "images": [image_base64]}],
            options=READER_OPTIONS,
            format=VISION_SCHEMA,
        )
        result["elapsed_seconds"] = round(time.time() - start, 1)
        data = json.loads(response["message"]["content"])
        result["valid_json"] = True
        result["hand_type"] = data.get("hand_type")
        lines = data.get("lines", {})
        result["lines"] = lines
        if expected:
            result["expected_status"] = lines.get(expected)
            result["expected_present"] = lines.get(expected) == "present"
            result["expected_unclear"] = lines.get(expected) == "unclear"
    except Exception as error:
        result["elapsed_seconds"] = round(time.time() - start, 1)
        result["valid_json"] = False
        result["error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description="Presence test on the book's hand figures")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of images to test (default: all)")
    parser.add_argument("--all", action="store_true", help="Test all images")
    parser.add_argument("--outdir", default=str(OUT_DIR))
    args = parser.parse_args()

    images = sorted(HANDS_DIR.glob("*_Hand_*.png"),
                    key=lambda p: (p.stem.split("_Hand_")[0],
                                   int(p.stem.rsplit("_", 1)[1])))
    if not args.all and args.limit:
        images = images[:args.limit]

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "vision_test_results.json"
    print("Testing %d figures (presence test)" % len(images))

    results = []
    for index, image in enumerate(images, 1):
        print("  [%d/%d] %s..." % (index, len(images), image.name), end="", flush=True)
        result = analyze_one(image)
        results.append(result)
        print(" %s hand=%s expected(%s)=%s" % (
            "OK" if result.get("valid_json") else "FAIL",
            result.get("hand_type"), result.get("expected_line"),
            result.get("expected_status")))
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    total = len(results)
    valid = sum(1 for r in results if r.get("valid_json"))
    applicable = [r for r in results if r.get("expected_line")]
    present = sum(1 for r in applicable if r.get("expected_present"))
    unclear = sum(1 for r in applicable if r.get("expected_unclear"))
    absent = sum(1 for r in applicable
                 if r.get("expected_status") == "absent")
    hands = {}
    for r in results:
        hands[r.get("hand_type")] = hands.get(r.get("hand_type"), 0) + 1

    print("\n" + "=" * 60)
    print("Presence test summary (%d figures):" % total)
    print("  valid JSON: %d/%d" % (valid, total))
    print("  plate's own line marked PRESENT: %d/%d" % (present, len(applicable)))
    print("  plate's own line marked UNCLEAR: %d/%d" % (unclear, len(applicable)))
    print("  plate's own line marked ABSENT:  %d/%d" % (absent, len(applicable)))
    print("  hand side reported: %s" % hands)
    print("  results saved: %s" % out_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
