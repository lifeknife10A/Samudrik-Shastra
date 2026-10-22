#!/usr/bin/env python3
"""
Hindi to English Book Translator for Samudrik Shastra
Translates transcribed Hindi pages into fluent, domain-aware English using Gemini Flash.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from google import genai

DEFAULT_INPUT_DIR = Path.home() / "Downloads" / "ocr_output"
DEFAULT_OUTPUT_DIR = Path.home() / "Downloads" / "translation_output"

TRANSLATION_PROMPT = """You are an expert translator specializing in classical Indian literature, Jyotish (Vedic astrology), and Samudrik Shastra (palmistry and physiognomy).
Translate the following Hindi text into fluent, natural, and scholarly English.

Guidelines:
1. Maintain formal tone, paragraph flow, and structure.
2. For specialized palmistry terms, use standard English palmistry equivalents:
   - सूर्य रेखा -> Sun Line / Apollo Line
   - भाग्य रेखा -> Fate Line / Saturn Line
   - हृदय रेखा -> Heart Line
   - जीवन रेखा -> Life Line
   - मङ्गल का स्थान / क्षेत्र -> Mount of Mars
   - गुरु पर्वत -> Mount of Jupiter
   - शुक्र पर्वत -> Mount of Venus
   - चन्द्र पर्वत -> Mount of Moon
   - द्वीप चिन्ह -> Island mark
   - वर्ग चिन्ह -> Square mark
3. If Sanskrit verses (shlokas) are present, translate their poetic meaning faithfully.
4. Output ONLY the English translation without conversational filler or markdown code backtick blocks.

Hindi Text to Translate:
"""

def get_gemini_client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("\n[ERROR] No GEMINI_API_KEY found.")
        print("Pass it via: --key \"your_api_key_here\"\n")
        sys.exit(1)
    return genai.Client(api_key=key)

def translate_page(client: genai.Client, text: str, model_name: str = "gemini-3.1-flash-lite") -> str:
    if not text.strip():
        return ""
    
    models_to_try = [model_name, "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview", "gemini-3-flash-preview"]
    models_to_try = list(dict.fromkeys(models_to_try))

    last_err = None
    for m in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=TRANSLATION_PROMPT + "\n" + text,
                )
                if response.text:
                    return response.text.strip()
            except Exception as e:
                last_err = e
                wait_time = 2.5 * (attempt + 1)
                time.sleep(wait_time)

    raise last_err or RuntimeError("Failed to translate page")

def main():
    parser = argparse.ArgumentParser(description="Translate Hindi Samudrik Shastra into English")
    parser.add_argument("--key", type=str, default=None, help="Gemini API Key")
    parser.add_argument("--indir", type=str, default=str(DEFAULT_INPUT_DIR), help="Input directory containing page_XXX.txt")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for translations")
    parser.add_argument("--start", type=int, default=1, help="Start page (1-indexed)")
    parser.add_argument("--end", type=int, default=328, help="End page (1-indexed)")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", help="Gemini model name")
    parser.add_argument("--delay", type=float, default=3.8, help="Delay in seconds between requests for Free Tier rate limit")
    args = parser.parse_args()

    in_dir = Path(args.indir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_file = out_dir / "Samudrik_Shastra_English.md"

    client = get_gemini_client(args.key)

    print("=" * 60)
    print("🌐 Hindi to English Translation: Samudrik Shastra")
    print(f"📁 Source: {in_dir}")
    print(f"📑 Pages: {args.start} to {args.end}")
    print(f"💾 Output directory: {out_dir}")
    print("=" * 60)

    for p in range(args.start, args.end + 1):
        in_file = in_dir / f"page_{p:03d}.txt"
        out_file = out_dir / f"page_{p:03d}_en.md"

        if not in_file.exists():
            continue

        # Resume support: skip already translated pages
        if out_file.exists() and out_file.stat().st_size > 0:
            print(f"⏩ Page {p}/{args.end} already translated. Skipping.")
            continue

        raw_hindi = in_file.read_text(encoding="utf-8").strip()
        if not raw_hindi:
            out_file.write_text("*(Blank page or illustration)*\n", encoding="utf-8")
            continue

        print(f"🌐 Translating page {p}/{args.end}...", end="", flush=True)

        try:
            translation = translate_page(client, raw_hindi, model_name=args.model)
            out_file.write_text(translation, encoding="utf-8")
            print(" ✅ Done!")

            if p < args.end:
                time.sleep(args.delay)

        except Exception as e:
            print(f" ❌ Failed: {e}")
            time.sleep(8)

    # Compile the complete book in order
    print("\n📚 Compiling complete English book...")
    with open(combined_file, "w", encoding="utf-8") as outfile:
        outfile.write("# Samudrik Shastra (The Science of Indian Palmistry & Physiognomy)\n")
        outfile.write("### English Translation\n\n---\n\n")
        for p in range(args.start, args.end + 1):
            p_file = out_dir / f"page_{p:03d}_en.md"
            if p_file.exists():
                text = p_file.read_text(encoding="utf-8").strip()
                outfile.write(f"## Page {p}\n\n")
                outfile.write(text + "\n\n---\n\n")

    print(f"🎉 Complete English book saved to: {combined_file}")

if __name__ == "__main__":
    main()
