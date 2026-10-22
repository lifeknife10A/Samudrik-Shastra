#!/usr/bin/env python3
"""
Hindi / Devanagari Book OCR using Gemini 2.0 Flash
Designed for vintage letterpress scans like 'Samudrik Shastra'.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import pymupdf as fitz
from google import genai
from google.genai import types

DEFAULT_PDF = Path.home() / "Downloads" / "2015.400540.Samudirk-Shastr.pdf"
DEFAULT_OUTPUT_DIR = Path.home() / "Downloads" / "ocr_output"

OCR_PROMPT = """You are an expert Hindi/Sanskrit scholar and OCR transcriber.
Transcribe all the text from this scanned book page into clean Unicode Devanagari Hindi text.

Guidelines:
1. Transcribe accurately, maintaining paragraphs, headings, and verse (shloka) layouts.
2. In vintage letterpress, some characters or matras (like ि, ी, े, nukta, halant) may be faint or partially broken. Use the surrounding linguistic context to reconstruct them faithfully.
3. If Sanskrit verses (shlokas) appear, preserve their original metre and formatting.
4. Output ONLY the transcribed Devanagari text. Do not add conversational explanations or wrap in markdown backticks.
"""

def get_gemini_client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("\n[ERROR] No GEMINI_API_KEY found.")
        print("Set it in your terminal:")
        print("    export GEMINI_API_KEY=\"your_api_key_here\"")
        print("Or pass it via the command line:")
        print("    python3 ocr_hindi_book.py --key \"your_api_key_here\"\n")
        sys.exit(1)
    return genai.Client(api_key=key)

def process_page(client: genai.Client, doc: fitz.Document, page_num: int, model_name: str = "gemini-flash-lite-latest") -> str:
    page = doc.load_page(page_num)
    # Render page at 200 DPI for crisp text recognition
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes(output="png")

    models_to_try = [model_name, "gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-flash-latest"]
    # Remove duplicates preserving order
    models_to_try = list(dict.fromkeys(models_to_try))

    last_err = None
    for m in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        OCR_PROMPT,
                    ],
                )
                if response.text:
                    return response.text.strip()
            except Exception as e:
                last_err = e
                # If model not found or unavailable, wait briefly or try next
                time.sleep(2.0 * (attempt + 1))
    
    raise last_err or RuntimeError("Failed to transcribe page")

def main():
    parser = argparse.ArgumentParser(description="OCR Vintage Hindi PDF using Gemini Flash")
    parser.add_argument("--pdf", type=str, default=str(DEFAULT_PDF), help="Path to the PDF file")
    parser.add_argument("--key", type=str, default=None, help="Gemini API Key")
    parser.add_argument("--start", type=int, default=1, help="Start page (1-indexed)")
    parser.add_argument("--end", type=int, default=None, help="End page (1-indexed, default: last page)")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--model", type=str, default="gemini-flash-lite-latest", help="Gemini model name")
    parser.add_argument("--delay", type=float, default=4.2, help="Delay in seconds between requests for Free Tier rate limit")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] PDF file not found at: {pdf_path}")
        sys.exit(1)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_file = out_dir / "full_book_transcription.md"

    client = get_gemini_client(args.key)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    start_page = max(1, args.start)
    end_page = min(total_pages, args.end) if args.end else total_pages

    print("=" * 60)
    print("📖 Vintage Hindi OCR with Gemini Flash")
    print(f"📄 PDF: {pdf_path.name} (Total: {total_pages} pages)")
    print(f"📑 Processing pages: {start_page} to {end_page}")
    print(f"💾 Output folder: {out_dir}")
    print("=" * 60)

    for p in range(start_page - 1, end_page):
        current_page_num = p + 1
        page_file = out_dir / f"page_{current_page_num:03d}.txt"

        # Resume support: skip already processed pages
        if page_file.exists() and page_file.stat().st_size > 0:
            print(f"⏩ Page {current_page_num}/{end_page} already processed. Skipping.")
            continue

        print(f"🔍 Transcribing page {current_page_num}/{end_page}...", end="", flush=True)

        try:
            transcription = process_page(client, doc, p, model_name=args.model)
            page_file.write_text(transcription, encoding="utf-8")

            # Append to combined book file
            with open(combined_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- [Page {current_page_num}] ---\n\n")
                f.write(transcription)

            print(" ✅ Done!")

            # Rate limit respect for free tier (15 RPM -> ~4s delay)
            if current_page_num < end_page:
                time.sleep(args.delay)

        except Exception as e:
            print(f" ❌ Failed: {e}")
            print("Waiting 10 seconds before continuing...")
            time.sleep(10)

    print("\n🎉 OCR processing complete!")
    print(f"📁 Combined file saved to: {combined_file}")

if __name__ == "__main__":
    main()
