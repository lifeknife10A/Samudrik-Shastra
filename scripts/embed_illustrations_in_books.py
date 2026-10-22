#!/usr/bin/env python3
"""
Embeds all extracted palm and foot diagram images directly into:
1. Samudrik_Shastra_English.md (English translated edition)
2. full_book_transcription.md (Original Hindi OCR edition)
"""

from pathlib import Path
import re

DOWNLOADS = Path.home() / "Downloads"
ENG_MD_PATH = DOWNLOADS / "translation_output" / "Samudrik_Shastra_English.md"
HINDI_MD_PATH = DOWNLOADS / "ocr_output" / "full_book_transcription.md"
ILLUST_DIR = DOWNLOADS / "samudrik_shastra_illustrations"
PLATES_DIR = ILLUST_DIR / "plates"
CROPPED_DIR = ILLUST_DIR / "cropped_hands"

# Mapping of page number to plate filename and cropped hand prefix
PAGE_ILLUSTRATIONS = {
    82: {"plate": "plate_082_heart_line_page_082.png", "title": "Heart Line Variations (हृदय रेखा)", "cropped_prefix": "Heart_Line"},
    85: {"plate": "plate_085_fate_line_page_085.png", "title": "Fate Line Variations (भाग्य रेखा)", "cropped_prefix": "Fate_Line"},
    89: {"plate": "plate_089_head_line_page_089.png", "title": "Head Line Variations (मस्तक रेखा)", "cropped_prefix": "Head_Line"},
    93: {"plate": "plate_093_life_line_page_093.png", "title": "Life Line Variations (जीवन रेखा)", "cropped_prefix": "Life_Line"},
    97: {"plate": "plate_097_sun_line_page_097.png", "title": "Sun Line Variations (सूर्य रेखा)", "cropped_prefix": "Sun_Line"},
    108: {"plate": "plate_108_health_line_page_108.png", "title": "Health Line Variations (स्वास्थ्य रेखा)", "cropped_prefix": "Health_Line"},
    110: {"plate": "plate_110_marriage_line_page_110.png", "title": "Marriage Line Variations (विवाह रेखा)", "cropped_prefix": "Marriage_Line"},
    162: {"plate": "foot_fig_02_page_162.png", "title": "Foot Lines: Figure 2 (पैर की रेखायें चित्र २)"},
    168: {"plate": "foot_fig_03_page_168.png", "title": "Foot Lines: Figure 3 (पैर की रेखायें चित्र ३)"},
    171: {"plate": "foot_fig_04_page_171.png", "title": "Foot Lines: Figure 4 (पैर की रेखायें चित्र ४)"},
    174: {"plate": "foot_fig_05_page_174.png", "title": "Foot Lines: Figure 5 (पैर की रेखायें चित्र ५)"},
    180: {"plate": "foot_fig_06_page_180.png", "title": "Foot Lines: Figure 6 (पैर की रेखायें चित्र ६)"},
    184: {"plate": "foot_fig_07_page_184.png", "title": "Foot Lines: Figure 7 - Chain & Net Formations"},
    185: {"plate": "foot_fig_08_page_185.png", "title": "Foot Lines: Figure 8 - Straight Line Formations"},
    189: {"plate": "foot_fig_09_page_189.png", "title": "Foot Lines: Figure 9 - Island Formations"},
    193: {"plate": "foot_fig_10_page_193.png", "title": "Foot Lines: Figure 10 - Circles & Crosses"},
    209: {"plate": "hand_fig_01_page_209.png", "title": "Master Hand Figure 1: Major Lines & Mounts Overview"},
    210: {"plate": "hand_fig_02_page_210.png", "title": "Master Hand Figure 2: Trajectories across Mounts"},
    211: {"plate": "hand_fig_03_page_211.png", "title": "Master Hand Figure 3: Intersecting Line Junctions"},
    212: {"plate": "hand_fig_04_page_212.png", "title": "Master Hand Figure 4: Planetary Mount Emergence"},
    213: {"plate": "hand_fig_05_06_page_213.png", "title": "Master Hand Figures 5 & 6: Triangles, Circles, Squares"},
    214: {"plate": "hand_fig_07_08_page_214.png", "title": "Master Hand Figures 7 & 8: Junction of Life, Head & Heart Lines"},
    215: {"plate": "hand_fig_09_10_page_215.png", "title": "Master Hand Figures 9 & 10: Serpent-Tongue Forked Lines"},
    216: {"plate": "hand_fig_11_12_page_216.png", "title": "Master Hand Figures 11 & 12: Branching Lines"},
    218: {"plate": "hand_fig_15_16_page_218.png", "title": "Master Hand Figures 15 & 16: Marriage Line Interactions"},
    219: {"plate": "hand_fig_17_page_219.png", "title": "Master Hand Figure 17: Bracelet Lines (Manibandh / Rascettes)"},
    220: {"plate": "hand_plate_life_var_page_220.png", "title": "Life Line Additional Variations", "cropped_prefix": "Life_Line_Extended"},
    221: {"plate": "hand_plate_head_var_page_221.png", "title": "Head Line Additional Variations", "cropped_prefix": "Head_Line_Extended"},
    223: {"plate": "hand_plate_book_var_page_223.png", "title": "Book / Special Line Variations", "cropped_prefix": "Special_Lines"},
    280: {"plate": "hand_fig_sacred_signs_page_280.png", "title": "Sacred Palm Signs: Geometry & Symbols"},
}

def generate_embed_block(page_num: int, info: dict, relative_to_folder: Path) -> str:
    """Generates the Markdown image embedding block for a given page."""
    plate_file = PLATES_DIR / info["plate"]
    rel_plate = plate_file.resolve().as_posix() # Use clean posix path
    
    title = info["title"]
    block = f"\n> ### 🖼️ Diagram: {title}\n>\n"
    block += f"> ![{title}]({rel_plate})\n"

    # If cropped individual hands exist for this plate, add gallery
    if "cropped_prefix" in info:
        prefix = info["cropped_prefix"]
        cropped_hands = sorted(list(CROPPED_DIR.glob(f"{prefix}_Hand_*.png")))
        if cropped_hands:
            block += ">\n> **Individual Hand Variations (१ to ६):**\n>\n"
            for hand_p in cropped_hands:
                hand_num = hand_p.stem.split("_")[-1]
                block += f"> * Hand {hand_num}: `Hand {hand_num}`\n"
                block += f"> ![{hand_p.stem}]({hand_p.resolve().as_posix()})\n>\n"

    block += "\n"
    return block

def update_markdown_file(md_path: Path, page_pattern: str, is_english: bool):
    """Inserts illustration blocks under corresponding page headers."""
    content = md_path.read_text(encoding="utf-8")
    folder = md_path.parent

    count = 0
    for page_num, info in PAGE_ILLUSTRATIONS.items():
        if is_english:
            header_regex = re.compile(rf"(## Page {page_num}\s*\n+)")
        else:
            header_regex = re.compile(rf"(## पृष्ठ {page_num}\s*\n+)")

        if header_regex.search(content):
            embed_text = generate_embed_block(page_num, info, folder)
            # Avoid duplicate embedding if run multiple times
            if info["plate"] not in content:
                content = header_regex.sub(rf"\1{embed_text}", content, count=1)
                count += 1

    md_path.write_text(content, encoding="utf-8")
    print(f"✅ Updated {md_path.name}: Embedded diagrams for {count} pages.")

def main():
    print("=" * 60)
    print("🖼️ Embedding Illustrations into Samudrik Shastra Books")
    print("=" * 60)
    
    if ENG_MD_PATH.exists():
        update_markdown_file(ENG_MD_PATH, "## Page", is_english=True)
    else:
        print(f"❌ Not found: {ENG_MD_PATH}")

    if HINDI_MD_PATH.exists():
        update_markdown_file(HINDI_MD_PATH, "## पृष्ठ", is_english=False)
    else:
        print(f"❌ Not found: {HINDI_MD_PATH}")

    print("\n🎉 Both English and Hindi Markdown editions now feature embedded diagrams!")

if __name__ == "__main__":
    main()
