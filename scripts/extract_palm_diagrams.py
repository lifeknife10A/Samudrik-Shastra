#!/usr/bin/env python3
"""
Extract, crop, and catalog all Palm and Foot illustrations from Samudrik Shastra PDF.
"""

from pathlib import Path
import pymupdf as fitz
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
PDF_PATH = REPO_ROOT / "pdf" / "2015.400540.Samudirk-Shastr.pdf"
OUT_DIR = REPO_ROOT
PLATES_DIR = OUT_DIR / "illustrations" / "plates"
CROPPED_DIR = OUT_DIR / "illustrations" / "cropped_hands"

PLATES_DIR.mkdir(parents=True, exist_ok=True)
CROPPED_DIR.mkdir(parents=True, exist_ok=True)

# Comprehensive catalog of illustration pages in the book.
# Grid plates carry explicit cols/rows: the layouts were verified against the
# book's own figure numbering (Hindi OCR) and re-checked by inspection.
#   Heart(82)=2x3, Life(93)=2x3, Sun(97)=2x3, LifeExt(220)=2x3, HeadExt(221)=2x3
#   Health(108)=2x3 (5 figures, last slot empty)
#   Fate(85)=3x2 (6 figures)
#   Head(89)=3x4 (12 hands; figure groups 10,11,12)
#   Marriage(110)=3x3 (7 figures, slots 8 and 9 empty)
#   Special(223)=3x4 (12 hands, numbered 1-12)
DIAGRAM_CATALOG = [
    # Major 6-Hand Plates (Part 2)
    {"page": 82, "id": "plate_082_heart_line", "title": "Heart Line (हृदय रेखा) - 6 Hand Variations", "grid": True, "line_name": "Heart_Line", "cols": 2, "rows": 3},
    {"page": 85, "id": "plate_085_fate_line", "title": "Fate Line / Saturn Line (भाग्य रेखा) - 8 Hand Variations", "grid": True, "line_name": "Fate_Line",
     "boxes": [(0, 142, 295, 546), (295, 142, 590, 546), (590, 142, 886, 546),
               (0, 546, 295, 950), (295, 546, 590, 950), (590, 546, 886, 950),
               (0, 950, 443, 1354), (443, 950, 886, 1354)]},
    {"page": 89, "id": "plate_089_head_line", "title": "Head Line / Mental Line (मस्तक रेखा) - 12 Hand Variations", "grid": True, "line_name": "Head_Line", "cols": 3, "rows": 4},
    {"page": 93, "id": "plate_093_life_line", "title": "Life Line (जीवन रेखा) - 6 Hand Variations", "grid": True, "line_name": "Life_Line", "cols": 2, "rows": 3},
    {"page": 97, "id": "plate_097_sun_line", "title": "Sun Line / Apollo Line (सूर्य रेखा) - 6 Hand Variations", "grid": True, "line_name": "Sun_Line", "cols": 2, "rows": 3},
    {"page": 108, "id": "plate_108_health_line", "title": "Health / Mercury Line (स्वास्थ्य रेखा) - 5 Hand Variations", "grid": True, "line_name": "Health_Line", "cols": 2, "rows": 3},
    {"page": 110, "id": "plate_110_marriage_line", "title": "Marriage Line (विवाह रेखा) - 7 Hand Variations", "grid": True, "line_name": "Marriage_Line",
     "boxes": [(0, 77, 295, 385), (295, 77, 590, 385), (590, 77, 886, 385),
               (25, 516, 367, 857), (404, 516, 741, 857),
               (12, 948, 346, 1284), (374, 948, 720, 1284)]},
    
    # Foot Line Plates (Padam Rekha - Part 3)
    {"page": 162, "id": "foot_fig_02", "title": "Foot Lines: Figure 2 (पैर की रेखायें चित्र २)", "grid": False},
    {"page": 168, "id": "foot_fig_03", "title": "Foot Lines: Figure 3 (पैर की रेखायें चित्र ३)", "grid": False},
    {"page": 171, "id": "foot_fig_04", "title": "Foot Lines: Figure 4 (पैर की रेखायें चित्र ४)", "grid": False},
    {"page": 174, "id": "foot_fig_05", "title": "Foot Lines: Figure 5 (पैर की रेखायें चित्र ५)", "grid": False},
    {"page": 180, "id": "foot_fig_06", "title": "Foot Lines: Figure 6 (पैर की रेखायें चित्र ६)", "grid": False},
    {"page": 184, "id": "foot_fig_07", "title": "Foot Lines: Figure 7 - Chains & Islands", "grid": False},
    {"page": 185, "id": "foot_fig_08", "title": "Foot Lines: Figure 8 - Straight Lines", "grid": False},
    {"page": 189, "id": "foot_fig_09", "title": "Foot Lines: Figure 9 - Islands in Feet", "grid": False},
    {"page": 193, "id": "foot_fig_10", "title": "Foot Lines: Figure 10 - Circles & Crosses", "grid": False},

    # Master Hand Diagrams (Figures 1 through 18)
    {"page": 209, "id": "hand_fig_01", "title": "Hand Figure 1: Master Overview of Palm Lines (विभिन्न रेखाओं का सही स्थान)", "grid": False},
    {"page": 210, "id": "hand_fig_02", "title": "Hand Figure 2: Line Trajectories across Mounts", "grid": False},
    {"page": 211, "id": "hand_fig_03", "title": "Hand Figure 3: Intersecting Line Junctions", "grid": False},
    {"page": 212, "id": "hand_fig_04", "title": "Hand Figure 4: Planetary Mount Emergence", "grid": False},
    {"page": 213, "id": "hand_fig_05_06", "title": "Hand Figures 5 & 6: Triangles, Circles, Squares on Palm", "grid": False},
    {"page": 214, "id": "hand_fig_07_08", "title": "Hand Figures 7 & 8: Junction of Life, Head and Heart Lines", "grid": False},
    {"page": 215, "id": "hand_fig_09_10", "title": "Hand Figures 9 & 10: Serpent-Tongue Forked Lines", "grid": False},
    {"page": 216, "id": "hand_fig_11_12", "title": "Hand Figures 11 & 12: Branching and Conjoined Lines", "grid": False},
    {"page": 218, "id": "hand_fig_15_16", "title": "Hand Figures 15 & 16: Heart and Marriage Line Interactions", "grid": False},
    {"page": 219, "id": "hand_fig_17", "title": "Hand Figure 17: Bracelet Lines (Manibandh / Rascettes)", "grid": False},
    {"page": 220, "id": "hand_plate_life_var", "title": "Life Line Additional Variations (चित्र 317)", "grid": True, "line_name": "Life_Line_Extended", "cols": 2, "rows": 3},
    {"page": 221, "id": "hand_plate_head_var", "title": "Head Line Additional Variations (चित्र 318)", "grid": True, "line_name": "Head_Line_Extended", "cols": 2, "rows": 3},
    {"page": 223, "id": "hand_plate_book_var", "title": "Pustak Rekha / Special Lines (चित्र 320)", "grid": True, "line_name": "Special_Lines", "cols": 3, "rows": 4},
    {"page": 280, "id": "hand_fig_sacred_signs", "title": "Sacred Palm Signs: Triangles, Crescents, Cones, Stars (चित्र ६)", "grid": False},
]


def crop_grid_plate(plate_img_path: Path, base_name: str, cols: int, rows: int) -> list[Path]:
    """Crops a plate into cols x rows individual hand images.

    Numbering follows grid position (row-major), so Hand_N is always the Nth
    slot of the plate. Slots with almost no ink (empty grid positions) are
    skipped, but their numbers are reserved.
    """
    im = Image.open(plate_img_path)
    w, h = im.size

    # Exclude top header (~10% height) and bottom margin (~5% height)
    header_h = int(h * 0.10)
    footer_h = int(h * 0.05)
    content_h = h - header_h - footer_h

    col_w = w // cols
    row_h = content_h // rows

    cropped_files = []
    for r in range(rows):
        for c in range(cols):
            hand_idx = r * cols + c + 1  # grid position, 1-based
            left = c * col_w
            upper = header_h + r * row_h
            right = (c + 1) * col_w
            lower = upper + row_h

            cropped = im.crop((left, upper, right, lower))
            # Skip grid slots with (almost) no ink: some plates have fewer
            # figures than grid cells. Real hands measure ~3.3% ink or more;
            # empty slots (e.g. Health slot 6) fall below 2%.
            gray = np.array(cropped.convert("L"))
            ink_fraction = float((gray < 128).mean())
            if ink_fraction < 0.03:
                print(f"      (slot {hand_idx}: empty, skipped)")
                continue

            out_filename = CROPPED_DIR / f"{base_name}_Hand_{hand_idx}.png"
            cropped.save(out_filename)
            cropped_files.append(out_filename)

    return cropped_files

def crop_boxes_plate(plate_img_path: Path, base_name: str, boxes: list) -> list[Path]:
    """Crops a plate using explicit per-figure boxes (for plates whose hands
    are not in a uniform grid, e.g. Fate 3+3+2 and Marriage 3+2+2)."""
    im = Image.open(plate_img_path)
    cropped_files = []
    for index, box in enumerate(boxes, 1):
        out_filename = CROPPED_DIR / f"{base_name}_Hand_{index}.png"
        im.crop(box).save(out_filename)
        cropped_files.append(out_filename)
    return cropped_files


def main():
    doc = fitz.open(PDF_PATH)
    catalog_md = OUT_DIR / "Illustrations_Catalog.md"

    print("=" * 60)
    print("🖐️ Extracting Samudrik Shastra Palm & Foot Illustrations")
    print(f"📁 PDF: {PDF_PATH.name}")
    print(f"💾 Destination: {OUT_DIR}")
    print("=" * 60)

    # Full regeneration: remove stale crops from earlier runs so plates with
    # fewer figures this time (e.g. Health has 5) don't leave leftovers.
    stale = list(CROPPED_DIR.glob("*_Hand_*.png"))
    for old_file in stale:
        old_file.unlink()
    if stale:
        print(f"🗑️  Removed {len(stale)} stale crop files")

    entries = []
    for item in DIAGRAM_CATALOG:
        p_num = item["page"]
        title = item["title"]
        file_id = item["id"]

        # 1-indexed to 0-indexed
        page = doc.load_page(p_num - 1)
        pix = page.get_pixmap(dpi=250)
        plate_file = PLATES_DIR / f"{file_id}_page_{p_num:03d}.png"
        pix.save(plate_file)
        print(f"✅ Extracted Plate: Page {p_num:03d} -> {plate_file.name}")

        cropped_subfiles = []
        if item.get("boxes"):
            cropped_subfiles = crop_boxes_plate(plate_file, item.get("line_name", file_id),
                                                item["boxes"])
        elif item.get("grid"):
            cropped_subfiles = crop_grid_plate(plate_file, item.get("line_name", file_id),
                                               item["cols"], item["rows"])
        if cropped_subfiles:
            print(f"   ✂️ Sliced into {len(cropped_subfiles)} individual hand images in cropped_hands/")

        entries.append((p_num, title, plate_file, cropped_subfiles))

    # Generate Markdown Catalog (relative paths so the file is portable)
    with open(catalog_md, "w", encoding="utf-8") as f:
        f.write("# Samudrik Shastra - Illustrations & Diagram Catalog\n\n")
        f.write("This catalog maps all palm, mount, line, and foot diagrams extracted from the book.\n\n---\n\n")

        for p_num, title, plate_file, subfiles in entries:
            f.write(f"## Page {p_num}: {title}\n\n")
            f.write(f"![Plate]({plate_file.relative_to(OUT_DIR).as_posix()})\n\n")
            if subfiles:
                f.write("### Individual Hand Figures:\n\n")
                f.write("<div style='display: flex; flex-wrap: wrap; gap: 10px;'>\n")
                for subf in subfiles:
                    rel = subf.relative_to(OUT_DIR).as_posix()
                    f.write(f"  <img src='{rel}' width='200' alt='{subf.stem}' />\n")
                f.write("</div>\n\n")
            f.write("---\n\n")

    print("\n🎉 All illustrations successfully extracted and cataloged!")
    print(f"📖 Catalog generated at: {catalog_md}")

if __name__ == "__main__":
    main()
