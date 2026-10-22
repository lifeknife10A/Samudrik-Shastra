#!/usr/bin/env python3
"""
Build figure_index.json: maps each figure plate to its figures,
passages, and knowledge-base rules.

Citations of the form "(Figure No. 3)", "[Figure No. 2]", "(Line number 1
in Figure No. 1)", "Observe ... in Figure No. 3." and similar all match.

Usage:
    python3 engine/build_figure_index.py

Output: engine/figure_index.json
Structure:
{
  "heart_line": {
    "1": {
      "crop": "Heart_Line_Hand_1.png",
      "crop_exists": true,
      "printed_pages": [141],
      "passages": ["...", "..."],
      "rule_ids": ["rule_41_3"],
      "rules": [{"id": "rule_41_3", "obs": "first 60 characters of the observation"}],
      "sub_labels": ["line 1-1"]
    },
    ...
  },
  ...
}
"""

import json
import re
from pathlib import Path

from build_vectordb import canonical_feature

BOOK_PATH = Path(__file__).parent.parent / "Samudrik_Shastra_English.md"
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.json"
OUTPUT_PATH = Path(__file__).parent / "figure_index.json"
HANDS_DIR = Path(__file__).parent.parent / "illustrations" / "cropped_hands"

PLATE_PREFIXES = {
    "heart_line": "Heart_Line",
    "life_line": "Life_Line",
    "head_line": "Head_Line",
    "fate_line": "Fate_Line",
    "sun_line": "Sun_Line",
    "health_line": "Health_Line",
    "marriage_line": "Marriage_Line",
}

# Line names used to assign a plate to one citation (nearest keyword wins).
PLATE_KEYWORDS = {
    "heart_line": ["heart line", "हृदय रेखा"],
    "life_line": ["life line", "जीवन रेखा"],
    "head_line": ["head line", "mental line", "मस्तक रेखा"],
    "fate_line": ["fate line", "saturn line", "भाग्य रेखा"],
    "sun_line": ["sun line", "apollo line", "सूर्य रेखा"],
    "health_line": ["health line", "स्वास्थ्य रेखा"],
    "marriage_line": ["marriage line", "विवाह रेखा"],
}

# Fallback: printed-page ranges per chapter (citation paragraph names no line).
PRINTED_RANGES = {
    "life_line": (121, 138),
    "health_line": (135, 138),
    "heart_line": (141, 150),
    "head_line": (152, 172),
    "fate_line": (176, 187),
    "sun_line": (198, 202),
    "marriage_line": (205, 210),
}

# Chapter ranges used to keep a figure's passages inside its own chapter.
# A citation is kept only when its printed page falls in the range of the
# chapter it was assigned to, so a figure cannot collect a passage from a
# different chapter. Ranges may overlap (life and health share 134-139).
CHAPTER_PRINTED_RANGES = {
    "life_line": (116, 139),
    "health_line": (134, 139),
    "heart_line": (140, 150),
    "head_line": (151, 172),
    "fate_line": (173, 187),
    "sun_line": (188, 204),
    "marriage_line": (205, 212),
}

# The scan misreads the middle digit of some three-digit printed numbers on
# PDF pages 92-107 (a printed 9 came through as 4, 6 or 1). The corrected
# values below come from the chapter sequence and from these known cases:
#   160->190, 161->191, 144->194, 146->196, 601->201.
# PDF 92 is deliberately NOT corrected: its Hindi marker is (११२) and the
# text continues the fate chapter, so the marker is left as read.
PRINTED_CORRECTIONS = {
    94: 190,    # marker read 160
    95: 191,    # marker read 161
    96: 192,    # marker read 162
    98: 194,    # marker read 144
    99: 195,    # marker read 165
    100: 196,   # marker read 166
    101: 197,   # marker read 167
    103: 196,   # marker read 146
    105: 201,   # marker read 601
}

# PDF pages 92-107 hold interleaved fate and sun pages, and their printed
# markers are misread, so a page number cannot decide the chapter here.
INTERLEAVED_PDF_PAGES = set(range(92, 108))
INTERLEAVED_PLATES = ("fate_line", "sun_line")

# Target figures per plate. Known genuine gaps: fate 6, marriage 7.
EXPECTED = {
    "heart_line": set(range(1, 7)),
    "life_line": set(range(1, 7)),
    "head_line": set(range(1, 13)),
    "health_line": set(range(1, 6)),
    "fate_line": {1, 2, 3, 4, 5, 7, 8},
    "sun_line": set(range(1, 7)),
    "marriage_line": set(range(1, 7)),
}

# Matches "(Figure No. 1)", "Figure no. 2", "[Figure No. 2]", "in Figure
# No. 1", "Observe ... in Figure No. 3." — requires no bracket.
FIGURE_RE = re.compile(
    r"(?:figure|fig\.?)\s*(?:no\.?\s*)?(\d{1,2})", re.IGNORECASE
)

# Sub-line labels such as "dotted line marked 4-4" or "lines marked 1-1".
SUB_LABEL_RE = re.compile(r"\b(\d{1,2}-\d{1,2})\b")


def split_into_pages(content):
    """Split the book into pages using ## Page N headers."""
    pattern = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(content))
    pages = []
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        page_text = content[start:end].strip()
        pages.append({"page": page_num, "text": page_text})
    return pages


def extract_printed_marker(text):
    """Extract printed page number from page body."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return None
    candidates = lines[:3] + lines[-3:]
    # Tolerate the scan's stray punctuation: "(( 0152. ))", "[ 164 ]".
    for line in candidates:
        match = re.match(r"^[\(\[\]\s]*(\d{1,4})\.?\s*[\)\]\s]*$", line)
        if match:
            return int(match.group(1))
    for line in candidates:
        match = re.match(r"^.+?\s+(\d{2,3})$", line)
        if match:
            value = int(match.group(1))
            if value >= 5:
                return value
    for line in candidates:
        if re.match(r"^\d{1,3}$", line):
            value = int(line)
            if value >= 50:
                return value
    return None


# Master figures and extended plates live on these PDF pages; they are out
# of v1 by decision (not cited by figure number in the line chapters).
MASTER_FIGURE_PAGES = set(range(209, 224))


def plate_from_text(text):
    """Plate with the most keyword mentions in the passage.

    Paragraphs routinely name neighbouring lines ("the marriage line
    inclines toward the heart line"), so counting mentions beats picking
    the nearest one; the subject line is repeated more often.
    """
    lower = text.lower()
    scores = {}
    first_pos = {}
    for plate, keywords in PLATE_KEYWORDS.items():
        # Max over the plate's synonyms, not the sum: "Sun Line (Apollo
        # Line)" must not count as two votes against the marriage line.
        score = 0
        earliest = None
        for keyword in keywords:
            count = lower.count(keyword)
            if count:
                score = max(score, count)
                pos = lower.find(keyword)
                if earliest is None or pos < earliest:
                    earliest = pos
        if score:
            scores[plate] = score
            first_pos[plate] = earliest
    if not scores:
        return None
    # Ties are common ("Health Line runs parallel to the Life Line"); the
    # subject line is normally named first, so the earliest keyword wins.
    best_score = max(scores.values())
    tied = [p for p in scores if scores[p] == best_score]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda p: first_pos[p])


def plate_from_rules(page_rules, printed_page):
    """Most common canonical plate among the page's rules."""
    votes = {}
    for rule in page_rules:
        if not isinstance(rule, dict):
            continue
        canon = canonical_feature(rule.get("feature_name", ""))
        if canon in PLATE_PREFIXES:
            votes[canon] = votes.get(canon, 0) + 1
    if not votes:
        return None
    best_score = max(votes.values())
    tied = [p for p in votes if votes[p] == best_score]
    if len(tied) == 1:
        return tied[0]
    # Ties: prefer the plate whose chapter contains this printed page.
    in_range = [p for p in tied
                if PRINTED_RANGES[p][0] <= printed_page <= PRINTED_RANGES[p][1]]
    if len(in_range) == 1:
        return in_range[0]
    return tied[0]


def plate_from_printed_range(printed_page):
    for plate, (lo, hi) in PRINTED_RANGES.items():
        if lo <= printed_page <= hi:
            return plate
    return None


def sub_labels_for_window(window):
    """Labels such as 'dotted line 4-4' or 'line 1-1' near a citation."""
    labels = []
    for m in SUB_LABEL_RE.finditer(window):
        label = m.group(1)
        lookback = window[:m.start()].lower()
        if "dotted" in lookback[-20:]:
            text = "dotted line " + label
        elif "line" in lookback[-20:]:
            text = "line " + label
        else:
            text = label
        if text not in labels:
            labels.append(text)
    return labels


def text_overlap(rule_text, passage_text):
    """Share of the rule's content words that appear in the passage."""
    rule_words = set(re.findall(r"[a-z]{3,}", rule_text.lower()))
    if not rule_words:
        return 0.0
    passage_words = set(re.findall(r"[a-z]{3,}", passage_text.lower()))
    return len(rule_words & passage_words) / len(rule_words)


RULE_LINK_THRESHOLD = 0.5


def link_rules_by_passage(figure_index, rules_by_page):
    """Links a rule only to the figure whose citing paragraph it comes from.

    Page-wide linking gave every figure on a page the same rule set, so
    picking figure 11 also recited figure 5's and 6's variations. Here a
    rule is attached only when at least half of its content words appear in
    that figure's passage window (the citing paragraph plus the one before).
    """
    linked = 0
    for plate, figures in figure_index.items():
        for fig_key, entry in figures.items():
            # Each passage is tested on its own. Joining them would let a rule
            # pass on words spread across different paragraphs.
            passages = entry["passages"]
            for pdf_page in entry.get("pdf_pages", []):
                for rule_idx, rule in enumerate(rules_by_page.get(pdf_page, [])):
                    if not isinstance(rule, dict):
                        continue
                    if canonical_feature(rule.get("feature_name", "")) != plate:
                        continue
                    rule_text = " ".join(str(rule.get(k) or "") for k in
                                         ("observation", "prediction", "variation"))
                    best = 0.0
                    for passage in passages:
                        score = text_overlap(rule_text, passage)
                        if score > best:
                            best = score
                    if best < RULE_LINK_THRESHOLD:
                        continue
                    rule_id = "rule_%d_%d" % (pdf_page, rule_idx)
                    if rule_id not in entry["rule_ids"]:
                        entry["rule_ids"].append(rule_id)
                        # The observation prefix is saved so the reader can
                        # tell whether the knowledge base has changed since
                        # this index was built.
                        observation = str(rule.get("observation") or "")
                        entry["rules"].append({"id": rule_id,
                                               "obs": observation[:60]})
                        linked += 1
    return linked


def corrected_printed_page(pdf_page, printed_page):
    """Printed number with the misread pages 92-107 corrected."""
    if pdf_page in PRINTED_CORRECTIONS:
        return PRINTED_CORRECTIONS[pdf_page]
    return printed_page


def build_chapter_pdf_pages(pages):
    """PDF pages belonging to each chapter, from corrected printed numbers."""
    chapter_pages = {plate: set() for plate in CHAPTER_PRINTED_RANGES}
    for page in pages:
        # Correct first: some of the misread pages (92-107) have a marker the
        # extractor cannot read at all, and the correction rescues them.
        printed_page = corrected_printed_page(page["page"],
                                              extract_printed_marker(page["text"]))
        if printed_page is None:
            continue
        for plate, (low, high) in CHAPTER_PRINTED_RANGES.items():
            if low <= printed_page <= high:
                chapter_pages[plate].add(page["page"])
    return chapter_pages


def record_drop(dropped, plate, fig_num, printed_page, pdf_page, passage, reason):
    """Records and prints a citation that is not used."""
    dropped.append((plate, fig_num, printed_page, passage[:80]))
    print("  DROP  %-14s fig %-2d printed %-4s (PDF %d) [%s]: %s"
          % (plate, fig_num, printed_page, pdf_page, reason, passage[:80]))


def print_corrections(pages):
    """Prints the corrected printed numbers so they can be checked by hand."""
    raw_markers = {page["page"]: extract_printed_marker(page["text"])
                   for page in pages}
    print("\nPrinted-page corrections (PDF pages 92-107):")
    for pdf_page in sorted(PRINTED_CORRECTIONS):
        print("  PDF %3d: marker read %-4s -> printed %d"
              % (pdf_page, raw_markers.get(pdf_page), PRINTED_CORRECTIONS[pdf_page]))


def main():
    if not BOOK_PATH.exists():
        print("[ERROR] Book not found at %s" % BOOK_PATH)
        return
    if not KNOWLEDGE_BASE_PATH.exists():
        print("[ERROR] Knowledge base not found at %s" % KNOWLEDGE_BASE_PATH)
        print("Run extract_knowledge.py first.")
        return

    content = BOOK_PATH.read_text(encoding="utf-8")
    pages = split_into_pages(content)
    print("Split book into %d pages" % len(pages))

    kb = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    rules_by_page = {}
    for rule in kb.get("rules", []):
        if not isinstance(rule, dict):
            continue
        pdf_page = rule.get("pdf_page")
        if pdf_page is not None:
            rules_by_page.setdefault(pdf_page, []).append(rule)

    figure_index = {}
    total_refs = 0
    dropped = []

    chapter_pdf_pages = build_chapter_pdf_pages(pages)
    print_corrections(pages)
    print("\nPDF pages per chapter:")
    for plate in sorted(chapter_pdf_pages):
        pages_in_chapter = sorted(chapter_pdf_pages[plate])
        print("  %-14s %d pages: %s"
              % (plate, len(pages_in_chapter), pages_in_chapter))

    for page in pages:
        pdf_page = page["page"]
        printed_page = extract_printed_marker(page["text"])
        if printed_page is None:
            continue

        # Remove embedded image blocks, then split into paragraphs.
        text_clean = re.sub(r"^>.*$", "", page["text"], flags=re.MULTILINE)
        text_clean = re.sub(r"!\[.*?\]\(.*?\)", "", text_clean)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text_clean) if p.strip()]
        page_rules = rules_by_page.get(pdf_page, [])

        if pdf_page in MASTER_FIGURE_PAGES:
            continue

        for para_idx, para in enumerate(paragraphs):
            for cite in FIGURE_RE.finditer(para):
                fig_num = int(cite.group(1))
                if fig_num < 1 or fig_num > 12:
                    continue
                total_refs += 1

                # Passage: the citing paragraph itself. The paragraph before
                # is added ONLY when the citation sits in a short fragment,
                # because including it otherwise pulls in the neighbouring
                # figure's description (that made two figures share a rule set
                # and let one figure recite another's variations).
                prev_para = paragraphs[para_idx - 1] if para_idx > 0 else ""
                passage = para
                if len(para.split()) < 40 and prev_para:
                    passage = prev_para + " " + para
                passage = re.sub(r"\s+", " ", passage).strip()

                if pdf_page in INTERLEAVED_PDF_PAGES:
                    # Interleaved fate/sun pages with misread markers: the
                    # page number says nothing, so the named line decides.
                    # The citing paragraph first, then the paragraph before
                    # it (the description often starts there), then the
                    # page's own rules as a last content-based signal.
                    plate = plate_from_text(para)
                    if plate is None:
                        plate = plate_from_text(prev_para + " " + para)
                    if plate is None:
                        plate = plate_from_rules(page_rules, printed_page)
                    if plate not in INTERLEAVED_PLATES:
                        record_drop(dropped, plate or "unknown", fig_num,
                                    printed_page, pdf_page, passage,
                                    "no fate/sun line named")
                        continue
                else:
                    # 1. Plate named most often in the citing paragraph
                    #    (previous paragraph only when this one is silent).
                    plate = plate_from_text(para)
                    if plate is None and prev_para:
                        plate = plate_from_text(prev_para + " " + para)
                    # 2. Page-level rule vote (ties broken by range).
                    if plate is None:
                        plate = plate_from_rules(page_rules, printed_page)
                    # 3. Any printed range containing the page.
                    if plate is None:
                        plate = plate_from_printed_range(printed_page)
                    # 4. Three-signal arbitration. Keywords and printed
                    #    markers are each occasionally wrong (the scan is
                    #    jumbled), but the page's rule vote corroborates
                    #    whichever is right: the exclusive range wins only
                    #    when the rules also vote for it.
                    range_hits = [p for p, (lo, hi) in PRINTED_RANGES.items()
                                  if lo <= printed_page <= hi]
                    if (len(range_hits) == 1 and range_hits[0] != plate):
                        rules_plate = plate_from_rules(page_rules, printed_page)
                        if rules_plate == range_hits[0]:
                            plate = range_hits[0]
                    if plate is None:
                        print("  Page %d (printed %d): figure %d, no plate could be assigned"
                              % (pdf_page, printed_page, fig_num))
                        continue

                    # Keep the passage inside the chapter it belongs to: a page
                    # outside the plate's own chapter cannot describe its figure.
                    if pdf_page not in chapter_pdf_pages.get(plate, set()):
                        record_drop(dropped, plate, fig_num,
                                    corrected_printed_page(pdf_page, printed_page),
                                    pdf_page, passage, "outside its chapter")
                        continue

                print("  Page %d (printed %d): %s fig %d"
                      % (pdf_page, printed_page, plate, fig_num))

                fig_key = str(fig_num)
                if plate not in figure_index:
                    figure_index[plate] = {}
                if fig_key not in figure_index[plate]:
                    crop_name = "%s_Hand_%d.png" % (PLATE_PREFIXES[plate], fig_num)
                    figure_index[plate][fig_key] = {
                        "crop": crop_name,
                        "crop_exists": (HANDS_DIR / crop_name).exists(),
                        "printed_pages": [],
                        "pdf_pages": [],
                        "passages": [],
                        "rule_ids": [],
                        "rules": [],
                        "sub_labels": [],
                    }
                entry = figure_index[plate][fig_key]

                if printed_page not in entry["printed_pages"]:
                    entry["printed_pages"].append(printed_page)
                if pdf_page not in entry["pdf_pages"]:
                    entry["pdf_pages"].append(pdf_page)
                if passage not in entry["passages"]:
                    entry["passages"].append(passage)
                for label in sub_labels_for_window(para[:cite.start() + 40]):
                    if label not in entry["sub_labels"]:
                        entry["sub_labels"].append(label)

    linked = link_rules_by_passage(figure_index, rules_by_page)
    print("\nTotal figure references matched: %d" % total_refs)
    print("Rules linked by passage window: %d" % linked)

    print("\nDropped citations (page outside the plate's chapter): %d" % len(dropped))
    for plate, fig_num, printed_page, text in dropped:
        print("  %-14s fig %-2d printed %-4s | %s"
              % (plate, fig_num, printed_page, text))
    print("\nCoverage check (target = everything except fate 6, marriage 7):")
    all_ok = True
    total_figures = 0
    for plate in sorted(EXPECTED):
        figures = figure_index.get(plate, {})
        found = {int(k) for k in figures}
        expected = EXPECTED[plate]
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        no_passage = sorted(int(k) for k, v in figures.items() if not v["passages"])
        no_rules = sorted(int(k) for k, v in figures.items() if not v["rule_ids"])
        print("  %-14s found %2d/%2d  missing=%s  extra=%s"
              % (plate, len(found & expected), len(expected), missing, extra))
        total_figures += len(figures)
        if no_passage:
            print("      no passage: %s" % no_passage)
            all_ok = False
        if no_rules:
            print("      no linked rules: %s" % no_rules)
        for k, v in figures.items():
            if not v["crop_exists"]:
                print("      crop missing on disk: %s (%s)" % (k, v["crop"]))

    if all_ok:
        print("\nAll expected figures have at least one passage.")

    with_rules = sum(1 for p in figure_index for k in figure_index[p]
                     if figure_index[p][k]["rule_ids"])
    print("Total figures in the index: %d (with rules: %d)"
          % (total_figures, with_rules))

    OUTPUT_PATH.write_text(
        json.dumps(figure_index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSaved to %s" % OUTPUT_PATH)


if __name__ == "__main__":
    main()
