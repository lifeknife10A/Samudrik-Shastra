#!/usr/bin/env python3
"""
Phase C review helper: prepares random samples for manual verification.

Writes engine/review_samples.md containing:
1. 30 randomly chosen rules (with pdf + printed page) to check against the book
2. 20 randomly chosen page mappings to check against the PDF

Usage:
    python3 engine/prepare_review_samples.py
"""

import json
import random
import sys
from pathlib import Path

KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.json"
FIGURE_INDEX_PATH = Path(__file__).parent / "figure_index.json"
OUTPUT_PATH = Path(__file__).parent / "review_samples.md"


def main():
    if not KNOWLEDGE_BASE_PATH.exists():
        print(f"[ERROR] {KNOWLEDGE_BASE_PATH} not found. Run extract_knowledge.py first.")
        sys.exit(1)

    knowledge_base = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    rules = knowledge_base.get("rules", [])
    page_map = knowledge_base.get("page_map", {})

    random.seed(42)  # reproducible samples
    rule_sample = random.sample(rules, min(30, len(rules)))
    pages = list(page_map.items())
    page_sample = random.sample(pages, min(20, len(pages)))

    lines = [
        "# Manual Review Samples",
        "",
        "Check each rule against the book text and each page mapping against the PDF.",
        "",
        "## Rules to verify (30)",
        "",
        "| # | pdf_page | printed_page | feature | observation | prediction |",
        "|---|----------|--------------|---------|-------------|------------|",
    ]
    for i, rule in enumerate(rule_sample, 1):
        observation = str(rule.get("observation", ""))[:60].replace("|", "/")
        prediction = str(rule.get("prediction", ""))[:60].replace("|", "/")
        feature = str(rule.get("feature_name", ""))[:20].replace("|", "/")
        lines.append(
            f"| {i} | {rule.get('pdf_page', '?')} | {rule.get('source_page', '?')} "
            f"| {feature} | {observation} | {prediction} |"
        )

    lines += [
        "",
        "## Page mappings to verify (20)",
        "",
        "| # | pdf_page | printed_page | source |",
        "|---|----------|--------------|--------|",
    ]
    for i, (pdf_num, entry) in enumerate(page_sample, 1):
        lines.append(f"| {i} | {pdf_num} | {entry.get('printed')} | {entry.get('source')} |")

    lines.append("")
    lines.append("*Verify printed numbers by opening the PDF and looking at the page "
                 "corresponding to each pdf_page.*")

    # --- Figure index hand-check: one passage next to each crop ---
    if FIGURE_INDEX_PATH.exists():
        figure_index = json.loads(FIGURE_INDEX_PATH.read_text(encoding="utf-8"))
        lines += [
            "",
            "---",
            "",
            "# Figure Index Hand-Check",
            "",
            "For each figure below, open the crop and confirm the passage really",
            "describes the line shown in that crop. Flag any mismatch.",
            "",
        ]
        total = 0
        for plate in sorted(figure_index):
            lines.append(f"## {plate}")
            lines.append("")
            for fig_key in sorted(figure_index[plate], key=int):
                entry = figure_index[plate][fig_key]
                total += 1
                passage = entry["passages"][0] if entry["passages"] else "(no passage)"
                passage = " ".join(passage.split())[:260]
                sub = ", ".join(entry.get("sub_labels", [])) or "-"
                lines.append(f"**Figure {fig_key}** — `{entry['crop']}`"
                             f" (printed {entry['printed_pages']}, "
                             f"rules {len(entry['rule_ids'])}, sub-labels: {sub})")
                lines.append("")
                lines.append(f"> {passage}")
                lines.append("")
        lines.append(f"*{total} figures to check.*")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Review samples written to {OUTPUT_PATH}")
    print(f"   {len(rule_sample)} rules + {len(page_sample)} page mappings + figures")


if __name__ == "__main__":
    main()
