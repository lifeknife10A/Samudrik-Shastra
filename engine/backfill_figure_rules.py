#!/usr/bin/env python3
"""
Back-fills rules for figures whose citing paragraph produced no rule.

Seven figures had a passage but no linked rule, so choosing one of them gave
an empty reading. This script sends those passages to the same extraction
prompt the book used, then appends the new rules to the END of
knowledge_base.json. Appending keeps every existing rule_<page>_<n> id valid,
because the ids are built from each page's rules in knowledge-base order.

Usage:
    python3 engine/backfill_figure_rules.py            # dry run, writes nothing
    python3 engine/backfill_figure_rules.py --write    # append the rules

After a --write run, re-run:
    python3 engine/build_figure_index.py
    python3 engine/build_vectordb.py
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract_knowledge import (BOOK_PATH, OUTPUT_PATH, extract_printed_marker,
                               extract_rules_with_llm, rule_key,
                               split_into_pages)
from build_vectordb import canonical_feature, is_degenerate_rule

FIGURE_INDEX_PATH = Path(__file__).parent / "figure_index.json"

# The untouched 2,279-rule knowledge base. No run may ever write to it.
ORIGINAL_BACKUP_PATH = Path(__file__).parent / "knowledge_base.original_2279.json"


def backup_path(now=None):
    """Per-run backup name, timestamped so no run overwrites another."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return Path(__file__).parent / ("knowledge_base.backup.%s.json" % stamp)

# A rule needs a describable feature. The extraction sometimes returns a
# one-word observation ("line", "mark"), which names nothing the user can
# look at, so those are not added.
MIN_OBSERVATION_WORDS = 3


def find_pdf_page(passage, pages):
    """PDF page whose text contains the start of this passage."""
    needle = " ".join(passage.split())[:60].lower()
    if not needle:
        return None
    for page in pages:
        text = " ".join(page["text"].split()).lower()
        if needle in text:
            return page["page"]
    return None


def figures_without_rules(figure_index):
    """Returns (plate, figure, entry) for every figure with no rule."""
    empty = []
    for plate in sorted(figure_index):
        for figure in sorted(figure_index[plate], key=int):
            entry = figure_index[plate][figure]
            if not entry["rule_ids"]:
                empty.append((plate, figure, entry))
    return empty


def main():
    parser = argparse.ArgumentParser(
        description="Back-fill rules for figures that have none")
    parser.add_argument("--write", action="store_true",
                        help="Append the rules to knowledge_base.json")
    args = parser.parse_args()

    figure_index = json.loads(FIGURE_INDEX_PATH.read_text(encoding="utf-8"))
    knowledge_base = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    pages = split_into_pages(BOOK_PATH.read_text(encoding="utf-8"))

    targets = figures_without_rules(figure_index)
    print("Figures with no linked rules: %d" % len(targets))
    for plate, figure, entry in targets:
        print("  %s figure %s (printed %s)" % (plate, figure, entry["printed_pages"]))
    print()

    existing_keys = {rule_key(rule) for rule in knowledge_base["rules"]}
    new_rules = []

    for plate, figure, entry in targets:
        print("--- %s figure %s ---" % (plate, figure))
        for passage in entry["passages"]:
            pdf_page = find_pdf_page(passage, pages)
            if pdf_page is None:
                print("    [WARN] passage not found in the book, skipped")
                continue
            printed_page = extract_printed_marker(
                next(p["text"] for p in pages if p["page"] == pdf_page))
            if printed_page is None:
                printed_page = entry["printed_pages"][0]
            print("    page %d (printed %s)" % (pdf_page, printed_page))
            try:
                rules = extract_rules_with_llm(passage)
            except Exception as error:
                print("    [WARN] extraction failed: %s" % error)
                continue
            kept = 0
            degenerate = 0
            for rule in rules:
                key = rule_key(rule)
                if not key or key in existing_keys:
                    continue
                existing_keys.add(key)
                # Same filter the vector database applies: a rule whose
                # observation just repeats its prediction carries nothing.
                if is_degenerate_rule(rule):
                    degenerate += 1
                    continue
                # A one-word observation describes nothing visible.
                if len(str(rule.get("observation") or "").split()) < MIN_OBSERVATION_WORDS:
                    degenerate += 1
                    continue
                # The paragraph belongs to this figure, so a generic feature
                # name ("line") is normalised to the figure's own plate.
                # Otherwise the rule cannot link to the figure it came from.
                if canonical_feature(rule.get("feature_name", "")) == "line":
                    print("      * feature normalised: %s -> %s"
                          % (rule.get("feature_name"), plate))
                    rule["feature_name"] = plate
                rule["source_page"] = printed_page
                rule["pdf_page"] = pdf_page
                rule["source_topic"] = "figure_backfill"
                rule["source"] = "figure_backfill"
                new_rules.append(rule)
                kept += 1
                print("      + %s | %s -> %s" % (
                    canonical_feature(rule.get("feature_name", "")),
                    str(rule.get("observation"))[:60],
                    str(rule.get("prediction"))[:60]))
            print("    %d rule(s) extracted, %d new, %d degenerate skipped"
                  % (len(rules), kept, degenerate))

    print()
    print("New rules to append: %d" % len(new_rules))
    if not new_rules:
        print("Nothing to do.")
        return
    if not args.write:
        print("Dry run: knowledge_base.json was not changed.")
        print("Re-run with --write to append them.")
        return

    destination = backup_path()
    if destination.name == ORIGINAL_BACKUP_PATH.name:
        print("[ERROR] refusing to overwrite the original backup")
        return
    shutil.copy(OUTPUT_PATH, destination)
    print("Backup written to %s" % destination.name)

    knowledge_base["rules"].extend(new_rules)
    for rule in new_rules:
        topic = rule.get("source_topic", "general")
        knowledge_base["topics"].setdefault(topic, []).append(rule)
    knowledge_base["metadata"]["total_rules"] = len(knowledge_base["rules"])
    knowledge_base["metadata"]["figure_backfill"] = len(new_rules)

    temporary = OUTPUT_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(knowledge_base, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    temporary.replace(OUTPUT_PATH)
    print("Appended %d rules. Total rules now: %d"
          % (len(new_rules), knowledge_base["metadata"]["total_rules"]))
    print("Now re-run build_figure_index.py and build_vectordb.py")


if __name__ == "__main__":
    main()
