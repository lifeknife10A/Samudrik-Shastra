#!/usr/bin/env python3
"""
One command that re-runs every check on the engine.

Usage:
    python3 engine/run_checks.py

Each check prints PASS or FAIL with a short reason, and the script exits 1
when any check failed. Nothing here writes a file and nothing here calls a
model: the readings are built straight from the rules, so the whole run takes
seconds and needs no photo.

The checks cover the parts that have gone wrong before: the sentence rewrites,
the figure picker against the reading, the rule ids against the knowledge
base, the cleaned rules file, the untouched original knowledge base, the
scripts' command lines, and the template reading itself.
"""

import contextlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import palm_reader
from clean_figure_rules import (best_passage_sentence, introduced_grave_word,
                                kept_share, source_for_rule)
from palm_reader import (CITATION_RE, PalmReader, parse_figures_argument,
                         test_sensitivity_rewrite)

ENGINE_DIR = Path(__file__).parent
FIGURE_INDEX_PATH = ENGINE_DIR / "figure_index.json"
KNOWLEDGE_BASE_PATH = ENGINE_DIR / "knowledge_base.json"
ORIGINAL_KB_PATH = ENGINE_DIR / "knowledge_base.original_2279.json"
CLEAN_RULES_PATH = ENGINE_DIR / "figure_rules_clean.json"

# The scripts a user is expected to be able to run.
SCRIPT_NAMES = ("palm_reader.py", "test_vision_hands.py",
                "clean_figure_rules.py", "backfill_figure_rules.py")

# The figures the book gives no printable rule text for. The picker refuses
# exactly these, and no others.
NO_PRINTABLE_FIGURES = ("heart_line 3", "head_line 2", "life_line 1",
                        "life_line 4", "marriage_line 2",
                        "marriage_line 5")

# A cleaned meaning may only say what its rule already said.
MIN_KEPT_SHARE = 0.50
ORIGINAL_RULE_COUNT = 2279

# The two readings every build must manage without a model call.
READING_CASES = (
    ("heart_line=1,fate_line=1", "right", "male"),
    ("life_line=5,heart_line=2", "left", "female"),
)

RESULTS = []


def report(name, ok, detail=""):
    """Prints one PASS/FAIL line and remembers the result."""
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    if not ok and detail:
        print("         %s" % detail)


def load_reader():
    """A reader with the index and the knowledge base, and no vector store.

    The vector store is only used for lines the user confirms absent, which no
    check here does, so loading it would only slow the checks down.
    """
    reader = PalmReader.__new__(PalmReader)
    reader.figure_index = json.loads(FIGURE_INDEX_PATH.read_text(encoding="utf-8"))
    reader.kb = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    reader._rule_by_id = reader._build_rule_lookup()
    reader.vector_collection = None
    return reader


def figure_keys(reader):
    """Every figure in the index, in a stable order."""
    keys = []
    for line in sorted(reader.figure_index):
        for figure in sorted(reader.figure_index[line], key=int):
            keys.append((line, figure))
    return keys


def build_reading(reader, line, figure):
    """The reading for one figure, built with no image and no model."""
    override = {line: {"status": "present", "figure": figure}}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = reader.read_palm(None, hand_type="right", gender="male",
                                  figures_override=override, use_vision=True)
    return result["reading"] if result else ""


def check_sensitivity():
    """a. The sentence rewrites still pass their own fixed cases."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ok = test_sensitivity_rewrite()
    output = buffer.getvalue()
    counted = re.search(r"all (\d+) cases correct", output)
    if ok and counted:
        return True, "%s cases correct" % counted.group(1)
    failed = [line.strip() for line in output.splitlines() if "[FAIL]" in line]
    return False, "; ".join(failed) if failed else "the test reported a failure"


def check_figure_counts(reader):
    """b. Each figure's picker count equals the rules its reading cites."""
    problems = []
    for line, figure in figure_keys(reader):
        counted = len(reader.figure_sentences(line, figure, "present"))
        reading = build_reading(reader, line, figure)
        cited = set(re.findall(r"\[R(\d+)\]", reading))
        if counted != len(cited):
            problems.append("%s figure %s: %d counted, %d cited"
                            % (line, figure, counted, len(cited)))
    if problems:
        return False, "; ".join(problems[:6])
    return True, "%d figures agree" % len(figure_keys(reader))


def check_no_printable_figures(reader):
    """c. Exactly the six known figures print nothing."""
    empty = []
    for line, figure in figure_keys(reader):
        if not reader.figure_sentences(line, figure, "present"):
            empty.append("%s %s" % (line, figure))
    if sorted(empty) == sorted(NO_PRINTABLE_FIGURES):
        return True, "%d figures, as expected" % len(empty)
    return False, "found: %s" % ", ".join(sorted(empty))


def check_rule_ids(reader):
    """d. Every rule id in the figure index still matches the knowledge base."""
    stale = sorted(reader._stale_rule_ids)
    if stale:
        return False, "%d stale id(s): %s" % (len(stale), ", ".join(stale[:6]))
    return True, "%d ids checked" % len(reader._rule_by_id)


def check_clean_rules(reader):
    """e. The cleaned rules file keeps its guarantees.

    The reader keeps one cleaned entry per rule id, so a rule shared by two
    figures is checked once, against the first figure it appears with (the
    figure the clean-up answered for), and its copies must agree.
    """
    clean = json.loads(CLEAN_RULES_PATH.read_text(encoding="utf-8"))
    problems = []
    statuses = {}
    first_answer = {}
    first_seen = {}
    checked = 0
    for line in sorted(clean):
        for figure in sorted(clean[line], key=int):
            entry_of_figure = reader.figure_index.get(line, {}).get(figure, {})
            passages = entry_of_figure.get("passages", [])
            for entry in clean[line][figure]:
                rule_id = entry["rule_id"]
                statuses.setdefault(rule_id, set()).add(entry["status"])
                answer = (entry["status"], entry.get("condition"),
                          entry.get("meaning"))
                if rule_id in first_answer:
                    if first_answer[rule_id] != answer:
                        problems.append("%s has copies that disagree" % rule_id)
                    continue  # one answer per rule id, checked at its first figure
                first_answer[rule_id] = answer
                first_seen[rule_id] = (line, figure)
                rule = reader._rule_by_id.get(rule_id)
                if rule is None:
                    problems.append("%s is not in the knowledge base" % rule_id)
                    continue
                if entry["status"] not in ("cleaned", "from_passage"):
                    continue
                checked += 1
                source, kind = source_for_rule(rule, passages)
                share = kept_share(entry.get("meaning", ""), source)
                if share < MIN_KEPT_SHARE:
                    problems.append("%s keeps %d%% of its %s"
                                    % (rule_id, round(share * 100), kind))
                rule_text = " ".join(str(rule.get(key) or "")
                                     for key in ("observation", "prediction",
                                                 "variation"))
                added = introduced_grave_word(entry.get("meaning", ""), rule_text)
                if added:
                    problems.append("%s adds '%s'" % (rule_id, added))
    for rule_id in sorted(statuses):
        if len(statuses[rule_id]) > 1:
            problems.append("%s has two statuses: %s"
                            % (rule_id, ", ".join(sorted(statuses[rule_id]))))
    if problems:
        return False, "; ".join(problems[:6])
    return True, ("%d cleaned meanings, %d rule ids, one answer each"
                  % (checked, len(first_seen)))


def check_original_rules():
    """f. The first 2,279 rules are untouched."""
    current = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_KB_PATH.read_text(encoding="utf-8"))
    current_rules = current.get("rules", [])
    original_rules = original.get("rules", [])
    if len(original_rules) != ORIGINAL_RULE_COUNT:
        return False, "the original file holds %d rules" % len(original_rules)
    if len(current_rules) < ORIGINAL_RULE_COUNT:
        return False, "the knowledge base holds %d rules" % len(current_rules)
    if current_rules[:ORIGINAL_RULE_COUNT] != original_rules:
        return False, ("the first %d rules differ from the original"
                       % ORIGINAL_RULE_COUNT)
    return True, "%d rules identical, %d added" % (
        ORIGINAL_RULE_COUNT, len(current_rules) - ORIGINAL_RULE_COUNT)





def check_help():
    """g. Every script prints its usage without an error."""
    problems = []
    for name in SCRIPT_NAMES:
        path = ENGINE_DIR / name
        if not path.exists():
            problems.append("%s is missing" % name)
            continue
        done = subprocess.run([sys.executable, str(path), "--help"],
                              capture_output=True, text=True)
        if done.returncode != 0 or "usage" not in done.stdout.lower():
            problems.append("%s --help failed" % name)
    if problems:
        return False, "; ".join(problems)
    return True, "%d scripts" % len(SCRIPT_NAMES)


def check_template_readings(reader):
    """h. Two readings build from the rules, with no model call."""
    class RefusingClient:
        def chat(self, *arguments, **keywords):
            raise AssertionError("a template reading must not call a model")

    saved = palm_reader.OLLAMA_CLIENT
    palm_reader.OLLAMA_CLIENT = RefusingClient()
    problems = []
    try:
        for figures, hand, gender in READING_CASES:
            override = parse_figures_argument(figures)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result = reader.read_palm(None, hand_type=hand, gender=gender,
                                          figures_override=override,
                                          use_vision=True)
            reading = result["reading"] if result else ""
            if not CITATION_RE.search(reading):
                problems.append("%s built no reading" % figures)
            elif "no model call" not in buffer.getvalue():
                problems.append("%s did not report a template build" % figures)
    except AssertionError as error:
        problems.append(str(error))
    finally:
        palm_reader.OLLAMA_CLIENT = saved
    if problems:
        return False, "; ".join(problems)
    return True, "%d readings" % len(READING_CASES)


def main():
    print("=" * 60)
    print("Engine checks")
    print("=" * 60)
    reader = load_reader()

    print("\na. Sentence rewrite tests")
    ok, detail = check_sensitivity()
    report("sensitivity rewrite", ok, detail)

    print("\nb. Picker count against the reading, figure by figure")
    ok, detail = check_figure_counts(reader)
    report("picker count == cited rules", ok, detail)

    print("\nc. Figures with no printable text")
    ok, detail = check_no_printable_figures(reader)
    report("exactly the six known figures", ok, detail)

    print("\nd. Rule ids against the knowledge base")
    ok, detail = check_rule_ids(reader)
    report("no stale rule ids", ok, detail)

    print("\ne. Cleaned rules file")
    ok, detail = check_clean_rules(reader)
    report("one status per id, kept words, no added grave words", ok, detail)

    print("\nf. Original knowledge base")
    ok, detail = check_original_rules()
    report("2,279 original rules identical", ok, detail)

    print("\ng. Script command lines")
    ok, detail = check_help()
    report("--help works", ok, detail)

    print("\nh. Template readings build with no model call")
    ok, detail = check_template_readings(reader)
    report("two readings", ok, detail)

    failed = [name for name, ok, detail in RESULTS if not ok]
    print("\n" + "=" * 60)
    if failed:
        print("%d of %d checks FAILED: %s"
              % (len(failed), len(RESULTS), ", ".join(failed)))
        return 1
    print("All %d checks passed." % len(RESULTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
