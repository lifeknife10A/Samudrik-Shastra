#!/usr/bin/env python3
"""
One-time clean-up of the figure rules into "condition / meaning" pairs.

The template reading prints one rule per sentence, which reads awkwardly when
the extraction put a noun phrase in the observation or a fragment in the
prediction. This script asks the local model to rewrite each usable figure
rule as:

    condition: a clause that can follow "when" and holds a verb
    meaning:   one or two sentences saying what the book says it indicates

Nothing is invented: the model sees the rule and the figure's own passage and
may use only words and facts from them. Every answer is then validated in
code, and a rejected answer leaves the original rule in place.

Usage:
    python3 engine/clean_figure_rules.py            # clean every figure
    python3 engine/clean_figure_rules.py --limit 5  # first 5 figures only
    python3 engine/clean_figure_rules.py --needs-review  # write the check list

Output:
    engine/figure_rules_clean.json    used by the reader
    engine/figure_rules_review.md     original vs cleaned, for checking
    engine/needs_review.md            the entries a check cannot settle

Re-running skips figures already in the output file, so a long run can be
resumed. knowledge_base.json and figure_index.json are never written.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from palm_reader import (GROUNDING_STOP_WORDS, OLLAMA_CLIENT,
                         PalmReader, TEXT_MODEL, has_verb)

FIGURE_INDEX_PATH = Path(__file__).parent / "figure_index.json"
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.json"
OUTPUT_PATH = Path(__file__).parent / "figure_rules_clean.json"
REVIEW_PATH = Path(__file__).parent / "figure_rules_review.md"

MIN_MEANING_WORDS = 6
RESTATEMENT_THRESHOLD = 0.60
MAX_INVENTED_WORDS = 3

# --- re-validation against the rule's own text ------------------------------
# A cleaned meaning may only say what the rule already said. These two checks
# enforce that: the meaning must keep enough of the rule's own words, and it
# may not introduce a grave or medical word the rule does not contain.
MIN_KEPT_SHARE = 0.50

# Words that carry no weight when measuring what a meaning kept.
KEPT_STOP_WORDS = {"book", "says", "such", "person", "individual", "their",
                   "will", "line"}

GRAVE_MEDICAL_WORDS = ("death", "die", "dies", "died", "widow", "widowhood",
                       "disease", "illness", "ailment", "tuberculosis",
                       "asthma", "kidney", "heart disease", "paralysis",
                       "blind", "poverty", "ruin", "separation", "divorce",
                       "killed", "accident")

UNVERIFIED_PATH = Path(__file__).parent / "figure_rules_clean.UNVERIFIED.json"

# ---------------------------------------------------------------------------
# Verb detection: a condition must say what the line does.



# ---------------------------------------------------------------------------
# Word helpers.
def content_words(text):
    """Lowercase words of 4+ letters, without the framing words."""
    words = re.findall(r"[a-z]{4,}", str(text).lower())
    return [word for word in words if word not in GROUNDING_STOP_WORDS]


def stems(text):
    """Five-letter stems, so "originates" matches "originating"."""
    return {word[:5] for word in content_words(text)}


def is_restatement(condition, meaning):
    """True when the meaning only says the condition again."""
    meaning_words = set(content_words(meaning))
    if not meaning_words:
        return True
    condition_words = set(content_words(condition))
    share = len(meaning_words & condition_words) / len(meaning_words)
    return share >= RESTATEMENT_THRESHOLD


def invented_words(meaning, known_text):
    """Content words of the meaning that are nowhere in the sources."""
    known = stems(known_text)
    return [word for word in set(content_words(meaning))
            if word[:5] not in known]


# --- re-validation helpers -------------------------------------------------
def kept_stems(text):
    """Five-letter stems of a text's words, without the filler words."""
    words = re.findall(r"[a-z]{4,}", str(text).lower())
    return {word[:5] for word in words if word not in KEPT_STOP_WORDS}


def kept_share(meaning, source):
    """Share of the source's own words that the meaning still carries."""
    source_stems = kept_stems(source)
    if not source_stems:
        return 1.0
    return len(kept_stems(meaning) & source_stems) / len(source_stems)


def is_real_prediction(rule):
    """True when the prediction says something of its own.

    A prediction is only a real prediction when it is longer than a fragment
    and does not simply restate the observation.
    """
    prediction = str(rule.get("prediction") or "").strip()
    if len(prediction.split()) < 5:
        return False
    return not is_restatement(str(rule.get("observation") or ""), prediction)


def best_passage_sentence(observation, passages):
    """The passage sentence whose words overlap the observation most."""
    observation_stems = kept_stems(observation)
    best, best_score = "", 0.0
    for passage in passages:
        for sentence in re.split(r"(?<=[.!?])\s+", str(passage)):
            sentence = sentence.strip()
            stems_here = kept_stems(sentence)
            if not stems_here:
                continue
            union = stems_here | observation_stems
            score = len(stems_here & observation_stems) / len(union) if union else 0.0
            if score > best_score:
                best, best_score = sentence, score
    return best


def introduced_grave_word(meaning, rule_text):
    """A grave or medical word the meaning adds, or None."""
    for word in GRAVE_MEDICAL_WORDS:
        pattern = r"\b%s\b" % re.escape(word)
        if re.search(pattern, meaning, re.IGNORECASE) and not re.search(
                pattern, rule_text, re.IGNORECASE):
            return word
    return None


def source_for_rule(rule, passages):
    """The text a cleaned meaning must come from, and which kind it is."""
    if is_real_prediction(rule):
        return str(rule.get("prediction") or ""), "prediction"
    return best_passage_sentence(rule.get("observation"), passages), "passage sentence"


def revalidate(rule, passages, condition, meaning):
    """Returns (status, reason, share) for a candidate cleaned answer."""
    rule_text = " ".join(str(rule.get(key) or "")
                         for key in ("observation", "prediction", "variation"))
    added = introduced_grave_word(meaning, rule_text)
    if added:
        return "original", "adds '%s', which the rule does not say" % added, 0.0
    if not condition or not has_verb(condition):
        return "original", "condition has no verb", 0.0
    if len(meaning.split()) < MIN_MEANING_WORDS:
        return "original", "meaning under %d words" % MIN_MEANING_WORDS, 0.0
    if is_restatement(condition, meaning):
        return "original", "meaning restates the condition", 0.0
    if SELF_HARM_METHOD_RE.search(condition + " " + meaning):
        return "original", "self-harm wording in the cleaned text", 0.0
    source, kind = source_for_rule(rule, passages)
    share = kept_share(meaning, source)
    if share < MIN_KEPT_SHARE:
        return ("original",
                "keeps only %d%% of the %s words" % (round(share * 100), kind),
                share)
    return ("cleaned" if kind == "prediction" else "from_passage", "", share)


# A cleaned meaning must not introduce self-harm wording. The reader's own
# filter matches "drowning himself", not the bare method noun, so a meaning
# like "extreme actions such as drowning, immolation, or poisoning" would slip
# past it. Such meanings are rejected here instead.
SELF_HARM_METHOD_RE = re.compile(
    r"\b(?:immolat\w*|drown\w*|poison\w*|hang\w*|suicid\w*|kill\w*)\b",
    re.IGNORECASE)


def enforce_self_harm_guard(clean):
    """Downgrades cleaned entries that carry self-harm wording."""
    changed = 0
    for line in clean:
        for figure in clean[line]:
            for entry in clean[line][figure]:
                if entry.get("status") != "cleaned":
                    continue
                text = "%s %s" % (entry.get("condition", ""),
                                  entry.get("meaning", ""))
                if SELF_HARM_METHOD_RE.search(text):
                    entry["status"] = "original"
                    entry["reason"] = "self-harm wording in the cleaned text"
                    entry["condition"] = ""
                    entry["meaning"] = ""
                    changed += 1
    return changed


def merge_duplicate_rules(clean):
    """One status per rule id, so the reader cannot load two different ones.

    A rule can be linked to two figures (rule_113_2 belongs to marriage 3 and
    marriage 4), and the clean-up may judge the two copies differently. The
    reader keys by rule id alone, so the copies are merged into one entry: a
    safety reject (self-harm) wins over a cleaned copy, otherwise the first
    cleaned copy is kept. Every merge is printed.
    """
    copies = {}
    order = []
    for line in sorted(clean):
        for figure in sorted(clean[line], key=int):
            for entry in clean[line][figure]:
                rule_id = entry.get("rule_id")
                if rule_id not in copies:
                    copies[rule_id] = []
                    order.append(rule_id)
                copies[rule_id].append(entry)

    merged = 0
    for rule_id in order:
        group = copies[rule_id]
        if len(group) < 2:
            continue
        statuses = sorted({entry["status"] for entry in group})
        safety = [entry for entry in group
                  if entry["status"] == "original"
                  and "self-harm" in (entry.get("reason") or "")]
        cleaned = [entry for entry in group if entry["status"] == "cleaned"]
        if safety:
            keeper = dict(safety[0])
        elif cleaned:
            keeper = dict(cleaned[0])
        else:
            keeper = dict(group[0])
        if all(entry == keeper for entry in group):
            continue
        for entry in group:
            entry.clear()
            entry.update(keeper)
        merged += 1
        print("  merged %s: %s -> %s"
              % (rule_id, "/".join(statuses), keeper["status"]))
    return merged


# ---------------------------------------------------------------------------
CLEAN_SCHEMA = {
    "type": "object",
    "properties": {
        "condition": {"type": "string"},
        "meaning": {"type": "string"},
        "keep": {"type": "boolean"},
    },
    "required": ["condition", "meaning", "keep"],
}

CLEAN_PROMPT = """You are cleaning one rule from the classical palmistry book
Samudrik Shastra so that it can be printed as one sentence.

You are given the rule (observation and prediction) and the book passage it
came from. Rewrite it as:
- "condition": one clause that can follow the word "when". It must contain a
  verb and describe something visible on the hand, for example "the Fate Line
  has a break" or "the Health Line is deep, long, and clear".
- "meaning": one or two sentences saying what the book says this indicates,
  written in the third person, for example "the book says that such a person
  is fortunate and overcomes obstacles". It must not merely repeat the
  condition.
- "keep": false when the passage gives no meaning for this rule at all, for
  example when the prediction is missing, cut off, or says nothing.

Use only words and facts that appear in the rule or the passage. Add no new
claim, number, date, timing or advice.

Rule:
"""


def ask_model(rule, passages):
    """One model call for one rule."""
    passage = " ".join(passages)
    if len(passage) > 2500:
        passage = passage[:2500]
    prompt = (CLEAN_PROMPT
              + "observation: %s\n" % (rule.get("observation") or "")
              + "prediction: %s\n" % (rule.get("prediction") or "")
              + "\nPassage:\n" + passage)
    response = OLLAMA_CLIENT.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 600},
        think=False,
        format=CLEAN_SCHEMA,
    )
    return json.loads(response["message"]["content"])


def judge(rule, passages, answer):
    """Returns (status, reason, condition, meaning)."""
    if not answer.get("keep", False):
        return "dropped", "model says the passage gives no meaning", "", ""
    condition = str(answer.get("condition") or "").strip().rstrip(".")
    meaning = str(answer.get("meaning") or "").strip()
    if not condition:
        return "original", "empty condition", "", ""
    if not has_verb(condition):
        return "original", "condition has no verb", "", ""
    if len(meaning.split()) < MIN_MEANING_WORDS:
        return "original", "meaning under %d words" % MIN_MEANING_WORDS, "", ""
    if is_restatement(condition, meaning):
        return "original", "meaning restates the condition", "", ""
    if SELF_HARM_METHOD_RE.search(condition + " " + meaning):
        return "original", "self-harm wording in the cleaned text", "", ""
    known_text = " ".join([str(rule.get("observation") or ""),
                           str(rule.get("prediction") or ""),
                           str(rule.get("variation") or "")] + list(passages))
    invented = invented_words(meaning, known_text)
    if len(invented) > MAX_INVENTED_WORDS:
        return ("original",
                "invented words: %s" % ", ".join(sorted(invented)[:5]), "", "")
    return "cleaned", "", condition, meaning


# ---------------------------------------------------------------------------
def clean_one(reader, line, figure, entry):
    """Cleans every usable rule of one figure."""
    usable, _ = reader.usable_rules_for_figure(line, figure, "present")
    passages = entry.get("passages", [])
    results = []
    for rule_id, rule in usable:
        try:
            answer = ask_model(rule, passages)
        except Exception as error:
            results.append({"rule_id": rule_id, "condition": "",
                            "meaning": "", "status": "original",
                            "reason": "model call failed: %s" % error})
            continue
        status, reason, condition, meaning = judge(rule, passages, answer)
        results.append({"rule_id": rule_id, "condition": condition,
                        "meaning": meaning, "status": status,
                        "reason": reason})
        time.sleep(0.2)
    return results


def write_review(clean, reader):
    """Original and cleaned side by side, so the pairs can be checked."""
    lines = ["# Figure rules: original vs cleaned", "",
             "Generated by `engine/clean_figure_rules.py`.", ""]
    for line in sorted(clean):
        for figure in sorted(clean[line], key=int):
            entries = clean[line][figure]
            if not entries:
                continue
            lines.append("## %s figure %s" % (line, figure))
            lines.append("")
            for entry in entries:
                rule = reader._rule_by_id.get(entry["rule_id"], {})
                title = "%s — %s" % (entry["rule_id"], entry["status"])
                if entry["reason"]:
                    title += " (%s)" % entry["reason"]
                lines.append("### %s" % title)
                lines.append("")
                lines.append("**Original observation:** %s"
                             % (rule.get("observation") or "—"))
                lines.append("")
                lines.append("**Original prediction:** %s"
                             % (rule.get("prediction") or "—"))
                lines.append("")
                if entry["status"] in ("cleaned", "from_passage"):
                    lines.append("**Cleaned condition:** %s" % entry["condition"])
                    lines.append("")
                    lines.append("**Cleaned meaning** (%s, kept %s%% of the %s "
                                 "words): %s"
                                 % (entry["status"],
                                    entry.get("kept_percent", "?"),
                                    "prediction" if entry["status"] == "cleaned"
                                    else "passage sentence",
                                    entry["meaning"]))
                elif entry["status"] == "dropped":
                    lines.append("**Dropped:** no sentence is printed for it.")
                else:
                    lines.append("**Kept original:** the template sentence is "
                                 "used unchanged%s."
                                 % (" (%s)" % entry["reason"]
                                    if entry.get("reason") else ""))
                lines.append("")
            lines.append("")
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_revalidation(reader):
    """Re-checks the UNVERIFIED clean-up against each rule's own text.

    A meaning that fails the check is sent to the model once more and the new
    answer is validated the same way; if it fails again the rule keeps its
    original prediction.
    """
    data = json.loads(UNVERIFIED_PATH.read_text(encoding="utf-8"))
    counts = {"cleaned": 0, "from_passage": 0, "original": 0}
    reasons = {}
    kept = []
    reasked = 0
    started = time.time()

    for line in sorted(data):
        for figure in sorted(data[line], key=int):
            passages = reader.figure_index[line][figure].get("passages", [])
            for entry in data[line][figure]:
                rule = reader._rule_by_id.get(entry["rule_id"], {})
                if entry.get("status") == "original":
                    counts["original"] += 1
                    entry["kept_percent"] = 0
                    continue
                status, reason, share = revalidate(
                    rule, passages, entry.get("condition", ""),
                    entry.get("meaning", ""))
                if status == "original":
                    reasked += 1
                    print("    re-asking %s (%s)" % (entry["rule_id"], reason))
                    try:
                        answer = ask_model(rule, passages)
                    except Exception as error:
                        answer = None
                        print("      [WARN] model call failed: %s" % error)
                    if answer and answer.get("keep"):
                        condition = str(answer.get("condition") or "").strip().rstrip(".")
                        meaning = str(answer.get("meaning") or "").strip()
                        new_status, new_reason, new_share = revalidate(
                            rule, passages, condition, meaning)
                        if new_status != "original":
                            entry["condition"] = condition
                            entry["meaning"] = meaning
                            status, reason, share = new_status, new_reason, new_share
                entry["status"] = status
                entry["reason"] = reason
                entry["kept_percent"] = round(share * 100)
                if status == "original":
                    entry["condition"] = ""
                    entry["meaning"] = ""
                counts[status] += 1
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
                kept.append(share)
        print("  %s done (%d cleaned, %d from passage, %d original so far)"
              % (line, counts["cleaned"], counts["from_passage"],
                 counts["original"]))
        # A rule shared by two figures can be judged differently for each, so
        # the copies are merged again before the file is written.
        merge_duplicate_rules(data)
        OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                               encoding="utf-8")

    merged = merge_duplicate_rules(data)
    if merged:
        print("  merged %d rule id(s) that appeared more than once" % merged)
    write_review(data, reader)
    print("\nRe-validation counts")
    for status in ("cleaned", "from_passage", "original"):
        print("  %-12s %d" % (status, counts[status]))
    if reasons:
        print("  reasons:")
        for reason in sorted(reasons, key=lambda r: -reasons[r]):
            print("    %-48s %d" % (reason, reasons[reason]))
    print("\n  model re-asks: %d" % reasked)
    kept_shares = [entry.get("kept_percent", 0) for line in data
                   for figure in data[line] for entry in data[line][figure]
                   if entry["status"] in ("cleaned", "from_passage")]
    if kept_shares:
        print("  kept-word share among kept entries: lowest %d%%, "
              "average %d%%" % (min(kept_shares),
                                round(sum(kept_shares) / len(kept_shares))))
    print("  time: %.1f min" % ((time.time() - started) / 60.0))
    print("  rules:  %s" % OUTPUT_PATH.name)
    print("  review: %s" % REVIEW_PATH.name)


NEEDS_REVIEW_PATH = Path(__file__).parent / "needs_review.md"


def write_needs_review(reader):
    """The entries a human should look at, in one list.

    The automatic checks settle most of the file. These two groups cannot be
    settled by a check: a meaning that came from a passage sentence instead of
    the rule's own prediction, and two different conditions that were given
    the same meaning.
    """
    clean = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    unverified = {}
    if UNVERIFIED_PATH.exists():
        unverified = json.loads(UNVERIFIED_PATH.read_text(encoding="utf-8"))

    lines = ["# Figure rules that need a human check", "",
             "Generated by `engine/clean_figure_rules.py --needs-review`.", "",
             "Every other entry in `figure_rules_clean.json` passed the",
             "automatic checks. These are the ones a check cannot settle.", ""]

    lines.append("## 1. Meanings taken from a passage sentence "
                 "(`from_passage`)")
    lines.append("")
    lines.append("The rule's own prediction was too short or only repeated the")
    lines.append("observation, so the meaning was written from the figure's own")
    lines.append("passage instead. Please check that the sentence below really")
    lines.append("supports the meaning.")
    lines.append("")
    checked_at = {}
    for line in sorted(clean):
        for figure in sorted(clean[line], key=int):
            index_entry = reader.figure_index.get(line, {}).get(figure, {})
            passages = index_entry.get("passages", [])
            for entry in clean[line][figure]:
                if entry.get("status") != "from_passage":
                    continue
                rule = reader._rule_by_id.get(entry["rule_id"], {})
                sentence = best_passage_sentence(rule.get("observation"),
                                                 passages)
                pages = index_entry.get("printed_pages", [])
                # One rule can belong to two figures. The clean-up checks it
                # once and copies the answer, so the sentence shown has to be
                # the one it was checked against.
                shared = ""
                if entry["rule_id"] in checked_at:
                    other_line, other_figure, sentence = \
                        checked_at[entry["rule_id"]]
                    shared = ("- **Same rule as %s figure %s:** the answer was "
                              "checked once there, so the sentence below is "
                              "from that figure." % (other_line, other_figure))
                else:
                    checked_at[entry["rule_id"]] = (line, figure, sentence)
                lines.append("### %s — %s figure %s"
                             % (entry["rule_id"], line, figure))
                lines.append("")
                if pages:
                    lines.append("- **Book page(s):** %s"
                                 % ", ".join(str(page) for page in pages))
                lines.append("- **Book observation:** %s"
                             % (rule.get("observation") or "—"))
                lines.append("- **Book prediction:** %s"
                             % (rule.get("prediction") or "—"))
                lines.append("- **Passage sentence the meaning was checked "
                             "against:** %s" % (sentence or "—"))
                lines.append("- **Condition written:** %s"
                             % entry.get("condition"))
                lines.append("- **Meaning written:** %s" % entry.get("meaning"))
                lines.append("- **Kept %s%% of that sentence's own words.**"
                             % entry.get("kept_percent"))
                if shared:
                    lines.append(shared)
                # A meaning can draw on a second sentence of the figure as
                # well, so that sentence is shown when the meaning's own words
                # clearly come from it.
                meaning_sentence = best_passage_sentence(entry.get("meaning"),
                                                         passages)
                if meaning_sentence and meaning_sentence != sentence:
                    together = kept_stems(entry.get("meaning")) & \
                        kept_stems(meaning_sentence)
                    if len(together) >= 3:
                        lines.append("- **Also draws on:** \"%s\""
                                     % meaning_sentence)
                lines.append("")

    lines.append("## 2. One meaning for two different conditions "
                 "(health line, figure 1)")
    lines.append("")
    lines.append("The clean-up gave both rules below the same meaning, so the")
    lines.append("reading could not tell a vague line from a clear one. Both are")
    lines.append("printed from the book's own prediction now, so nothing false")
    lines.append("reaches the reading, but the pair is worth a look.")
    lines.append("")
    for rule_id in ("rule_35_1", "rule_35_2"):
        rule = reader._rule_by_id.get(rule_id, {})
        lines.append("**%s**" % rule_id)
        lines.append("")
        lines.append("- **Book observation:** %s"
                     % (rule.get("observation") or "—"))
        lines.append("- **Book prediction:** %s"
                     % (rule.get("prediction") or "—"))
        earlier = None
        for entry in unverified.get("health_line", {}).get("1", []):
            if entry.get("rule_id") == rule_id:
                earlier = entry
        if earlier:
            lines.append("- **Meaning the clean-up first wrote:** %s -> %s"
                         % (earlier.get("condition"), earlier.get("meaning")))
        for entry in clean.get("health_line", {}).get("1", []):
            if entry.get("rule_id") == rule_id:
                lines.append("- **Status in the current file:** %s (%s)"
                             % (entry.get("status"), entry.get("reason")))
        lines.append("")

    health_entry = reader.figure_index.get("health_line", {}).get("1", {})
    lines.append("**Page %s passage the two conditions come from:**"
                 % ", ".join(str(page)
                             for page in health_entry.get("printed_pages", [])))
    lines.append("")
    for passage in health_entry.get("passages", []):
        lines.append("> %s" % passage)
        lines.append("")

    NEEDS_REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Clean figure rules into condition / meaning pairs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only clean the first N figures")
    parser.add_argument("--revalidate", action="store_true",
                        help="Re-check the UNVERIFIED clean-up against each "
                             "rule's own prediction or passage sentence")
    parser.add_argument("--needs-review", action="store_true",
                        help="Write engine/needs_review.md from the current "
                             "rules file")
    args = parser.parse_args()

    reader = PalmReader.__new__(PalmReader)
    reader.figure_index = json.loads(FIGURE_INDEX_PATH.read_text(encoding="utf-8"))
    reader.kb = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    reader._rule_by_id = reader._build_rule_lookup()

    if args.needs_review:
        if not OUTPUT_PATH.exists():
            print("[ERROR] %s not found" % OUTPUT_PATH.name)
            return
        write_needs_review(reader)
        print("  needs review: %s" % NEEDS_REVIEW_PATH.name)
        return

    if args.revalidate:
        if not UNVERIFIED_PATH.exists():
            print("[ERROR] %s not found" % UNVERIFIED_PATH.name)
            return
        run_revalidation(reader)
        return

    clean = {}
    if OUTPUT_PATH.exists():
        try:
            clean = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            done = sum(len(clean[line][fig]) for line in clean for fig in clean[line])
            print("Resuming: %d figures already cleaned (%d rules)"
                  % (sum(len(clean[line]) for line in clean), done))
        except Exception as error:
            print("[WARN] could not read the existing output (%s); starting fresh"
                  % error)
            clean = {}

    targets = []
    for line in sorted(reader.figure_index):
        for figure in sorted(reader.figure_index[line], key=int):
            if clean.get(line, {}).get(figure):
                continue
            targets.append((line, figure))
    if args.limit:
        targets = targets[:args.limit]

    print("Figures to clean: %d" % len(targets))
    repaired = enforce_self_harm_guard(clean)
    if repaired:
        print("Self-harm guard: %d cleaned entry/entries kept as original"
              % repaired)
    merged = merge_duplicate_rules(clean)
    if merged:
        print("Merged %d rule id(s) that appeared more than once" % merged)
    started = time.time()

    for index, (line, figure) in enumerate(targets, 1):
        entry = reader.figure_index[line][figure]
        print("  [%d/%d] %s figure %s..." % (index, len(targets), line, figure),
              end="", flush=True)
        results = clean_one(reader, line, figure, entry)
        clean.setdefault(line, {})[figure] = results
        print(" %d rules" % len(results))
        OUTPUT_PATH.write_text(json.dumps(clean, indent=2, ensure_ascii=False),
                               encoding="utf-8")

    # Merge again, in case this run cleaned a figure whose rules are shared
    # with a figure that was already done.
    merge_duplicate_rules(clean)

    # Count the whole file, so a resumed run still reports the real totals.
    counts = {"cleaned": 0, "original": 0, "dropped": 0}
    reasons = {}
    for line in clean:
        for figure in clean[line]:
            for result in clean[line][figure]:
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                if result.get("reason"):
                    reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
    OUTPUT_PATH.write_text(json.dumps(clean, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    write_review(clean, reader)

    print("\nClean-up counts")
    print("  cleaned : %d" % counts["cleaned"])
    print("  original: %d" % counts["original"])
    print("  dropped : %d" % counts["dropped"])
    if reasons:
        print("  reasons:")
        for reason in sorted(reasons, key=lambda r: -reasons[r]):
            print("    %-42s %d" % (reason, reasons[reason]))
    print("\n  time: %.1f min" % ((time.time() - started) / 60.0))
    print("  rules:  %s" % OUTPUT_PATH.name)
    print("  review: %s" % REVIEW_PATH.name)


if __name__ == "__main__":
    main()
