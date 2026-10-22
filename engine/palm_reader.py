#!/usr/bin/env python3
"""
Samudrik Shastra Palm Reader - figure-matcher pipeline.

1. Vision (qwen2.5vl:7b) reports the hand side and whether each of the 7
   major lines is present, absent or unclear. No descriptions.
2. For every present or unclear line the user picks the matching figure from
   the book's contact sheet (or "none match").
3. Rules come from that figure's passages and the rules extracted from the
   same pages.
4. qwen3:8b synthesises the reading, citing [Rn] rule numbers.

Usage:
    python3 engine/palm_reader.py palm.jpg --gender male
    python3 engine/palm_reader.py palm.jpg --figures heart_line=3,fate_line=5
    python3 engine/palm_reader.py --figures head_line=2 --hand right

100% offline. Zero API keys.
"""

import json
import base64
import re
import sys
import time
import argparse
from io import BytesIO
from pathlib import Path

try:
    import ollama
    OLLAMA_CLIENT = ollama.Client(timeout=600.0)
except ImportError:
    print("[ERROR] ollama package not installed. Run: pip3 install ollama")
    sys.exit(1)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("[ERROR] chromadb not installed. Run: pip3 install chromadb sentence-transformers")
    sys.exit(1)

# Same degenerate-rule filter the vector database uses, so figure rules are
# filtered exactly like database rules.
from build_vectordb import is_degenerate_rule

VISION_MODEL = "qwen2.5vl:7b"
TEXT_MODEL = "qwen3:8b"
VECTOR_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "samudrik_shastra_rules"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FIGURE_INDEX_PATH = Path(__file__).parent / "figure_index.json"
CLEAN_RULES_PATH = Path(__file__).parent / "figure_rules_clean.json"
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.json"
CONTACT_SHEETS_DIR = Path(__file__).parent.parent / "illustrations" / "contact_sheets"
HANDS_DIR = Path(__file__).parent.parent / "illustrations" / "cropped_hands"

MAJOR_LINES = ["heart_line", "head_line", "life_line", "fate_line",
               "sun_line", "health_line", "marriage_line"]

LINE_PLATES = {
    "heart_line": "Heart_Line",
    "head_line": "Head_Line",
    "life_line": "Life_Line",
    "fate_line": "Fate_Line",
    "sun_line": "Sun_Line",
    "health_line": "Health_Line",
    "marriage_line": "Marriage_Line",
}

VISION_ANALYSIS_PROMPT = """You are an expert in Samudrik Shastra (Indian Palmistry).
Analyze this palm image and report ONLY the presence of the seven major lines.

First decide the hand side: "left" or "right" (look at the thumb position).
Use "unknown" if you cannot tell.

Then for each line answer exactly one of:
- "present" - you can clearly see the line
- "absent" - the line is not there at all
- "unclear" - you cannot tell confidently

Lines: heart_line, head_line, life_line, fate_line, sun_line, health_line,
marriage_line.

Be conservative: when in doubt answer "unclear". Do not invent lines.
Answer with JSON only."""

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "hand_type": {"type": "string", "enum": ["left", "right", "unknown"]},
        "lines": {
            "type": "object",
            "properties": {
                line: {"type": "string", "enum": ["present", "absent", "unclear"]}
                for line in MAJOR_LINES
            },
            "required": MAJOR_LINES,
            "additionalProperties": False,
        },
    },
    "required": ["hand_type", "lines"],
}


# --- sentence-level safety check -------------------------------------------
# The finished reading is checked one sentence at a time. A sentence is
# rewritten only when it carries a grave topic AND states it as a certainty,
# or states the grave event outright. A sentence without a grave word is
# never touched, so ordinary predictions stay exactly as written.

GRAVE_WORDS = ("death", "die", "dies", "died", "untimely", "widow",
               "widowhood", "suicide", "fatal", "killed", "drowning", "perish")

CERTAINTY_MARKERS = ("certain", "inevitable", "inevitably", "surely",
                     "definitely", "undoubtedly", "bound to", "will die",
                     "will meet", "is sure to", "destined", "certainly")

# A plain "will <verb>" plus a grave or medical word is a certainty too
# ("will culminate in widowhood", "will become a victim of heart diseases").
WILL_VERB_RE = re.compile(r"\bwill\s+[a-z]+", re.IGNORECASE)

# A grave event stated as a plain fact, with no certainty word needed.
# Word boundaries keep "dies" out of "bodies" and "studies".
GRAVE_FACT_RE = re.compile(
    r"\b(?:death occurs|dies|will die|will perish|perishes|perish|"
    r"meets (?:his|her|their) end|loses (?:his|her|their) life)\b",
    re.IGNORECASE)

# Statements of the grave event itself, with no certainty word needed.
DIRECT_GRAVE_PHRASES = ("untimely death", "meets death", "meet death",
                        "foretells death", "predicts death", "dies young",
                        "die young", "ends their life", "end their life",
                        "death in a foreign land", "certain death")

# When a grave word sits inside a protective phrase it is good news, not a
# warning ("free from untimely death", "safe from danger"), so the sentence
# is left alone.
PROTECTIVE_PHRASES = ("free from", "free of", "protected from", "no fear of",
                      "escapes", "escape from", "safe from", "long life",
                      "healthy life")

SAFE_SENTENCE = ("The book links this sign with a serious danger; this is a "
                 "traditional belief, not a certainty.")

# --- medical claims --------------------------------------------------------
# The book states illnesses as fact. A certain medical claim is rewritten; a
# plain statement of illness keeps its sentence and gains a note. Plurals and
# the "weakness of <organ>" phrasing are matched too.
MEDICAL_WORDS = ("disease", "illness", "ailment", "fever", "infection",
                 "tuberculosis", "asthma", "kidney", "palpitations",
                 "paralysis", "blindness", "epilepsy", "insanity")

MEDICAL_PHRASE_RE = re.compile(r"\bweakness\s+of\s+(?:the\s+)?[a-z]+",
                               re.IGNORECASE)

MEDICAL_SENTENCE = ("The book links this sign with a risk to health; this is a "
                    "traditional belief, not a medical finding.")

MEDICAL_NOTE = " (a traditional belief, not a medical finding)"

# --- self-harm suppression -------------------------------------------------
# Self-harm is handled separately from the grave-outcome check: it is
# replaced whatever the hedging ("may", "could"), because no phrasing of it
# belongs in a reading. "Jumping into water" is not self-harm on its own
# (someone can jump in to rescue a child), so it only matches as the manner
# of a method ("by jumping into water").
SELF_HARM_RE = re.compile(
    r"\b(?:suicid\w*"                                  # suicide, suicidal
    r"|self[- ]destruct\w*"                            # self-destruction
    r"|kill(?:s|ed)?\s+(?:himself|herself|themselves|oneself)"
    r"|tak(?:e|es|ing)\s+(?:his|her|their)\s+own\s+life"
    r"|tak(?:e|es|ing)\s+poison"
    r"|harm(?:s|ed)?\s+(?:himself|herself|themselves)"
    r"|end(?:s|ed)?\s+(?:his|her|their)\s+life"
    r"|play\s+of\s+(?:his|her|their)\s+life"
    r"|set(?:s|ting)?\s+(?:it|him|her|his\s+body|her\s+body)\s+on\s+fire"
    r"|(?:drown|hang|poison)(?:s|ed|ing)?\s+(?:him|her|them)sel(?:f|ves)"
    r"|by\s+jump(?:s|ing|ed)?\s+into\s+water)\b",
    re.IGNORECASE)

SELF_HARM_PREDICTION = "The book associates this sign with severe distress."

HELPLINE_NOTE = ("If you are going through a difficult time, please talk to "
                 "someone you trust or contact a local helpline.")

# --- rule filters ----------------------------------------------------------
# Some linked rules cannot describe the user's palm: the prediction is a
# placeholder, the observation only points at a place in the printed figure,
# or the rule describes a line being absent while the user picked a figure
# from a line that is there.

PLACEHOLDER_WORDS = ("specified", "mentioned", "stated", "provided", "given")

# Words that carry no content of their own, so "not specified in the
# passage" is still only a placeholder.
PLACEHOLDER_FILLER_WORDS = {"not", "in", "the", "a", "an", "of", "is", "are",
                            "was", "were", "this", "that", "passage", "text",
                            "source", "book", "rule", "available", "any"}

FIGURE_REF_RE = re.compile(r"\b(?:figure|fig\.?)\s*(?:no\.?\s*)?\d+",
                           re.IGNORECASE)

# Words that point at a spot in a plate rather than describe a feature,
# as in "see the site of line no. 1 marked with a dot in Figure no. 1".
FIGURE_POINTER_WORDS = {"figure", "fig", "no", "number", "line", "lines",
                        "see", "site", "sites", "observe", "observed",
                        "note", "noted", "marked", "mark", "dot", "dotted",
                        "necessary", "also"}

ABSENT_LINE_RE = re.compile(
    r"\b(?:absent|missing|not\s+present|entirely\s+absent)\b", re.IGNORECASE)

# Two rules for the same figure are duplicates when their words overlap this
# much (Jaccard); the first one is kept.
DUPLICATE_THRESHOLD = 0.6

# How many rules a single line may contribute to a template reading.
MAX_RULES_PER_LINE = 6

# A prediction that only repeats its observation makes a tautology
# ("when X, X"), so no sentence is printed for it either.
RESTATEMENT_THRESHOLD = 0.6

# A sentence contained in an earlier one is a repeat too, even when the
# lengths differ (Jaccard alone misses a short sentence inside a long one).
CONTAINMENT_THRESHOLD = 0.8

# --- verb detection --------------------------------------------------------
# Used to tell a condition from a fragment ("the individual possessing such a
# line" has no finite verb).
AUXILIARY_VERBS = {"is", "are", "was", "were", "be", "been", "being", "has",
                   "have", "had", "does", "do", "did", "can", "could", "may",
                   "might", "must", "will", "would", "shall", "should"}

BASE_VERBS = {
    "accept", "accompany", "accumulate", "achieve", "act", "admit",
    "advance", "announce", "appear", "arrive", "ascend", "attain", "avoid",
    "bear", "become", "begin", "bend", "bestow", "bind", "break", "bring",
    "carry", "catch", "cause", "chase", "claim", "come", "confirm",
    "connect", "contain", "continue", "cross", "curve", "cut", "decline",
    "declare", "demand", "denote", "deny", "descend", "describe", "deserve",
    "desire", "develop", "die", "display", "divide", "earn", "end", "engage",
    "enjoy", "entail", "establish", "exhibit", "exist", "expand",
    "experience", "express", "extend", "face", "fail", "fall", "fear",
    "flourish", "follow", "form", "fork", "generate", "give", "go", "grant",
    "grow", "hold", "imply", "improve", "incline", "increase", "indicate",
    "influence", "intersect", "involve", "join", "justify", "lead", "leave",
    "lie", "live", "love", "maintain", "make", "mark", "marry", "mean",
    "meet", "mention", "merge", "merit", "obtain", "occupy", "originate",
    "pass", "possess", "precede", "prefer", "prioritize", "produce",
    "progress", "promise", "prosper", "prove", "pursue", "reach", "reduce",
    "reflect", "refuse", "reject", "remain", "render", "represent",
    "require", "result", "reveal", "rise", "rule", "run", "secure",
    "separate", "serve", "show", "signify", "sit", "slant", "spend",
    "split", "start", "state", "stay", "struggle", "succeed", "suffer",
    "suggest", "surround", "symbolize", "take", "tend", "terminate",
    "touch", "travel", "turn", "wander", "worsen", "yield",
}

# A short prediction with no verb is a fragment; a longer one is left alone,
# because a verb list can never be complete ("reveals", "renders", ...).
FRAGMENT_MAX_WORDS = 8


def build_verb_forms():
    """Every simple form of the base verbs, including participles."""
    forms = set()
    for base in BASE_VERBS:
        forms.add(base)
        forms.add(base + "s")
        forms.add(base + "es")
        forms.add(base + "ed")
        forms.add(base + "ing")
        if base.endswith("e"):
            forms.add(base[:-1] + "ing")          # curve -> curving
        if base.endswith("y"):
            forms.add(base[:-1] + "ies")          # carry -> carries
        if base.endswith(("t", "p", "g", "n")):
            forms.add(base + base[-1] + "ing")    # cut -> cutting
    return forms


VERB_FORMS = build_verb_forms()

# Finite forms only: no -ing or -ed participle.
FINITE_VERB_FORMS = {form for form in VERB_FORMS
                     if not form.endswith(("ing", "ed"))}


def has_verb(clause):
    """True when the clause holds a verb, an auxiliary or a modal."""
    for word in re.findall(r"[a-z]+", str(clause).lower()):
        if word in AUXILIARY_VERBS or word in VERB_FORMS:
            return True
        if word.endswith("ing") and len(word) > 5:
            return True
    return False


def has_finite_verb(text):
    """True when the text holds a finite verb, so it is not a fragment."""
    for word in re.findall(r"[a-z]+", str(text).lower()):
        if word in AUXILIARY_VERBS or word in FINITE_VERB_FORMS:
            return True
    return False

# --- template reading ------------------------------------------------------
# First words that keep their capital letter inside "when ...".
KEEP_CAPITAL_WORDS = {"Heart", "Head", "Life", "Fate", "Sun", "Health",
                      "Marriage", "Mount", "Jupiter", "Saturn", "Mercury",
                      "Venus", "Moon", "Mars"}

# The polish pass only rewrites the template for flow; it may not add or
# drop statements, and the grounding check runs on its output afterwards.
POLISH_PROMPT = """You are editing a palm reading that was written directly from
the book's rules. Rewrite it so it flows, without adding, removing or changing
any statement. Keep every [Rn] citation attached to the same claim, keep the
same sections and headings, and do not add a conclusion, advice or remedy.
Return only the rewritten reading.

Reading:
"""


# One sentence, plus any [Rn] citations that trail it.
SENTENCE_RE = re.compile(r"([^.!?]+[.!?]+)((?:\s*\[R\d+\])*)")
CITATION_RE = re.compile(r"\[R\d+\]")

# --- grounding check -------------------------------------------------------
# A sentence survives only when it cites rules and draws its own words from
# them; this is what removes the model's commentary about effort and destiny.
GROUNDING_THRESHOLD = 0.35

GROUNDING_STOP_WORDS = {
    # words the task names
    "line", "lines", "that", "this", "these", "those", "with", "indicates",
    "indicate", "suggests", "suggest", "person", "people", "book",
    # framing verbs and linkers
    "says", "said", "states", "stated", "according", "however", "therefore",
    "moreover", "furthermore", "thus", "hence", "also", "while", "whereas",
    "although", "though", "about", "because", "being", "between", "other",
    "among", "upon", "within", "without", "their", "there", "which", "when",
    "what", "from", "have", "will", "would", "could", "should", "they",
    "them", "then", "than", "into", "over", "such", "some", "more", "most",
    "very", "does", "your", "yours", "each", "both", "been", "were", "what",
}

# Lines that are structure, not prose: headings, the hand line, the rule.
# Compared in lowercase, because the polish pass may change the capitalisation.
GROUNDING_SKIP_PREFIXES = ("#", "**hand analyzed:**",
                           "*this reading is based", "---")

READING_DISCLAIMER = (
    "\n\n---\n*This reading is based on the classical Samudrik Shastra "
    "(Jyotishacharya Bhriguraj, Pustak Mandir, Mathura). It reflects traditional "
    "beliefs and is offered for cultural and educational interest only. It is not "
    "medical, legal, financial or psychological advice, and no prediction in it is "
    "a certainty. Lines change with effort and time.*"
)




class PalmReader:
    """Presence-only vision, user-chosen figures, rules from those figures."""

    def __init__(self):
        if not FIGURE_INDEX_PATH.exists():
            print("[ERROR] Figure index not found. Run: python3 engine/build_figure_index.py")
            sys.exit(1)
        if not KNOWLEDGE_BASE_PATH.exists():
            print("[ERROR] Knowledge base not found. Run: python3 engine/extract_knowledge.py")
            sys.exit(1)
        self.figure_index = json.loads(FIGURE_INDEX_PATH.read_text(encoding="utf-8"))
        self.kb = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
        self._rule_by_id = self._build_rule_lookup()
        self.vector_collection = self._load_vector_db()

    def _build_rule_lookup(self):
        """Rebuilds the builder's rule ids: rule_<pdf_page>_<index>.

        build_figure_index.py enumerated each page's rules in knowledge-base
        order, so the same enumeration here resolves an id back to its rule.
        Every id is then checked against the observation prefix saved in the
        figure index: a mismatch means the knowledge base was regenerated
        after the index was built, so the id may point at a different rule.
        """
        by_page = {}
        for rule in self.kb.get("rules", []):
            if not isinstance(rule, dict):
                continue
            page = rule.get("pdf_page")
            if page is None:
                continue
            by_page.setdefault(page, []).append(rule)
        lookup = {}
        for page, rules in by_page.items():
            for index, rule in enumerate(rules):
                lookup["rule_%d_%d" % (page, index)] = rule

        self._stale_rule_ids = self._verify_rule_ids(lookup)
        return lookup

    def _verify_rule_ids(self, lookup):
        """Returns the ids whose rule no longer matches the saved text."""
        stale = set()
        for plate in self.figure_index:
            for fig_key in self.figure_index[plate]:
                entry = self.figure_index[plate][fig_key]
                for saved in entry.get("rules", []):
                    rule_id = saved.get("id")
                    stored = saved.get("obs") or ""
                    rule = lookup.get(rule_id)
                    if rule is None:
                        print("[ERROR] figure_index.json is out of date: "
                              "re-run build_figure_index.py (%s is missing "
                              "from the knowledge base)" % rule_id)
                        stale.add(rule_id)
                        continue
                    current = str(rule.get("observation") or "")
                    if not current.startswith(stored):
                        print("[ERROR] figure_index.json is out of date: "
                              "re-run build_figure_index.py (%s no longer "
                              "matches its saved observation)" % rule_id)
                        stale.add(rule_id)
        if stale:
            print("[ERROR] %d rule id(s) skipped. The figure index and the "
                  "knowledge base disagree." % len(stale))
        return stale

    def _load_vector_db(self):
        if not VECTOR_DB_PATH.exists():
            print("[ERROR] Vector DB not found. Run: python3 engine/build_vectordb.py")
            sys.exit(1)
        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    # ---------- step 1: vision presence ----------

    def analyze_image(self, image_path, hand_type="auto"):
        print("\n[1/4] Vision: reading line presence from %s" % image_path)
        image_file = Path(image_path)
        if not image_file.exists():
            print("   [ERROR] Image not found: %s" % image_path)
            return None

        image_bytes = image_file.read_bytes()
        try:
            from PIL import Image
            with Image.open(BytesIO(image_bytes)) as img:
                img.thumbnail((1024, 1024))
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
        except Exception:
            pass
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = VISION_ANALYSIS_PROMPT
        if hand_type != "auto":
            prompt += "\n\nNOTE: the user states this is a %s hand." % hand_type.upper()

        try:
            response = OLLAMA_CLIENT.chat(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": prompt, "images": [image_base64]}],
                options={"temperature": 0.2, "num_predict": 600, "num_ctx": 8192},
                format=VISION_SCHEMA,
            )
            structured = json.loads(response["message"]["content"])
        except Exception as error:
            print("   [ERROR] Vision analysis failed: %s" % error)
            return None

        lines = structured.get("lines", {})
        detected_hand = structured.get("hand_type", "unknown")
        resolved_hand = hand_type if hand_type != "auto" else detected_hand
        if not lines:
            print("   [ERROR] Vision returned no line results.")
            return None

        present = [l for l in MAJOR_LINES if lines.get(l) == "present"]
        absent = [l for l in MAJOR_LINES if lines.get(l) == "absent"]
        unclear = [l for l in MAJOR_LINES if lines.get(l) == "unclear"]
        print("   hand=%s | present=%s | absent=%s | unclear=%s"
              % (resolved_hand, present, absent, unclear))
        return {"hand_type": resolved_hand, "lines": lines}

    # ---------- step 2: figure choice ----------

    @staticmethod
    def figures_on_sheet(line):
        """Figure numbers drawn on the plate's contact sheet."""
        numbers = set()
        for path in HANDS_DIR.glob("%s_Hand_*.png" % LINE_PLATES[line]):
            tail = path.stem.rsplit("_", 1)[-1]
            if tail.isdigit():
                numbers.add(int(tail))
        return sorted(numbers)

    def choose_figures(self, lines):
        """Asks the user to pick a figure for each present/unclear line."""
        print("\n[2/4] Figure matching (contact sheets open in Preview)")
        chosen = {}
        for line in MAJOR_LINES:
            status = lines.get(line, "unclear")
            if status == "absent":
                # The plan requires the user to confirm an absence; the 7B
                # presence check has never been measured.
                try:
                    answer = input("    confirm %s is ABSENT? (y/n): "
                                   % line.replace("_", " ")).strip().lower()
                except EOFError:
                    answer = "n"
                if answer == "y":
                    chosen[line] = {"status": "absent", "figure": None}
                    print("    confirmed absent")
                    continue
                # Any other answer means the line is there after all, so
                # offer its figures as an unclear line.
                print("    absence not confirmed; treating %s as unclear" % line)
                status = "unclear"
            figures = self.figure_index.get(line, {})

            sheet = CONTACT_SHEETS_DIR / (LINE_PLATES[line] + ".png")
            print("\n--- %s (%s) ---" % (line.replace("_", " ").title(), status))
            if sheet.exists():
                try:
                    import subprocess
                    subprocess.run(["open", str(sheet)], check=False)
                    print("    opened %s" % sheet.name)
                except Exception as error:
                    print("    [WARN] could not open the sheet: %s" % error)
                    print("    open it manually: %s" % sheet)
            else:
                print("    [WARN] contact sheet missing: %s" % sheet)

            # Every figure drawn on the sheet is listed, including the ones
            # the book gives no rule text for, so the user sees why a number
            # is not offered. The count is the sentences the reading would
            # actually print, filters and drops included.
            numbers = self.figures_on_sheet(line)
            if not numbers:
                numbers = sorted(int(k) for k in figures)
            usable = []
            parts = []
            for number in numbers:
                entry = figures.get(str(number))
                sentences = []
                if entry:
                    sentences = self.figure_sentences(line, number, status)
                if sentences:
                    usable.append(str(number))
                    parts.append("%s (%d sentences)" % (number, len(sentences)))
                else:
                    parts.append("%s (no printable text)" % number)
            print("    figures: %s" % ", ".join(parts))
            labels = sorted({lab for k in figures for lab in figures[k].get("sub_labels", [])})
            if labels:
                print("    sub-labels: %s" % ", ".join(labels))

            for attempt in range(3):
                try:
                    answer = input("    closest figure number, or 'none' to skip: ").strip()
                except EOFError:
                    answer = "none"
                if not answer.isdigit() or int(answer) not in numbers:
                    print("    skipping %s" % line)
                    answer = None
                    break
                if answer not in usable:
                    print("    figure %s has no printable text; pick another "
                          "or 'none'" % answer)
                    continue
                chosen[line] = {"status": status, "figure": answer}
                print("    confirmed figure %s" % answer)
                break
        return chosen

    # ---------- step 3: rules for the chosen figures ----------

    @staticmethod
    def suppress_self_harm_rule(rule):
        """Replaces a self-harm prediction before it reaches the model.

        The rule keeps its id, feature and observation; only the prediction
        changes, so the reading can still cite it.
        """
        text = " ".join(str(rule.get(key) or "")
                        for key in ("observation", "prediction", "variation"))
        if not SELF_HARM_RE.search(text):
            return rule, False
        safe = dict(rule)
        safe["prediction"] = SELF_HARM_PREDICTION
        return safe, True

    @staticmethod
    def is_placeholder_prediction(rule):
        """True when the prediction says nothing ("not specified")."""
        prediction = re.sub(r"[^a-z ]+", " ",
                            str(rule.get("prediction") or "").lower())
        words = prediction.split()
        if not words:
            return False
        if not any(word in PLACEHOLDER_WORDS for word in words):
            return False
        content = [word for word in words
                   if word not in PLACEHOLDER_FILLER_WORDS]
        return len(content) <= 1

    @staticmethod
    def is_figure_reference(rule):
        """True when the observation only points at a place in a figure."""
        observation = str(rule.get("observation") or "")
        if not FIGURE_REF_RE.search(observation):
            return False
        words = re.findall(r"[a-z]{3,}",
                           FIGURE_REF_RE.sub(" ", observation).lower())
        descriptive = [word for word in words
                       if word not in FIGURE_POINTER_WORDS]
        return len(descriptive) < 8

    @staticmethod
    def is_absent_line_rule(rule):
        """True when the rule describes the line being absent."""
        return bool(ABSENT_LINE_RE.search(str(rule.get("observation") or "")))

    @staticmethod
    def rule_word_set(rule):
        """Words of 4+ letters in a rule's observation and prediction."""
        text = " ".join(str(rule.get(key) or "")
                        for key in ("observation", "prediction"))
        return set(re.findall(r"[a-z]{4,}", text.lower()))

    @classmethod
    def is_duplicate_rule(cls, rule, kept_rules):
        """True when a rule repeats one already kept for this figure."""
        words = cls.rule_word_set(rule)
        if not words:
            return False
        for other in kept_rules:
            other_words = cls.rule_word_set(other)
            if not other_words:
                continue
            union = words | other_words
            if len(words & other_words) / len(union) >= DUPLICATE_THRESHOLD:
                return True
        return False

    def usable_rules_for_figure(self, line, figure, status):
        """Rules for one figure after every filter and the dedupe.

        This is the single place that decides what the reader will use, so
        the picker's rule counts and the reading agree.
        """
        removed = {"placeholder": 0, "figure_reference": 0, "absent_line": 0,
                   "duplicate": 0}
        entry = self.figure_index.get(line, {}).get(str(figure))
        if not entry:
            return [], removed
        kept = []
        for rule_id in entry["rule_ids"]:
            if rule_id in self._stale_rule_ids:
                continue
            rule = self._rule_by_id.get(rule_id)
            if rule is None:
                continue
            if is_degenerate_rule(rule):
                continue
            if self.is_placeholder_prediction(rule):
                removed["placeholder"] += 1
                continue
            if self.is_figure_reference(rule):
                removed["figure_reference"] += 1
                continue
            if status != "absent" and self.is_absent_line_rule(rule):
                removed["absent_line"] += 1
                continue
            if self.is_duplicate_rule(rule, [other for _, other in kept]):
                removed["duplicate"] += 1
                continue
            kept.append((rule_id, rule))
        return kept, removed

    def rules_for_figures(self, chosen):
        """Collects the rules linked to each chosen figure.

        Every rule passes the filters and the dedupe in
        usable_rules_for_figure(); self-harm predictions are then replaced
        before they reach the model.
        """
        matched = []
        seen = set()
        totals = {"placeholder": 0, "figure_reference": 0, "absent_line": 0,
                  "duplicate": 0}
        for line, info in chosen.items():
            figure = info.get("figure")
            if not figure:
                continue
            usable, removed = self.usable_rules_for_figure(
                line, figure, info.get("status"))
            for key in totals:
                totals[key] += removed[key]
            for rule_id, rule in usable:
                if rule_id in seen:
                    continue
                seen.add(rule_id)
                tagged = dict(rule)
                tagged["rule_id"] = rule_id
                tagged["matched_observation"] = line
                tagged["matched_figure"] = "%s fig %s" % (line, figure)
                tagged, _ = self.suppress_self_harm_rule(tagged)
                matched.append(tagged)
        print("   rule filters: %d placeholder, %d figure reference, "
              "%d absent-line, %d duplicate removed"
              % (totals["placeholder"], totals["figure_reference"],
                 totals["absent_line"], totals["duplicate"]))
        return matched

    def rules_for_absent_lines(self, lines, chosen):
        """Filtered semantic search for lines the user confirmed absent."""
        matched = []
        for line in MAJOR_LINES:
            if lines.get(line) != "absent":
                continue
            if chosen.get(line, {}).get("status") != "absent":
                continue  # the user did not confirm the absence
            query = "%s absent not visible missing" % line.replace("_", " ")
            try:
                results = self.vector_collection.query(
                    query_texts=[query], n_results=3,
                    where={"canonical_feature": line},
                )
            except Exception:
                continue
            for rule_id, metadata, distance in zip(
                results["ids"][0], results["metadatas"][0], results["distances"][0]
            ):
                if distance > 0.7:
                    continue
                tagged = dict(metadata)
                tagged["rule_id"] = rule_id
                tagged["matched_observation"] = line
                tagged["matched_figure"] = "%s absent" % line
                tagged, _ = self.suppress_self_harm_rule(tagged)
                matched.append(tagged)
        return matched

    # ---------- step 4: synthesis ----------

    @staticmethod
    def lowercase_unless_proper(text):
        """Lowercase the first letter unless the word is a line or a mount."""
        if not text:
            return text
        first_word = re.sub(r"[^A-Za-z]", "", text.split()[0])
        if first_word in KEEP_CAPITAL_WORDS:
            return text
        return text[0].lower() + text[1:]

    @classmethod
    def clean_observation(cls, text):
        """An observation as it reads after "when "."""
        observation = str(text or "").strip().rstrip(".")
        for prefix in ("If ", "When "):
            if observation.startswith(prefix):
                observation = observation[len(prefix):]
                break
        return cls.lowercase_unless_proper(observation)

    def _load_clean_rules(self):
        """Cleaned condition/meaning pairs, when the clean-up has been run."""
        if not CLEAN_RULES_PATH.exists():
            return {}
        try:
            data = json.loads(CLEAN_RULES_PATH.read_text(encoding="utf-8"))
        except Exception as error:
            print("[WARN] could not read %s (%s); using the raw rules"
                  % (CLEAN_RULES_PATH.name, error))
            return {}
        lookup = {}
        conflicts = []
        for line in data:
            for figure in data[line]:
                for entry in data[line][figure]:
                    rule_id = entry.get("rule_id")
                    if not rule_id:
                        continue
                    existing = lookup.get(rule_id)
                    if (existing is not None
                            and existing.get("status") != entry.get("status")):
                        conflicts.append(rule_id)
                        continue
                    lookup[rule_id] = entry
        if conflicts:
            print("[ERROR] %s gives one rule id two different statuses: %s"
                  % (CLEAN_RULES_PATH.name, ", ".join(sorted(set(conflicts)))))
            print("        re-run engine/clean_figure_rules.py to merge them; "
                  "using the plain template for now")
            return {}
        return lookup

    def cleaned_rule(self, rule_id):
        """The cleaned entry for a rule id, or None."""
        if not hasattr(self, "_clean_rules"):
            self._clean_rules = self._load_clean_rules()
        return self._clean_rules.get(rule_id)

    @classmethod
    def cleaned_sentence(cls, entry, number):
        """One cleaned pair as a sentence: "The book says that when ..." """
        condition = str(entry.get("condition") or "").strip().rstrip(".")
        meaning = str(entry.get("meaning") or "").strip()
        parts = [part.strip().rstrip(".!?") for part in
                 re.split(r"(?<=[.!?])\s+", meaning) if part.strip()]
        if not condition or not parts:
            return ""
        # The condition follows "when", so it starts lowercase unless its
        # first word is a line or a mount ("when the Sun Line ...").
        condition = cls.lowercase_unless_proper(condition)
        # The clean-up writes the meaning as "the book says that ...", which
        # would repeat the opening of the sentence.
        for prefix in ("the book says that ", "the book says ",
                       "the book states that ", "the book states "):
            if parts[0].lower().startswith(prefix):
                parts[0] = parts[0][len(prefix):]
                break
        first = cls.lowercase_unless_proper(parts[0])
        sentences = ["The book says that when %s, %s [R%d]."
                     % (condition, first, number)]
        for extra in parts[1:]:
            sentences.append("%s [R%d]." % (extra, number))
        return " ".join(sentences)

    @classmethod
    def template_sentence(cls, rule, number):
        """One rule as sentences: "The book says that when ..., ... [Rn]." """
        observation = cls.clean_observation(rule.get("observation"))
        prediction = str(rule.get("prediction") or "").strip().rstrip(".")
        # The prediction follows a comma, so it starts lowercase unless the
        # first word is a line, mount or proper name ("The book says that
        # when ..., such individuals ...").
        first_word = ""
        if prediction:
            first_word = re.sub(r"[^A-Za-z]", "", prediction.split()[0])
        if prediction and first_word not in KEEP_CAPITAL_WORDS:
            prediction = prediction[0].lower() + prediction[1:]
        if prediction.lower().startswith(("indicates", "signifies", "denotes",
                                          "suggests", "means")):
            prediction = "this " + prediction

        # A prediction may hold several sentences; each one carries the
        # citation, so no sentence is left without a source.
        parts = [part.strip().rstrip(".!?") for part in
                 re.split(r"(?<=[.!?])\s+", prediction) if part.strip()]
        if not parts:
            return ""
        if observation:
            sentences = ["The book says that when %s, %s [R%d]."
                         % (observation, parts[0], number)]
        else:
            sentences = ["The book says that %s [R%d]."
                         % (parts[0], number)]
        for extra in parts[1:]:
            sentences.append("%s [R%d]." % (extra, number))
        return " ".join(sentences)

    @staticmethod
    def rule_words(*texts):
        """Words of 4+ letters in a rule's condition and meaning, or its
        observation and prediction."""
        text = " ".join(str(text or "") for text in texts)
        return set(re.findall(r"[a-z]{4,}", text.lower()))

    @staticmethod
    def is_repeated_sentence(words, earlier_word_sets):
        """True when a sentence says what an earlier one in the same section
        already said.

        Jaccard catches sentences of similar length that say the same thing;
        the containment test (overlap over the smaller set) also catches a
        short sentence that sits inside a longer one.
        """
        if not words:
            return False
        for earlier in earlier_word_sets:
            union = words | earlier
            if union and len(words & earlier) / len(union) >= DUPLICATE_THRESHOLD:
                return True
            smaller = min(len(words), len(earlier))
            if smaller and len(words & earlier) / smaller >= CONTAINMENT_THRESHOLD:
                return True
        return False

    def sentence_for_rule(self, rule, cleaned, number):
        """The sentence one rule prints, or (None, set()) if it prints none.

        Both the reading and the picker's sentence count go through here, so
        they can never disagree about what a rule can print.
        """
        if cleaned and cleaned.get("status") in ("cleaned", "from_passage"):
            sentence = self.cleaned_sentence(cleaned, number)
            if sentence:
                return sentence, self.rule_words(cleaned.get("condition"),
                                                 cleaned.get("meaning"))
        # An original prediction that is a fragment ("the individual
        # possessing such a line") makes no sentence. Only short verb-less
        # predictions are treated as fragments, so a missed verb cannot drop
        # a real one.
        prediction = str(rule.get("prediction") or "")
        word_count = len(prediction.split())
        if (word_count < 5
                or (word_count <= FRAGMENT_MAX_WORDS
                    and not has_finite_verb(prediction))):
            return None, set()
        # A prediction that only repeats the observation would print a
        # tautology ("when X, X"), so it makes no sentence either.
        prediction_words = self.rule_words(prediction)
        observation_words = self.rule_words(rule.get("observation"))
        if prediction_words and observation_words:
            shared = len(prediction_words & observation_words)
            if shared / len(prediction_words) >= RESTATEMENT_THRESHOLD:
                return None, set()
        sentence = self.template_sentence(rule, number)
        return sentence, self.rule_words(rule.get("observation"), prediction)

    def figure_sentences(self, line, figure, status):
        """The sentences one figure would print, in order.

        This walks the same path build_reading_from_rules walks: the usable
        rules, the cleaned entries, the fragment and restatement drops, the
        per-line cap and the repeat check.
        """
        usable, _ = self.usable_rules_for_figure(line, figure, status)
        printable = []
        for rule_id, rule in usable:
            cleaned = self.cleaned_rule(rule_id)
            if cleaned and cleaned.get("status") == "dropped":
                continue
            # The reading runs the self-harm pass over its rules first, so the
            # count has to look at the same text it will print.
            rule, _ = self.suppress_self_harm_rule(rule)
            printable.append((rule, cleaned))
        sentences = []
        kept_word_sets = []
        for rule, cleaned in printable[:MAX_RULES_PER_LINE]:
            sentence, words = self.sentence_for_rule(rule, cleaned, 0)
            if not sentence:
                continue
            if self.is_repeated_sentence(words, kept_word_sets):
                continue
            kept_word_sets.append(words)
            sentences.append(sentence)
        return sentences

    def drop_unprintable_figures(self, chosen):
        """Warns about scripted figures that print nothing, and skips them."""
        kept = {}
        skipped = []
        for line, info in chosen.items():
            figure = info.get("figure")
            if not figure:
                kept[line] = info
                continue
            if self.figure_sentences(line, figure, info.get("status")):
                kept[line] = info
                continue
            skipped.append("%s figure %s" % (line, figure))
        if skipped:
            print("   [WARN] no printable text for: %s; skipped"
                  % ", ".join(skipped))
        return kept

    def build_reading_from_rules(self, chosen, matched_rules, hand_type, gender):
        """Writes the reading straight from the rules, with no model call."""
        numbers = {}
        for number, rule in enumerate(matched_rules, 1):
            numbers[rule.get("rule_id")] = number

        title = str(hand_type or "unknown").upper()
        role = self.hand_role(hand_type, gender)
        note = ""
        if gender and hand_type in ("left", "right"):
            whose = "woman" if gender == "female" else "man"
            if role == "active":
                note = (" (a %s's %s hand, per the book's convention)"
                        % (whose, hand_type))
            else:
                note = (" (a %s's %s hand, the secondary hand by the book's "
                        "convention)" % (whose, hand_type))

        output = ["## Palm Reading Report",
                  "**Hand analyzed:** %s  **Hand role:** %s%s"
                  % (title, role, note)]

        missing = []
        empty = []
        for line in MAJOR_LINES:
            info = chosen.get(line) or {}
            if not info.get("figure") and info.get("status") != "absent":
                continue
            line_rules = [rule for rule in matched_rules
                          if rule.get("matched_observation") == line]
            # A rule the clean-up dropped produces no sentence, so it does not
            # count against the per-line cap.
            printable = []
            for rule in line_rules:
                cleaned = self.cleaned_rule(rule.get("rule_id"))
                if cleaned and cleaned.get("status") == "dropped":
                    continue
                printable.append((rule, cleaned))
            if not printable:
                missing.append((line, info.get("figure"), info.get("status")))
                print("      %s: 0 rules, section omitted" % line)
                continue
            shown = printable[:MAX_RULES_PER_LINE]
            print("      %s: %d rules, %d shown, %d left out"
                  % (line, len(printable), len(shown),
                     len(printable) - len(shown)))
            name = line.replace("_", " ").title()
            sentences = []
            kept_word_sets = []
            repeats = 0
            fragments = 0
            for rule, cleaned in shown:
                number = numbers.get(rule.get("rule_id"))
                if number is None:
                    continue
                sentence, words = self.sentence_for_rule(rule, cleaned, number)
                if not sentence:
                    fragments += 1
                    continue
                # A sentence that repeats an earlier one in this section adds
                # nothing, so it is dropped.
                if self.is_repeated_sentence(words, kept_word_sets):
                    repeats += 1
                    continue
                kept_word_sets.append(words)
                sentences.append(sentence)
            if repeats:
                print("      %s: %d repeated sentence(s) dropped"
                      % (line, repeats))
            if fragments:
                print("      %s: %d fragment sentence(s) dropped"
                      % (line, fragments))
            if not sentences:
                # Every sentence went, so the heading would dangle.
                empty.append((line, info.get("figure"), info.get("status")))
                print("      %s: every sentence was dropped, section omitted"
                      % line)
                continue
            if info.get("status") == "absent":
                output.append("### %s (absent)" % name)
            else:
                output.append("### %s (Figure %s)" % (name, info.get("figure")))
            output.append(" ".join(sentences))

        if missing:
            parts = []
            for line, figure, status in missing:
                name = line.replace("_", " ").title()
                if status == "absent":
                    parts.append("%s (absent)" % name)
                else:
                    parts.append("%s (figure %s)" % (name, figure))
            output.append("No rule text in the book for: %s" % ", ".join(parts))
        if empty:
            parts = ["%s (figure %s)" % (line.replace("_", " ").title(), figure)
                     for line, figure, status in empty]
            output.append("No printable rule text for: %s" % ", ".join(parts))
        return "\n\n".join(output)

    def polish_reading(self, template_text):
        """Rewrites the template for flow with the local model."""
        print("   polishing with %s" % TEXT_MODEL)
        response = OLLAMA_CLIENT.chat(
            model=TEXT_MODEL,
            messages=[{"role": "user",
                       "content": POLISH_PROMPT + template_text}],
            options={"temperature": 0.2, "num_predict": 3000, "num_ctx": 16384},
            think=False,
        )
        return response["message"]["content"]

    @staticmethod
    def hand_role(hand_type, gender):
        if not gender or hand_type in ("auto", "unknown"):
            return "unspecified"
        primary = "right" if gender == "male" else "left"
        return "active" if hand_type == primary else "secondary"

    @staticmethod
    def is_risky_sentence(sentence):
        """True when a sentence states a grave outcome as a certainty.

        Three ways in: a grave topic plus a certainty marker, a grave event
        stated as a plain fact ("death occurs", "meets his end"), or one of
        the direct grave phrases. A sentence with no grave word is never
        risky, and a protective phrase ("free from untimely death") vetoes
        the whole sentence, because that is good news.
        """
        lowered = sentence.lower()
        if any(phrase in lowered for phrase in PROTECTIVE_PHRASES):
            return False
        # The fact patterns are specific enough to stand on their own, and
        # they catch "loses his life" and "meets her end", which carry no
        # word from GRAVE_WORDS.
        if GRAVE_FACT_RE.search(sentence):
            return True
        if not any(word in lowered for word in GRAVE_WORDS):
            return False
        if any(marker in lowered for marker in CERTAINTY_MARKERS):
            return True
        if WILL_VERB_RE.search(sentence):
            return True
        return any(phrase in lowered for phrase in DIRECT_GRAVE_PHRASES)

    def suppress_self_harm_sentences(self, reading):
        """Replaces any self-harm sentence, whatever its hedging.

        This runs separately from the grave-outcome check, so "may commit
        suicide" is replaced just like "will die".
        """
        replaced = []

        def replace_sentence(match):
            sentence = match.group(0)
            if not SELF_HARM_RE.search(sentence):
                return sentence
            replaced.append(sentence.strip())
            leading = sentence[:len(sentence) - len(sentence.lstrip())]
            if not leading and match.start() > 0:
                leading = " "
            citations = " ".join(CITATION_RE.findall(sentence))
            if citations:
                return leading + citations + " " + SELF_HARM_PREDICTION
            return leading + SELF_HARM_PREDICTION

        cleaned = SENTENCE_RE.sub(replace_sentence, reading)
        return cleaned, replaced

    @staticmethod
    def content_stems(text):
        """Five-letter stems of the content words in a text.

        Words under four letters and stop words are dropped; five-letter
        stems let "originates" match "originating" without a stemmer.
        """
        words = re.findall(r"[a-z]{4,}", text.lower())
        return {word[:5] for word in words
                if word not in GROUNDING_STOP_WORDS}

    @staticmethod
    def split_sentences(paragraph):
        """Sentences of one paragraph, each with its trailing [Rn] tags."""
        sentences = []
        last_end = 0
        for match in SENTENCE_RE.finditer(paragraph):
            if match.start() > last_end:
                fragment = paragraph[last_end:match.start()].strip()
                if fragment:
                    sentences.append(fragment)
            sentences.append(match.group(0).strip())
            last_end = match.end()
        tail = paragraph[last_end:].strip()
        if tail:
            sentences.append(tail)
        return sentences

    def judge_sentence(self, sentence, rule_text_by_number):
        """Returns "kept", "uncited" or "weak" for one sentence."""
        # Sentences the system wrote are never judged or removed.
        if SAFE_SENTENCE in sentence or SELF_HARM_PREDICTION in sentence:
            return "kept"
        numbers = [int(number) for number in re.findall(r"\[R(\d+)\]", sentence)]
        if not numbers:
            return "uncited"
        cited_text = " ".join(rule_text_by_number.get(number, "")
                              for number in numbers)
        sentence_stems = self.content_stems(CITATION_RE.sub(" ", sentence))
        if not sentence_stems:
            return "kept"  # nothing to judge, so leave it alone
        rule_stems = self.content_stems(cited_text)
        share = len(sentence_stems & rule_stems) / len(sentence_stems)
        return "kept" if share >= GROUNDING_THRESHOLD else "weak"

    def check_grounding(self, reading, matched_rules, keep_unsupported=False):
        """Removes sentences that do not trace back to the cited rules."""
        rule_text_by_number = {}
        for number, rule in enumerate(matched_rules, 1):
            parts = [str(rule.get(key) or "")
                     for key in ("observation", "prediction")]
            # A sentence built from a cleaned pair draws on the cleaned words,
            # so those count as the rule's own text here too.
            cleaned = self.cleaned_rule(rule.get("rule_id"))
            if cleaned and cleaned.get("status") == "cleaned":
                parts.append(str(cleaned.get("condition") or ""))
                parts.append(str(cleaned.get("meaning") or ""))
            rule_text_by_number[number] = " ".join(parts)

        kept_lines = []
        stats = {"kept": 0, "uncited": 0, "weak": 0}
        removed = []
        for line in reading.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.lower().startswith(GROUNDING_SKIP_PREFIXES):
                kept_lines.append(line)
                continue
            kept_sentences = []
            for sentence in self.split_sentences(stripped):
                verdict = self.judge_sentence(sentence, rule_text_by_number)
                if verdict == "kept":
                    stats["kept"] += 1
                    kept_sentences.append(sentence)
                    continue
                stats[verdict] += 1
                removed.append((verdict, sentence))
                if keep_unsupported:
                    kept_sentences.append(sentence)
            if kept_sentences:
                kept_lines.append(" ".join(kept_sentences))
        return "\n".join(kept_lines), stats, removed

    @staticmethod
    def with_citations(sentence, citations):
        """A rewritten sentence with its [Rn] tags at the end."""
        sentence = sentence.rstrip()
        if sentence.endswith("."):
            sentence = sentence[:-1]
        if citations:
            return "%s %s." % (sentence, citations)
        return sentence + "."

    @staticmethod
    def has_medical_word(sentence):
        """True when the sentence names an illness or a body organ condition."""
        if MEDICAL_PHRASE_RE.search(sentence):
            return True
        lowered = sentence.lower()
        for word in MEDICAL_WORDS:
            if re.search(r"\b%s(?:s|es)?\b" % re.escape(word), lowered):
                return True
        return False

    def soften_medical_sentences(self, reading):
        """Rewrites certain medical claims; notes plain statements of illness."""
        rewritten = []
        noted = []

        def replace_sentence(match):
            sentence = match.group(0)
            if not self.has_medical_word(sentence):
                return sentence
            citations = " ".join(CITATION_RE.findall(sentence))
            lowered = sentence.lower()
            certain = (any(marker in lowered for marker in CERTAINTY_MARKERS)
                       or WILL_VERB_RE.search(sentence))
            if certain:
                rewritten.append(sentence.strip())
                # Keep the whitespace before the sentence, so the replacement
                # does not run into the previous one, and put the tags at the
                # end of the rewritten sentence.
                leading = sentence[:len(sentence) - len(sentence.lstrip())]
                if not leading and match.start() > 0:
                    leading = " "
                return leading + self.with_citations(MEDICAL_SENTENCE, citations)
            # A plain statement of illness keeps its wording and gains a note
            # before the citation.
            noted.append(sentence.strip())
            first = CITATION_RE.search(sentence)
            if first:
                head = sentence[:first.start()].rstrip()
                return "%s%s %s" % (head, MEDICAL_NOTE, sentence[first.start():])
            return sentence.rstrip() + MEDICAL_NOTE

        cleaned = SENTENCE_RE.sub(replace_sentence, reading)
        return cleaned, rewritten, noted

    def check_sensitivity(self, reading):
        """Returns the sentences that state a grave outcome as certain.

        The prompt asks the model not to do this, but a 7B model sometimes
        copies the book's wording, so the finished text is checked
        deterministically as well.
        """
        found = []
        for match in SENTENCE_RE.finditer(reading):
            if self.is_risky_sentence(match.group(0)):
                found.append(match.group(0).strip())
        return found

    def soften_reading(self, reading):
        """Replaces each risky sentence, keeping its [Rn] citations."""
        replaced = []

        def replace_sentence(match):
            sentence = match.group(0)
            if not self.is_risky_sentence(sentence):
                return sentence
            replaced.append(sentence.strip())
            # Keep the whitespace that came before the sentence, so the
            # inserted text never runs into the previous sentence
            # ("...raised. [R11] The book links..."), and put the tags at the
            # end of the rewritten sentence.
            leading = sentence[:len(sentence) - len(sentence.lstrip())]
            if not leading and match.start() > 0:
                leading = " "
            citations = " ".join(CITATION_RE.findall(sentence))
            return leading + self.with_citations(SAFE_SENTENCE, citations)

        softened = SENTENCE_RE.sub(replace_sentence, reading)
        return softened, replaced

    def verify_citations(self, reading, matched_rules):
        """Every [Rn] in the reading must be a retrieved rule."""
        cited = {int(n) for n in re.findall(r"\[R(\d+)\]", reading)}
        valid = set(range(1, len(matched_rules) + 1))
        unknown = sorted(cited - valid)
        if not cited:
            return [], "no citations found"
        if unknown:
            return unknown, "cited %d non-retrieved rule(s)" % len(unknown)
        return [], "all citations valid"

    # ---------- pipeline ----------

    def read_palm(self, image_path, hand_type="auto", gender=None,
                  interactive=True, figures_override=None, use_vision=True,
                  keep_unsupported=False, polish=False):
        overall_started = time.time()
        # Step 1: vision (only with --vision, and only when an image is given).
        if use_vision and image_path:
            vision = self.analyze_image(image_path, hand_type)
            if not vision:
                return None
            lines = vision["lines"]
            resolved_hand = vision["hand_type"]
        elif use_vision:
            print("\n[1/4] Vision skipped (no image given)")
            lines = {l: "present" for l in MAJOR_LINES}
            resolved_hand = hand_type if hand_type != "auto" else "unknown"
        else:
            # No vision: every line is unclear, so the user is asked about
            # each one, and the hand side comes from --hand.
            print("\n[1/4] Vision skipped (--vision not given); "
                  "all 7 lines are treated as unclear")
            lines = {l: "unclear" for l in MAJOR_LINES}
            resolved_hand = hand_type if hand_type != "auto" else "unknown"

        # Step 2: figure choice.
        if figures_override is not None:
            print("\n[2/4] Figures taken from --figures: %s" % figures_override)
            chosen = self.drop_unprintable_figures(figures_override)
        elif interactive:
            chosen = self.choose_figures(lines)
        else:
            chosen = {}
        if not chosen:
            print("   [ERROR] No figures chosen, nothing to read. Aborting.")
            return None

        # Step 3: rules.
        print("\n[3/4] Collecting rules for the chosen figures")
        matched_rules = self.rules_for_figures(chosen)
        matched_rules += self.rules_for_absent_lines(lines, chosen)
        print("   %d rules collected" % len(matched_rules))
        if not matched_rules:
            print("   [ERROR] No rules found for the chosen figures. Aborting.")
            return None

        # Step 4: reading, built from the rules unless --polish asks for a
        # rewrite pass.
        build_started = time.time()
        print("\n[4/4] Building the reading from the rules")
        reading = self.build_reading_from_rules(chosen, matched_rules,
                                                resolved_hand, gender)
        if not reading:
            print("   [ERROR] Could not build a reading. Aborting.")
            return None
        if polish:
            reading = self.polish_reading(reading)
            print("   reading polished in %.1fs" % (time.time() - build_started))
        else:
            print("   reading built in %.1fs (no model call)"
                  % (time.time() - build_started))

        unknown, status = self.verify_citations(reading, matched_rules)
        if status != "all citations valid":
            print("   [WARN] Citation check: %s %s" % (status, unknown))
        else:
            print("   citation check: %s" % status)

        # Self-harm is removed first, whatever the hedging, and the note is
        # added when either the rule filter or this check fired.
        rules_suppressed = any(rule.get("prediction") == SELF_HARM_PREDICTION
                               for rule in matched_rules)
        reading, suppressed_sentences = self.suppress_self_harm_sentences(reading)
        if rules_suppressed or suppressed_sentences:
            print("   [WARN] Self-harm content suppressed "
                  "(rules filtered: %s, sentences replaced: %d)"
                  % (rules_suppressed, len(suppressed_sentences)))

        # Medical claims first: a certain one is rewritten, a plain one gains
        # a note. Neither wording carries a grave word, so the next pass
        # leaves them alone.
        reading, medical_rewritten, medical_noted = self.soften_medical_sentences(reading)
        if medical_rewritten or medical_noted:
            print("   [WARN] Medical wording: %d sentence(s) rewritten, "
                  "%d sentence(s) noted"
                  % (len(medical_rewritten), len(medical_noted)))

        notes = []
        risky = self.check_sensitivity(reading)
        if risky:
            print("   [WARN] Sensitivity check: %d sentence(s) state a grave "
                  "outcome as certain" % len(risky))
            for sentence in risky:
                print("          %s" % sentence)
            reading, replaced = self.soften_reading(reading)
            print("   [WARN] Rewritten: %d sentence(s) replaced, citations kept"
                  % len(replaced))
            notes.append("*Note: where the classical text predicts severe "
                         "misfortune, that is the book's traditional wording, "
                         "not a certainty about any person.*")
        else:
            print("   sensitivity check: no sentence states a grave outcome "
                  "as certain")

        # Grounding only applies to the polished text: template sentences are
        # cut directly from the rules, so they cannot drift.
        if polish:
            reading, grounding, removed = self.check_grounding(
                reading, matched_rules, keep_unsupported)
            print("   grounding: %d sentences kept, %d uncited removed, "
                  "%d weak removed"
                  % (grounding["kept"], grounding["uncited"], grounding["weak"]))
            for verdict, sentence in removed:
                print("      [%s] %s" % (verdict.upper(), sentence))
        else:
            grounding = {"kept": 0, "uncited": 0, "weak": 0, "skipped": True}
            print("   grounding: skipped (template sentences come from rules)")

        if rules_suppressed or suppressed_sentences:
            notes.append(HELPLINE_NOTE)
        for note in notes:
            reading += "\n\n" + note

        reading = reading + READING_DISCLAIMER

        print("   total time: %.1fs" % (time.time() - overall_started))

        return {
            "reading": reading,
            "lines": lines,
            "chosen_figures": chosen,
            "matched_rules": matched_rules,
            "hand_type": resolved_hand,
            "gender": gender,
            "citation_status": status,
            "risky_phrases": risky,
            "grounding": grounding,
        }


def test_sensitivity_rewrite():
    """Checks the sentence-level rewrite on fixed inputs (no model needed)."""
    reader = PalmReader.__new__(PalmReader)
    # (text, should change, expected result when one is required)
    cases = [
        ("Widowhood is deemed inevitable when the marriage line ends in a star. [R2]",
         True, None),
        ("Such a person will die young. [R4]", True, None),
        ("The person meets an untimely death in a foreign land. [R1]", True, None),
        ("He is bound to die before forty. [R3]", True, None),
        ("Success in trade is guaranteed for this person. [R5]", False, None),
        ("He will surely become a famous scholar. [R6]", False, None),
        ("Death occurs at the age shown by the break. [R7]", True, None),
        ("He will surely live a long life, free from untimely death. [R8]",
         False, None),
        ("The Mt. of Saturn is raised. Such a person will die young. [R11]", True,
         "The Mt. of Saturn is raised. "
         + SAFE_SENTENCE.rstrip(".") + " [R11]."),
    ]
    print("Sensitivity rewrite test (one sentence at a time)")
    all_ok = True
    for text, should_change, expected in cases:
        risky = reader.is_risky_sentence(text)
        rewritten, replaced = reader.soften_reading(text)
        if should_change:
            # The tags sit at the end now, so the core sentence is checked
            # without its full stop.
            ok = (risky and rewritten != text
                  and SAFE_SENTENCE.rstrip(".") in rewritten)
        else:
            ok = not risky and rewritten == text
        if expected is not None:
            ok = ok and rewritten == expected
        all_ok = all_ok and ok
        print("  [%s] %s" % ("PASS" if ok else "FAIL", text))
        print("        risky=%s -> %s" % (risky, rewritten))
        if expected is not None and rewritten != expected:
            print("        expected: %s" % expected)

    # Self-harm is a separate filter: it fires whatever the hedging.
    self_harm_cases = [
        ("The person may commit suicide by jumping into water. [R2]", True),
        ("Such a person may take his own life. [R3]", True),
        ("He may harm himself in despair. [R4]", True),
        ("He jumps into water to save a drowning child. [R5]", False),
        ("He burns his body by setting it on fire. [R7]", True),
        ("He ends the play of his life by falling from a great height. [R8]", True),
        ("Such a person drowns himself in a well. [R1]", True),
        ("It indicates a tendency toward suicidal thoughts. [R9]", True),
        ("He falls from a great height and is injured. [R10]", False),
        ("The fire in his heart drives him to succeed. [R11]", False),
    ]
    print("\nSelf-harm suppression test")
    for text, should_replace in self_harm_cases:
        cleaned, replaced = reader.suppress_self_harm_sentences(text)
        if should_replace:
            ok = cleaned != text and SELF_HARM_PREDICTION in cleaned
            # the citation must survive the rewrite
            ok = ok and all(tag in cleaned for tag in CITATION_RE.findall(text))
        else:
            ok = cleaned == text
        all_ok = all_ok and ok
        print("  [%s] %s" % ("PASS" if ok else "FAIL", text))
        print("        replaced=%s -> %s" % (bool(replaced), cleaned))

    # Medical claims: a certain one is rewritten, a plain one is noted, and a
    # sentence with no medical word at all is left alone.
    medical_cases = [
        ("...the couple's life will be filled with strife and will culminate "
         "in widowhood [R23].", "rewritten"),
        ("...such an individual inevitably becomes a victim of heart diseases "
         "[R19].", "rewritten"),
        ("...such an individual is destined for death under these conditions "
         "[R4].", "rewritten"),
        ("...such an individual suffers from kidney disease [R21].", "noted"),
        ("He will become a famous scholar [R6].", "unchanged"),
        ("...the individual falls prey to severe illnesses [R18].", "noted"),
        ("The person suffers frequent fevers [R7].", "noted"),
    ]
    print("\nMedical wording test")
    for text, expectation in medical_cases:
        cleaned, medical_rewritten, medical_noted = \
            reader.soften_medical_sentences(text)
        cleaned, grave = reader.soften_reading(cleaned)
        if expectation == "rewritten":
            # the tags now sit at the end of the rewritten sentence
            ok = (bool(medical_rewritten) or bool(grave)) and (
                MEDICAL_SENTENCE.rstrip(".") in cleaned
                or SAFE_SENTENCE.rstrip(".") in cleaned)
        elif expectation == "noted":
            ok = (bool(medical_noted) and MEDICAL_NOTE in cleaned
                  and MEDICAL_SENTENCE.rstrip(".") not in cleaned
                  and SAFE_SENTENCE.rstrip(".") not in cleaned)
        else:
            ok = cleaned == text
        all_ok = all_ok and ok
        print("  [%s] %s" % ("PASS" if ok else "FAIL", text))
        print("        -> %s" % cleaned)

    print("Result: %s" % ("all %d cases correct"
                          % (len(cases) + len(self_harm_cases) + len(medical_cases))
                          if all_ok else "FAILURES PRESENT"))
    return all_ok


def audit_self_harm_rules():
    """Lists the knowledge-base rules the self-harm pattern matches.

    Run this after widening SELF_HARM_RE: it shows exactly which passages
    the filter would rewrite, so the list can be checked by hand.
    """
    knowledge_base = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    matches = []
    for rule in knowledge_base.get("rules", []):
        if not isinstance(rule, dict):
            continue
        text = " ".join(str(rule.get(key) or "")
                        for key in ("observation", "prediction", "variation"))
        if SELF_HARM_RE.search(text):
            matches.append((rule, text))

    print("Knowledge-base rules matched by SELF_HARM_RE: %d" % len(matches))
    for rule, text in matches:
        matched = SELF_HARM_RE.search(text)
        print("  printed p.%-4s (pdf %-4s) %-24s | %s  [matched: %s]"
              % (rule.get("source_page"), rule.get("pdf_page"),
                 str(rule.get("feature_name"))[:24], text[:100],
                 matched.group(0)))
    return len(matches)


def parse_figures_argument(text):
    """Parses "heart_line=3,fate_line=5" into a chosen-figures dict."""
    chosen = {}
    for part in text.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        line, figure = part.split("=", 1)
        line, figure = line.strip(), figure.strip()
        if line in MAJOR_LINES and figure.isdigit():
            chosen[line] = {"status": "present", "figure": figure}
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Samudrik Shastra Palm Reader (figure matcher)")
    parser.add_argument("image", nargs="?",
                        help="Path to palm image (optional when --figures is used)")
    parser.add_argument("--hand", choices=["left", "right", "auto"], default="auto")
    parser.add_argument("--gender", choices=["male", "female"], default=None)
    parser.add_argument("--no-interactive", dest="interactive",
                        action="store_false", default=True)
    parser.add_argument("--figures",
                        help="Scripted figure choices, e.g. heart_line=3,fate_line=5")
    parser.add_argument("--output", help="Save the reading to a file")
    parser.add_argument("--vision", action="store_true",
                        help="Use the local vision model to read line presence "
                             "(needs an image; without it every line is treated "
                             "as unclear)")
    parser.add_argument("--keep-unsupported", dest="keep_unsupported",
                        action="store_true",
                        help="Keep sentences that fail the grounding check "
                             "(debugging only)")
    parser.add_argument("--polish", action="store_true",
                        help="Let the local model rewrite the template reading "
                             "for flow, then run the grounding check")
    parser.add_argument("--test-sensitivity", dest="test_sensitivity",
                        action="store_true",
                        help="Run the sentence-rewrite test and exit")
    parser.add_argument("--audit-self-harm", dest="audit_self_harm",
                        action="store_true",
                        help="List the knowledge-base rules the self-harm "
                             "filter matches, then exit")
    args = parser.parse_args()

    if args.test_sensitivity:
        test_sensitivity_rewrite()
        return

    if args.audit_self_harm:
        audit_self_harm_rules()
        return

    if args.vision and not args.image:
        parser.error("--vision needs an image")

    if not args.image and not args.figures:
        parser.error("give an image, or --figures for a scripted run")

    override = parse_figures_argument(args.figures) if args.figures else None
    if args.figures and not override:
        parser.error("could not parse --figures; expected heart_line=3,fate_line=5")

    reader = PalmReader()
    result = reader.read_palm(args.image, args.hand, args.gender,
                              interactive=args.interactive,
                              figures_override=override,
                              use_vision=args.vision,
                              keep_unsupported=args.keep_unsupported,
                              polish=args.polish)
    if not result:
        print("[ERROR] Analysis failed.")
        return

    print("\n" + "=" * 60)
    print(result["reading"])
    print("=" * 60)
    print("\nChosen figures: %s" % ", ".join(
        "%s=%s" % (l, i["figure"]) for l, i in result["chosen_figures"].items()))
    print("Rules used: %d | citations: %s" % (
        len(result["matched_rules"]), result["citation_status"]))

    if args.output:
        Path(args.output).write_text(result["reading"], encoding="utf-8")
        print("Reading saved to %s" % args.output)


if __name__ == "__main__":
    main()
