#!/usr/bin/env python3
"""
Phase 1: Knowledge Extraction from Samudrik Shastra
Parses the English Markdown edition and extracts structured palmistry rules
into a machine-readable JSON knowledge base.

Uses the local Ollama LLM to intelligently parse each section into
observation -> prediction rule pairs.
"""

import json
import re
import time
import sys
from pathlib import Path

try:
    import ollama
    OLLAMA_AVAILABLE = True
    # Dedicated client with a timeout: if Ollama dies mid-request, the call
    # raises (and the page is recorded as failed) instead of hanging the
    # whole extraction silently.
    OLLAMA_CLIENT = ollama.Client(timeout=300.0)
except ImportError:
    OLLAMA_AVAILABLE = False
    OLLAMA_CLIENT = None
    print("[WARN] ollama package not installed. Run: pip3 install ollama")

BOOK_PATH = Path(__file__).parent.parent / "Samudrik_Shastra_English.md"
OUTPUT_PATH = Path(__file__).parent / "knowledge_base.json"
CHECKPOINT_PATH = OUTPUT_PATH.with_suffix(".partial.json")
CHECKPOINT_VERSION = 1
TEXT_MODEL = "qwen3:8b"


def split_into_pages(content: str) -> list[dict]:
    """Splits the Markdown content into individual page entries."""
    pattern = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(content))
    pages = []
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        page_text = content[start:end].strip()
        # Remove ALL embedded image blocks (blockquote lines starting with >)
        page_text = re.sub(r"^>.*$", "", page_text, flags=re.MULTILINE)
        # Remove any remaining image markdown
        page_text = re.sub(r"!\[.*?\]\(.*?\)", "", page_text)
        # Remove the trailing Markdown separator that ends every page body.
        page_text = re.sub(r"\n---\s*$", "", page_text)
        # Clean up excessive blank lines
        page_text = re.sub(r"\n{3,}", "\n\n", page_text)
        page_text = page_text.strip()
        pages.append({"page": page_num, "text": page_text})
    return pages


def identify_chapters(pages: list[dict]) -> list[dict]:
    """Identifies chapter boundaries and assigns topic labels."""
    chapter_keywords = {
        "life_line": ["life line", "जीवन रेखा"],
        "heart_line": ["heart line", "हृदय रेखा", "line of affection"],
        "head_line": ["head line", "mental line", "मस्तक रेखा"],
        "fate_line": ["fate line", "saturn line", "भाग्य रेखा"],
        "sun_line": ["sun line", "apollo line", "सूर्य रेखा"],
        "health_line": ["health line", "mercury line", "स्वास्थ्य रेखा"],
        "marriage_line": ["marriage line", "विवाह रेखा"],
        "progeny_lines": ["progeny line", "children line"],
        "mounts": ["mount of jupiter", "mount of saturn", "mount of venus",
                   "mount of mercury", "mount of moon", "mount of mars"],
        "special_marks": ["island", "square", "triangle", "star", "cross", "circle"],
        "bracelet_lines": ["bracelet", "rascette", "manibandh"],
        "foot_reading": ["foot", "padam rekha", "पैर"],
        "physiognomy": ["face", "forehead", "nose", "eyes", "lips", "body"],
        "hand_shape": ["hand shape", "fingers", "thumb", "nails"],
    }
    chapters = []
    current_chapter = "general"
    for page in pages:
        text_lower = page["text"].lower()
        # Score every topic by total keyword hits; the highest-scoring topic
        # wins. Word boundaries stop "star" matching "start", "cross" matching
        # "across". This replaces the old first-match logic that let the first
        # keyword in dict order dominate.
        topic_scores = {}
        for topic, keywords in chapter_keywords.items():
            score = 0
            for keyword in keywords:
                score += len(re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", text_lower))
            if score > 0:
                topic_scores[topic] = score
        chapter_match = re.search(r"[Cc]hapter\s+(?:-\s*)?\w+", page["text"])
        if chapter_match and topic_scores:
            current_chapter = max(topic_scores, key=topic_scores.get)
        primary_topic = max(topic_scores, key=topic_scores.get) if topic_scores else current_chapter
        chapters.append({
            "page": page["page"],
            "topic": primary_topic,
            "detected_topics": sorted(topic_scores, key=topic_scores.get, reverse=True),
            "text": page["text"]
        })

    return chapters


def extract_printed_marker(text: str) -> int | None:
    """Extracts the printed page number from the top or bottom of a page body.

    Handles "(160)", "( 161 )", "[ 162 ]", bare "163" on its own line, and
    inline numbers like "Head Line 182". Bracketed markers are preferred,
    then inline trailing numbers, then bare numbers. Values below 5 are
    rejected to skip figure/verse labels like "1" and "2".
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return None
    candidates = lines[:3] + lines[-3:]
    # Pass 1: bracketed markers are the most reliable.
    for line in candidates:
        match = re.match(r"^[\(\[]\s*(\d{1,3})\s*[\)\]]$", line)
        if match:
            return int(match.group(1))
    # Pass 2: inline trailing numbers, e.g. "Head Line 182".
    for line in candidates:
        match = re.match(r"^.+?\s+(\d{2,3})$", line)
        if match:
            value = int(match.group(1))
            if value >= 5:
                return value
    # Pass 3: bare numbers on their own line. Require >= 50 because body
    # page numbers in this book start at 56, while figure labels like
    # "5" and "6" sit on their own lines everywhere.
    for line in candidates:
        if re.match(r"^\d{1,3}$", line):
            value = int(line)
            if value >= 50:
                return value
    return None


def trailing_unfinished_sentence(page_text: str) -> str:
    """Returns the unfinished sentence at the end of a page body, or "".

    A page ends mid-sentence when its last text does not end with
    sentence-ending punctuation. Returns only the text after the last
    sentence-ending character, with trailing page-marker lines removed.
    Returns "" when the page ends with punctuation (complete sentence).
    """
    body = page_text.strip()
    if not body:
        return ""
    last_index = -1
    for ending in (".", "?", "!", "।"):
        index = body.rfind(ending)
        if index > last_index:
            last_index = index
    if last_index == -1:
        remainder = body
    else:
        remainder = body[last_index + 1:].strip()
        if not remainder:
            return ""  # page ends with punctuation: the sentence is complete
    # Drop trailing page-marker lines like "( 5 )" or bare "167".
    lines = [line.strip() for line in remainder.split("\n") if line.strip()]
    while lines and re.match(r"^[\(\[]?\s*\d{1,3}\s*[\)\]]?$", lines[-1]):
        lines.pop()
    remainder = " ".join(lines).strip()
    # A lone quote, bracket or figure label is not an unfinished sentence.
    if not re.search(r"[a-zA-Z]{2,}", remainder):
        return ""
    return remainder[-300:]


def build_printed_page_map(pages: list[dict]) -> dict:
    """Maps pdf_page -> printed page number.

    The number printed on each scan IS the physical page number, so the
    marker is trusted directly. Missing markers are interpolated from the
    previous page. Duplicate printed numbers across PDF pages are expected
    (double scans and jumbled scan order) and kept as-is: every rule cites
    the physical page it actually came from.
    """
    printed_map = {}
    prev_printed = None
    for page in pages:
        pdf_num = page["page"]
        marker = extract_printed_marker(page["text"])
        if marker is None:
            if prev_printed is None:
                printed = pdf_num  # first page has no marker: fall back to PDF number
                source = "fallback"
            else:
                printed = prev_printed + 1
                source = "interpolated"
        else:
            printed = marker
            source = "marker"
        printed_map[pdf_num] = {"printed": printed, "source": source}
        prev_printed = printed
    return printed_map


EXTRACTION_PROMPT = """You are a knowledge extraction engine for Samudrik Shastra (Indian Palmistry).
Given a passage from the book, extract ALL palmistry rules.

Each rule must have:
- "observation": What visual feature is described
- "feature_type": one of: "line", "mount", "mark", "shape", "general"
- "feature_name": The specific name in snake_case (e.g., "heart_line", "mount_of_jupiter", "island_mark")
- "variation": The specific variation described
- "prediction": What this indicates about the person
- "confidence": "high" if clearly stated, "medium" if implied, "low" if speculative

Return ONLY a JSON object containing a "rules" array, matching the required schema.
If no rules can be extracted, return an empty rules array.

The passage may contain a "[PAGE START]" marker. Text BEFORE that marker is
the unfinished ending of the previous page. Use it only to complete a rule
that continues on this page. Do not extract rules from it alone.

Passage to analyze:
"""

# JSON schema enforced via Ollama's format= parameter (guarantees valid JSON).
RULES_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "observation": {"type": "string"},
                    "feature_type": {"type": "string",
                                     "enum": ["line", "mount", "mark", "shape", "general"]},
                    "feature_name": {"type": "string"},
                    "variation": {"type": "string"},
                    "prediction": {"type": "string"},
                    "confidence": {"type": "string",
                                   "enum": ["high", "medium", "low"]}
                },
                "required": ["observation", "feature_type", "feature_name",
                             "variation", "prediction", "confidence"]
            }
        }
    },
    "required": ["rules"]
}


def extract_rules_with_llm(text: str) -> list[dict]:
    """Uses the local Ollama LLM to extract structured rules from a text passage."""
    if not text.strip() or len(text.strip()) < 50:
        return []
    if len(text) > 3000:
        text = text[:3000]
    response = OLLAMA_CLIENT.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT + text}],
        options={"temperature": 0.1, "num_predict": 4096},
        think=False,
        format=RULES_SCHEMA
    )
    raw_output = response["message"]["content"].strip()
    raw_output = re.sub(r"```json\s*", "", raw_output)
    raw_output = re.sub(r"```\s*", "", raw_output)
    raw_output = raw_output.strip()
    data = json.loads(raw_output)  # raises JSONDecodeError; caller decides what to do
    if not isinstance(data, dict) or "rules" not in data:
        raise json.JSONDecodeError("Expected object with 'rules' key", raw_output, 0)
    rules = data.get("rules", [])
    # Schema should guarantee dicts, but never trust the model completely.
    return [rule for rule in rules if isinstance(rule, dict)]


def rule_key(rule: dict) -> str:
    """Normalised text used to spot duplicate rules from double-scanned pages."""
    parts = [rule.get("observation", ""), rule.get("prediction", "")]
    joined = " ".join(str(part) for part in parts).lower()
    joined = re.sub(r"[^a-z0-9 ]+", " ", joined)
    return " ".join(joined.split())


def deduplicate_rules(rules: list[dict]) -> list[dict]:
    """Drops rules whose normalised text repeats, keeping the first occurrence."""
    seen = set()
    unique = []
    for rule in rules:
        key = rule_key(rule)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(rule)
    return unique


def save_checkpoint(rules: list, processed_pages: set, failed_pages: list):
    """Writes partial progress atomically (tmp + rename) so a crash never
    loses the whole run and a kill mid-write cannot corrupt the file."""
    partial = {
        "version": CHECKPOINT_VERSION,
        "metadata": {"partial": True},
        "rules": rules,
        "processed_pages": sorted(processed_pages),
        "failed_pages": failed_pages
    }
    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(CHECKPOINT_PATH)


def build_knowledge_base() -> dict:
    """Main function: reads the book, extracts rules, builds the knowledge base."""
    if not BOOK_PATH.exists():
        print(f"[ERROR] Book not found at: {BOOK_PATH}")
        sys.exit(1)

    content = BOOK_PATH.read_text(encoding="utf-8")
    print(f"📖 Read book: {len(content)} characters")

    pages = split_into_pages(content)
    print(f"📄 Split into {len(pages)} pages")

    printed_map = build_printed_page_map(pages)
    map_sources = {}
    for entry in printed_map.values():
        map_sources[entry["source"]] = map_sources.get(entry["source"], 0) + 1
    print(f"🔢 Printed page map: {map_sources}")

    chapters = identify_chapters(pages)

    topic_pages = {}
    for chapter in chapters:
        topic = chapter["topic"]
        if topic not in topic_pages:
            topic_pages[topic] = []
        topic_pages[topic].append(chapter)
    print(f"📚 Identified {len(topic_pages)} topics: {list(topic_pages.keys())}")

    # Resume support: load partial checkpoint if present.
    all_rules = []
    processed_pages = set()
    failed_pages = []
    if CHECKPOINT_PATH.exists():
        try:
            partial = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            if partial.get("version") != CHECKPOINT_VERSION:
                print("[WARN] Checkpoint has an unknown format (version "
                      f"{partial.get('version')}); ignoring it and starting fresh.")
            else:
                all_rules = partial.get("rules", [])
                processed_pages = set(partial.get("processed_pages", []))
                failed_pages = partial.get("failed_pages", [])
                print(f"♻️ Resuming from checkpoint: {len(all_rules)} rules, "
                      f"{len(processed_pages)} pages already done")
        except Exception as error:
            print(f"[WARN] Could not read checkpoint ({error}); starting fresh.")

    total_pages = len(chapters)
    previous_page_num = None
    previous_unfinished = ""

    for index, chapter in enumerate(chapters):
        page_num = chapter["page"]
        topic = chapter["topic"]
        text = chapter["text"]

        # Carry the previous page's unfinished sentence ONLY when the printed
        # page numbers prove the two scans are back to back.
        carry_text = ""
        if previous_page_num is not None:
            prev_entry = printed_map.get(previous_page_num, {})
            curr_entry = printed_map.get(page_num, {})
            back_to_back = (
                prev_entry.get("source") == "marker"
                and curr_entry.get("source") == "marker"
                and curr_entry.get("printed", 0) == prev_entry.get("printed", 0) + 1
            )
            if back_to_back and previous_unfinished:
                carry_text = previous_unfinished

        previous_page_num = page_num
        previous_unfinished = trailing_unfinished_sentence(text)

        if not text.strip() or len(text.strip()) < 50:
            continue
        if page_num in processed_pages:
            continue

        print(f"  [{index + 1}/{total_pages}] Page {page_num} ({topic})...", end="", flush=True)

        extract_text = f"{carry_text}\n\n[PAGE START]\n\n{text}" if carry_text else text

        try:
            rules = extract_rules_with_llm(extract_text)
        except Exception as error:
            print(f" ❌ FAILED: {error}")
            if page_num not in failed_pages:
                failed_pages.append(page_num)
            # Page stays unprocessed, so a resumed run retries it.
            save_checkpoint(all_rules, processed_pages, failed_pages)
            continue
        time.sleep(0.3)

        for rule in rules:
            if not isinstance(rule, dict):
                print(f"    [WARN] Skipping non-dict rule on page {page_num}")
                continue
            rule["source_page"] = printed_map.get(page_num, {}).get("printed", page_num)
            rule["pdf_page"] = page_num
            rule["source_topic"] = topic

        all_rules.extend(rules)
        processed_pages.add(page_num)
        if page_num in failed_pages:
            failed_pages.remove(page_num)  # retried successfully this run
        print(f" ✅ {len(rules)} rules")

        # Checkpoint after every page so a crash never loses the run.
        save_checkpoint(all_rules, processed_pages, failed_pages)

    # Remove exact-duplicate rules produced by double-scanned pages.
    before_dedupe = len(all_rules)
    all_rules = deduplicate_rules(all_rules)
    if before_dedupe != len(all_rules):
        print(f"\n🧹 Removed {before_dedupe - len(all_rules)} duplicate rules")

    knowledge_base = {
        "metadata": {
            "title": "Samudrik Shastra - Structured Knowledge Base",
            "source": "Samudrik_Shastra_English.md",
            "author": "Jyotishacharya Bhriguraj",
            "total_rules": len(all_rules),
            "total_pages_processed": len(processed_pages),
            "extraction_method": "llm",
            "failed_pages": failed_pages
        },
        "page_map": printed_map,
        "topics": {},
        "rules": all_rules
    }

    for rule in all_rules:
        topic = rule.get("source_topic", "general")
        if topic not in knowledge_base["topics"]:
            knowledge_base["topics"][topic] = []
        knowledge_base["topics"][topic].append(rule)

    print("\n📊 Topic Summary:")
    for topic, topic_rules in knowledge_base["topics"].items():
        print(f"  📌 {topic}: {len(topic_rules)} rules")

    if failed_pages:
        print(f"\n⚠️ Pages that failed (will be retried on resume): {failed_pages}")

    return knowledge_base


def main():
    print("=" * 60)
    print("📚 Samudrik Shastra Knowledge Extraction")
    print("=" * 60)

    # The regex fallback produced much weaker rules, so we no longer silently
    # degrade: extraction requires the local LLM.
    if not OLLAMA_AVAILABLE:
        print("[ERROR] The ollama package is required. Run: pip3 install ollama")
        sys.exit(1)
    try:
        response = OLLAMA_CLIENT.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": "Hello"}],
                    options={"num_predict": 10}, think=False)
        if not response.message.content:
            raise ValueError("Empty response - model may need think=False")
        print(f"🤖 Using LLM: {TEXT_MODEL}")
    except Exception as error:
        print(f"[ERROR] Ollama is not responding: {error}")
        print("Start it with: ollama serve, then run this script again.")
        sys.exit(1)

    knowledge_base = build_knowledge_base()

    OUTPUT_PATH.write_text(json.dumps(knowledge_base, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = knowledge_base["metadata"].get("failed_pages", [])
    if failed:
        # Keep the checkpoint so a resumed run retries the failed pages.
        print(f"⚠️ {len(failed)} page(s) failed: {failed}")
        print("   The checkpoint was kept; run this script again to retry them.")
    else:
        CHECKPOINT_PATH.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print(f"🎉 Knowledge base saved to: {OUTPUT_PATH}")
    print(f"📊 Total rules extracted: {knowledge_base['metadata']['total_rules']}")
    print(f"📚 Topics covered: {len(knowledge_base['topics'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()

