# सामुद्रिक शास्त्र (Samudrik Shastra)
### Ancient Indian Palmistry, Physiognomy & Line Analysis
**Author**: Jyotishacharya Bhriguraj  
**Publication**: Pustak Mandir, Mathura (Digital Library of India Archive: `2015.400540`)

---

## 📖 Overview

This repository contains the complete digitization, transcription, translation, and illustration extraction of the classical 328-page treatise ***Samudrik Shastra***. 

Every page was processed using **Gemini Flash Vision OCR**, translated into domain-aware English (incorporating standard palmistry nomenclature), and augmented with high-resolution extracted diagram plates and individual hand diagrams.

---

## 📂 Project Structure

All primary Markdown editions are kept at the root level for immediate reading:

```
Samudrik-Shastra/
│
├── README.md                           # This repository documentation
├── Samudrik_Shastra_English.md         # 🌟 Full English translation with embedded diagrams (328 pages)
├── Samudrik_Shastra_Hindi_OCR.md       # 🌟 Complete Hindi Devanagari OCR with embedded diagrams (328 pages)
├── Illustrations_Catalog.md            # 🌟 Visual catalog of all 30 plates and 60 cropped hand diagrams
│
├── pdf/                                # Original source document
│   └── 2015.400540.Samudirk-Shastr.pdf # Complete 328-page scanned book (Archive scan)
│
├── illustrations/                      # Extracted visual assets
│   ├── plates/                         # 30 high-resolution full diagram plates (250 DPI)
│   └── cropped_hands/                  # 60 individually cropped hand variations (Hand 1 to Hand 6)
│
├── pages_translated_english/           # Individual page translations (page_001_en.md - page_328_en.md)
├── pages_ocr_hindi/                    # Individual OCR transcripts (page_001.txt - page_328.txt)
│
└── scripts/                            # Processing pipeline tools
    ├── ocr_hindi_book.py               # Batch vision OCR script using Gemini Flash
    ├── translate_book.py               # Domain-aware Hindi-to-English translation script
    ├── extract_palm_diagrams.py        # PDF illustration extraction & 6-hand grid cropping script
    └── embed_illustrations_in_books.py # Injects diagram images into both Markdown editions
```

---

## 🌟 Key Editions

### 1. [English Edition (Samudrik_Shastra_English.md)](Samudrik_Shastra_English.md)
* Full translation of all 328 pages.
* Preserves classical commentary, verse metrics, and structured headings.
* Translates esoteric Indian palmistry concepts into standard Western/English equivalents (*Mount of Jupiter, Mount of Mars, Apollo/Sun Line, Saturn/Fate Line, Rascettes/Bracelets*).
* Features embedded illustration plates and cropped hand figures inline.

### 2. [Original Hindi OCR Edition (Samudrik_Shastra_Hindi_OCR.md)](Samudrik_Shastra_Hindi_OCR.md)
* Clean Unicode Devanagari text transcribed from the original letterpress book.
* Corrects broken conjuncts, faded matras, and scanning bleed.
* Includes all diagrams linked inline.

### 3. [Illustrations Catalog (Illustrations_Catalog.md)](Illustrations_Catalog.md)
* Visual mapping of every diagram plate in the book.
* Includes:
  * **Major Palm Lines**: Heart Line (Page 82), Fate Line (Page 85), Head Line (Page 89), Life Line (Page 93), Sun Line (Page 97), Health Line (Page 108), Marriage Line (Page 110).
  * **Master Hand Layouts**: Figures 1 to 18 (Pages 209–223).
  * **Auspicious Geometry**: Triangles, Stars, Circles, Crescents, Cones (Page 280).
  * **Foot Reading (*Padam Rekha*)**: Figures 2 through 10 (Pages 162–193).

---

## 🛠️ Reproduction & Scripts

All scripts are in the `scripts/` folder:

```bash
# 1. Run Vision OCR on the PDF:
python3 scripts/ocr_hindi_book.py --key "YOUR_GEMINI_API_KEY"

# 2. Run Hindi-to-English translation:
python3 scripts/translate_book.py --key "YOUR_GEMINI_API_KEY"

# 3. Extract and crop all diagrams from the PDF:
python3 scripts/extract_palm_diagrams.py

# 4. Embed diagrams into Markdown files:
python3 scripts/embed_illustrations_in_books.py
```

---

## 🔮 Reading Engine (`engine/`)

The engine turns the translated book into a grounded palm reading. It runs
fully offline (Ollama) and every statement it makes is tied to a printed page
of the book.

### How to get a reading

1. Run it on a photo of a palm (right hand is the book's convention for a man,
   left for a woman):

   ```bash
   python3 engine/palm_reader.py my_palm.jpg --vision --gender male
   ```

   `--gender female` uses the left hand. Without `--vision` the engine asks you
   about every line instead of looking at the photo.

2. The engine opens the book's own contact sheet for each line. Compare the
   line in the photo with the numbered figures and type the closest number. If
   none of them matches, type `none`. Figures that would print nothing are
   marked "(no printable text)" and cannot be picked.

3. The reading then prints one short section per line. Every sentence is what
   the book says about the figure you picked, and ends with `[Rn]` — the rule
   it came from, with the book page in the run log. Severe wording is framed as
   the book's view, and the reading ends with a disclaimer: this is a
   traditional text for cultural interest, not medical or other advice.

### How it works (figure matcher)

1. **Line presence is optional vision.** With `--vision`, a local vision model
   looks at the photo and answers only two things: which hand it is, and
   whether each of the seven major lines is present, absent or unclear. It
   never describes a line, so it cannot invent features, and an absence is
   confirmed by the user before it is used. Without `--vision` nothing is
   guessed: every line is treated as unclear and the user is asked about each
   one.
2. **The user picks the matching figure.** For each line the engine opens the
   book's own contact sheet and lists every figure drawn on it, with the number
   of rules that figure would actually print: "3 (2 rules)" or
   "4 (no printable text)". The count comes from the same path the reading
   takes, so it cannot disagree with it, and a figure with no printable text
   cannot be chosen. A scripted run can skip the picker with `--figures`; a
   figure named there that would print nothing is reported and skipped.
3. **Rules come from that figure's paragraph.** Each figure in
   `figure_index.json` carries the rules extracted from the paragraph that
   cites it, so no rule is borrowed from a neighbouring figure.
4. **The reading cites every rule.** Each sentence carries an `[Rn]` tag and
   the engine verifies that all citations resolve. Severe wording from the
   book is framed as the book's view and re-checked after generation.
5. **Safety rewrite, one sentence at a time.** A sentence is rewritten only
   when it names a grave topic (death, widowhood, untimely …) and states it as
   certain, or states the event outright. Its `[Rn]` citations are kept, at the
   end of the rewritten sentence, so the sentence becomes "The book links this
   sign with a serious danger; this is a traditional belief, not a certainty
   [R11]." A sentence with no grave word is never touched, so ordinary
   predictions stay as written. The same applies to medical wording: a certain
   claim is replaced, and a plain one keeps its sentence and gains
   "(a traditional belief, not a medical finding)" before its citation.
6. **Self-harm is filtered separately, whatever the hedging.** A rule or
   sentence that mentions suicide, self-harm or a method of it is replaced with
   "The book associates this sign with severe distress.", and the reading ends
   with a short helpline note. `--audit-self-harm` lists the book passages the
   filter matches.
7. **Rule ids are guarded.** `figure_index.json` stores each linked rule's id
   with the first 60 characters of its observation. If the knowledge base is
   regenerated and a rule no longer matches, the reader reports it and skips
   that rule instead of quoting the wrong text.

### Commands

```bash
# Reading without vision: every line is asked about
python3 engine/palm_reader.py palm.jpg --gender male

# Reading with vision (presence only; absent lines are confirmed by the user)
python3 engine/palm_reader.py palm.jpg --vision --hand right --gender male

# Scripted run: figures given by hand, no vision needed
python3 engine/palm_reader.py --figures heart_line=1,fate_line=5 --hand right

# Safety checks
python3 engine/palm_reader.py --test-sensitivity
python3 engine/palm_reader.py --audit-self-harm

# Rebuild the knowledge base, vector DB and figure index (in this order)
python3 engine/extract_knowledge.py
python3 engine/build_vectordb.py
python3 engine/build_figure_index.py

# Back-fill rules for figures whose paragraph yielded none (dry run first)
python3 engine/backfill_figure_rules.py
python3 engine/backfill_figure_rules.py --write

# One-time clean-up of the figure rules into condition / meaning pairs
# (reads the book's rules, writes figure_rules_clean.json + a review file)
python3 engine/clean_figure_rules.py

# List the entries a human should check by hand (needs_review.md)
python3 engine/clean_figure_rules.py --needs-review

# Re-run every check on the engine (fast, offline, no photo needed)
python3 engine/run_checks.py

# Vision presence benchmark over the book's hand figures
python3 engine/test_vision_hands.py --all
```

### Files

| File | Purpose |
|------|---------|
| `engine/knowledge_base.json` | Rules extracted from all 315 pages |
| `engine/knowledge_base.original_2279.json` | Untouched copy of the original extraction (never written by any script) |
| `engine/knowledge_base.backup.<timestamp>.json` | Per-run backup written by the back-fill script |
| `engine/chroma_db/` | Vector store used to search rules by meaning |
| `engine/figure_index.json` | 46 book figures with passages and linked rules |
| `engine/figure_rules_clean.json` | Cleaned condition / meaning pairs, used by the template reading |
| `engine/figure_rules_review.md` | Original vs cleaned rules, side by side, for checking |
| `engine/needs_review.md` | The entries a check cannot settle, with the verdict from the book text |
| `engine/palm_reader.py` | The reader (figure-matcher pipeline, optional vision) |
| `engine/run_checks.py` | Re-runs every check and prints PASS/FAIL per item |
| `engine/backfill_figure_rules.py` | Fills figures whose paragraph gave no rule |
| `engine/clean_figure_rules.py` | One-time clean-up of the figure rules into readable pairs |
| `engine/archive/` | Finished with: `status.py`, its state file, the pre-revalidation clean-up, `palm_reader_v1.py` |
| `engine/test_photos/` | Two sample palm photos, kept out of git |

`figure_index.json` and the reader must agree on rule IDs (`rule_<page>_<index>`).
Both enumerate each page's rules in knowledge-base order, so rebuild the index
whenever `knowledge_base.json` is regenerated. New rules are always appended to
the end of the knowledge base, which keeps existing ids valid, and the back-fill
script writes a timestamped backup before every write. The old fixed-name backup
is kept as `engine/knowledge_base.backup.legacy-2296.json`.

