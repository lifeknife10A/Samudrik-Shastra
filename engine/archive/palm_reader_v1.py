#!/usr/bin/env python3
"""
Phase 3: Samudrik Shastra Palm Reader Engine
The main AI engine that analyzes palm images using local Vision AI + RAG.

Pipeline:
1. Vision model (Qwen2.5-VL via Ollama) analyzes the palm image
2. Extracted observations are searched against the ChromaDB knowledge base
3. Local LLM synthesizes a structured reading from matched rules

100% offline. Zero API keys.
"""

import json
import base64
import sys
import argparse
import re
import subprocess
from pathlib import Path

try:
    import ollama
except ImportError:
    print("[ERROR] ollama package not installed. Run: pip3 install ollama")
    sys.exit(1)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("[ERROR] chromadb not installed. Run: pip3 install chromadb sentence-transformers")
    sys.exit(1)

try:
    from build_vectordb import canonical_feature
except ImportError:
    print("[ERROR] build_vectordb.py not found next to palm_reader.py")
    sys.exit(1)

VISION_MODEL = "qwen2.5vl:7b"
TEXT_MODEL = "qwen3:8b"
VECTOR_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "samudrik_shastra_rules"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ILLUSTRATIONS_DIR = Path(__file__).parent.parent / "illustrations" / "cropped_hands"

# Canonical categories that have 6-hand figure plates in the book, used by the
# visual matcher (Phase C item 5).
CANONICAL_LINE_PLATES = {
    "heart_line": "Heart_Line",
    "fate_line": "Fate_Line",
    "head_line": "Head_Line",
    "life_line": "Life_Line",
    "sun_line": "Sun_Line",
    "health_line": "Health_Line",
    "marriage_line": "Marriage_Line",
}

# Fixed disclaimer appended to every reading (Phase C item 4).
READING_DISCLAIMER = (
    "\n\n---\n*This reading is based on the classical Samudrik Shastra "
    "(Jyotishacharya Bhriguraj, Pustak Mandir, Mathura). It reflects traditional "
    "beliefs and is offered for cultural and educational interest only. It is not "
    "medical, legal, financial or psychological advice, and no prediction in it is "
    "a certainty. Lines change with effort and time.*"
)

FIGURE_DESCRIBE_PROMPT = """This is a palmistry diagram from the classical book
Samudrik Shastra. Describe the palm line shown in the diagram in one or two
sentences, focusing on its shape: where it starts, where it ends, its depth,
and any forks, branches, breaks, chains or islands. Do not add commentary."""

VISION_ANALYSIS_PROMPT = """You are an expert in Samudrik Shastra (Indian Palmistry) and hand analysis.
Analyze this palm image and report ONLY what you can actually see.

First, determine hand_type: "left" or "right" (look at thumb position). Use "unknown" if unsure.

Then describe the features you can see, starting with the clearest line.
For each observation give:
- feature_type: one of "line", "mount", "mark", "shape", "finger", "nail"
- feature_name: use one of these names when it fits:
  lines: "heart_line", "head_line", "life_line", "fate_line", "sun_line",
         "health_line", "marriage_line"
  mounts: "mount_of_jupiter", "mount_of_saturn", "mount_of_apollo",
          "mount_of_mercury", "mount_of_venus", "mount_of_moon", "mount_of_mars"
  other: "hand_shape", "thumb", "nails", or a mark such as "star_mark" or "island_mark"
  For any other clearly visible feature, use a short snake_case name.
- description: what you see (length, depth, curvature, where it starts and
  ends, breaks, forks, islands, fullness)
- confidence: "high" if clearly visible, "medium" if faint or partly hidden.

Rules:
- Include a feature ONLY if you can actually see it. Leave out anything you
  cannot see; do not list features just to say they are missing.
- It is fine to report only a few observations.
- Report each feature only once.
- Report no more than 12 observations."""

# JSON schema enforced via Ollama's format= parameter (guarantees valid JSON).
VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "hand_type": {"type": "string", "enum": ["left", "right", "unknown"]},
        "observations": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "feature_type": {"type": "string"},
                    "feature_name": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["feature_type", "feature_name", "description", "confidence"]
            }
        }
    },
    "required": ["hand_type", "observations"]
}

READING_SYNTHESIS_PROMPT = """You are a Samudrik Shastra (Indian Palmistry) expert consultant.
Below are observations of a palm and matching rules retrieved from the classical
book Samudrik Shastra by Jyotishacharya Bhriguraj.

GROUNDING RULES (strict):
1. Every statement you make MUST cite the rule it comes from, like this: [R3].
   You may cite several rules at once: [R1][R5].
2. You may ONLY use rules numbered in the KNOWLEDGE BASE MATCHES below.
   If no rule supports a topic, do not write that section at all.
3. Never invent predictions, statistics, or rules. Never paraphrase a rule into
   a different meaning. If the rules conflict, present both and say they differ.
4. Only include sections for which at least one matching rule exists.
   Do not pad the reading with generic advice.

SENSITIVITY RULES:
5. Some rules mention death, widowhood, illness or grave misfortune. Present
   these calmly and factually, never as certainty, never with dates or timing.
   Frame every prediction as a tendency described in a classical text.
6. Avoid fear-mongering language. Balance any difficult prediction with the
   book's own remedies or the note that lines change with effort (Purushartha).

HAND CONVENTION:
7. The classical convention: read a man's RIGHT hand (active, acquired traits)
   and a woman's LEFT hand. The reading below states which hand was analyzed
   and, if it is the person's secondary hand, notes that it reflects inherited
   traits and potential rather than present life.

Structure your response with ONLY the sections you have rules for, e.g.:
## Palm Reading Report
**Hand analyzed:** ...  **Hand role:** active/secondary
### (Section per supported topic, e.g. Life Line, Heart Line, Mounts)
...
---
[Concluding paragraph: one honest summary, citing rules]

---
KNOWLEDGE BASE MATCHES:
"""


class PalmReaderEngine:
    """The main palm reading engine that orchestrates the full pipeline."""

    def __init__(self):
        self.vector_collection = None

    def initialize(self):
        """Initialize Ollama connection and ChromaDB vector store."""
        print("=" * 60)
        print("Samudrik Shastra Palm Reader Engine")
        print("=" * 60)

        print("Connecting to Ollama...")
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]
            print(f"   Available models: {model_names}")
            if VISION_MODEL not in model_names:
                print(f"\n[ERROR] Vision model '{VISION_MODEL}' not found!")
                print(f"   Run: ollama pull {VISION_MODEL}")
                sys.exit(1)
            if TEXT_MODEL not in model_names:
                print(f"\n[ERROR] Text model '{TEXT_MODEL}' not found!")
                print(f"   Run: ollama pull {TEXT_MODEL}")
                sys.exit(1)
            print(f"   Vision model: {VISION_MODEL}")
            print(f"   Text model: {TEXT_MODEL}")
        except Exception as error:
            print(f"[ERROR] Cannot connect to Ollama: {error}")
            sys.exit(1)

        print("Loading vector store...")
        if not VECTOR_DB_PATH.exists():
            print(f"[ERROR] Vector store not found at: {VECTOR_DB_PATH}")
            print("   Run build_vectordb.py first.")
            sys.exit(1)
        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.vector_collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )
        print(f"   Loaded {self.vector_collection.count()} rules from knowledge base")

    def analyze_image(self, image_path, hand_type="auto"):
        """Step 1: Use vision model to analyze the palm image."""
        print(f"\nAnalyzing palm image: {image_path}")
        image_file = Path(image_path)
        if not image_file.exists():
            print(f"[ERROR] Image not found: {image_path}")
            return None

        with open(image_file, "rb") as file:
            image_bytes = file.read()

        # Shrink large photos so the vision call does not overflow its context.
        try:
            from PIL import Image
            from io import BytesIO
            with Image.open(BytesIO(image_bytes)) as img:
                img.thumbnail((1024, 1024))
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
        except Exception:
            pass  # keep original bytes if resize fails

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = VISION_ANALYSIS_PROMPT
        if hand_type != "auto":
            prompt += f"\n\nNOTE: The user has indicated this is a {hand_type.upper()} hand."

        print("   Running vision analysis (this may take 30-60 seconds)...")
        try:
            response = ollama.chat(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": prompt, "images": [image_base64]}],
                options={"temperature": 0.2, "num_predict": 3000, "num_ctx": 8192},
                format=VISION_SCHEMA
            )
        except Exception as error:
            print(f"   [ERROR] Vision analysis failed: {error}")
            print("   Cannot produce a grounded reading. Aborting.")
            return None
        raw_output = response["message"]["content"]
        try:
            structured = json.loads(raw_output)
        except json.JSONDecodeError:
            print("   [ERROR] Vision model returned invalid or truncated JSON.")
            print("   Cannot produce a grounded reading. Aborting.")
            return None

        observations = structured.get("observations", [])
        if not observations:
            print("   [ERROR] Vision model detected no features in the image.")
            print("   Cannot produce a grounded reading. Aborting.")
            return None

        # Defensive: drop repeated feature names the model may have emitted,
        # keeping the entry with the most informative description (visible
        # beats "not visible"; longer beats shorter).
        seen_names = {}
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            name = obs.get("feature_name", "")
            if name not in seen_names:
                seen_names[name] = obs
                continue
            current = seen_names[name]
            candidate = obs
            def is_visible(entry):
                return "not visible" not in entry.get("description", "").lower()
            if is_visible(candidate) != is_visible(current):
                if is_visible(candidate):
                    seen_names[name] = candidate
            elif len(candidate.get("description", "")) > len(current.get("description", "")):
                seen_names[name] = candidate
        observations = list(seen_names.values())

        detected_hand = structured.get("hand_type", "unknown")
        # Trust the user's explicit hand choice over the model's guess.
        resolved_hand = hand_type if hand_type != "auto" else detected_hand

        # Render a human-readable text version for the synthesis prompt / verbose output.
        analysis_text = self._render_observations(observations)
        print(f"   Vision analysis complete ({len(observations)} observations, hand={resolved_hand})")
        return {
            "hand_type": resolved_hand,
            "observations": observations,
            "analysis": analysis_text
        }

    def _render_observations(self, observations):
        """Renders structured observations into readable text."""
        if not observations:
            return "(no features detected)"
        lines = []
        for obs in observations:
            name = obs.get("feature_name", "unknown").replace("_", " ")
            desc = obs.get("description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def retrieve_matching_rules(self, observations, top_k=20):
        """Step 2: one query per observation, filtered by canonical feature.

        Each observation queries the store with a metadata filter, so a
        palm-line query cannot return foot- or face-reading rules. Slots are
        allocated per observation (3 each) instead of letting the first,
        most generic observation take the whole global top-k.
        """
        print(f"\nSearching knowledge base for matching rules...")
        per_observation = []
        seen_ids = set()

        for obs in observations:
            if not isinstance(obs, dict):
                continue
            description = obs.get("description", "")
            lowered = description.strip().lower().rstrip(".")
            if not description:
                continue
            if "not visible" in lowered or "not clearly visible" in lowered:
                continue
            # Only treat "unclear" as invisible when it describes the WHOLE
            # feature: "long and deep, start point unclear" is still visible.
            if lowered in ("cannot tell", "cannot see") or lowered.startswith("unclear"):
                continue
            confidence = obs.get("confidence", "medium")
            feature_name = obs.get("feature_name", "")
            query = f"{feature_name.replace('_', ' ')} {description}".strip()[:200]
            if len(query) < 5:
                continue

            canonical = canonical_feature(feature_name)
            if canonical == "other":
                # An unfiltered search would pull rules from unrelated
                # topics, so unmappable observations are skipped.
                print(f"   ⏭️ Skipping '{feature_name}' (no rule group for it)")
                continue
            query_args = {"query_texts": [query], "n_results": 3}
            query_args["where"] = {"canonical_feature": canonical}
            results = self.vector_collection.query(**query_args)

            if not results["ids"] or not results["ids"][0]:
                continue
            for doc_id, metadata, distance in zip(
                results["ids"][0], results["metadatas"][0], results["distances"][0]
            ):
                # Cosine distance: 0 = identical, 1 = unrelated. 0.7 drops only
                # clear junk; rank order does the real selection work.
                if doc_id in seen_ids or distance > 0.7:
                    continue
                seen_ids.add(doc_id)
                tagged = dict(metadata)
                tagged["search_distance"] = distance
                tagged["matched_observation"] = feature_name
                tagged["matched_query"] = query
                tagged["observation_confidence"] = confidence
                per_observation.append(tagged)

        # Allocate: best 3 per observation, then merge globally by distance.
        by_observation = {}
        for tagged in per_observation:
            by_observation.setdefault(tagged["matched_observation"], []).append(tagged)
        merged = []
        for obs_matches in by_observation.values():
            obs_matches.sort(key=lambda x: x.get("search_distance", 99))
            merged.extend(obs_matches[:3])
        merged.sort(key=lambda x: x.get("search_distance", 99))
        top_matches = merged[:top_k]
        print(f"   Found {len(top_matches)} matching rules across {len(by_observation)} observations")
        return top_matches

    def generate_reading(self, vision_analysis, matched_rules, hand_type, gender=None):
        """Step 3: Use LLM to synthesize the final reading, citing rule IDs."""
        print(f"\nGenerating palm reading...")
        rules_text = ""
        for i, rule in enumerate(matched_rules, 1):
            matched_obs = rule.get("matched_observation", "") or "general"
            rules_text += f"\n[R{i}] (matched observation: {matched_obs}):\n"
            rules_text += f"  Observation: {rule.get('observation') or 'N/A'}\n"
            rules_text += f"  Feature: {rule.get('feature_name') or 'N/A'}\n"
            rules_text += f"  Prediction: {rule.get('prediction') or 'N/A'}\n"
            rules_text += f"  Source: Page {rule.get('source_page') or 'N/A'}\n"

        role = self.hand_role(hand_type, gender)
        full_prompt = READING_SYNTHESIS_PROMPT + rules_text
        full_prompt += f"\n\n---\nVISUAL ANALYSIS OF THE PALM:\n{vision_analysis}"
        full_prompt += f"\n\nHAND FACTS: hand analyzed = {hand_type.upper()}, hand role = {role}"
        if gender:
            full_prompt += f", person's gender = {gender}"

        response = ollama.chat(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"temperature": 0.3, "num_predict": 3000, "num_ctx": 16384},
            think=False
        )
        reading = response["message"]["content"]
        print(f"   Reading generated ({len(reading)} chars)")
        return reading

    @staticmethod
    def hand_role(hand_type, gender):
        """Classical convention: men read the right hand, women the left."""
        if not gender or hand_type in ("auto", "unknown"):
            return "unspecified"
        primary = "right" if gender == "male" else "left"
        return "active" if hand_type == primary else "secondary"

    def verify_citations(self, reading, matched_rules):
        """Checks every [Rn] cited in the reading is a real retrieved rule."""
        cited = {int(number) for number in re.findall(r"\[R(\d+)\]", reading)}
        valid = set(range(1, len(matched_rules) + 1))
        unknown = cited - valid
        if not cited:
            return [], "no citations found"
        if unknown:
            return sorted(unknown), f"cited {len(unknown)} non-retrieved rule(s)"
        return [], "all citations valid"

    def describe_reference_figure(self, image_path):
        """Runs the vision model on one book figure to get a line description."""
        try:
            with open(image_path, "rb") as file:
                image_b64 = base64.b64encode(file.read()).decode("utf-8")
            response = ollama.chat(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": FIGURE_DESCRIBE_PROMPT,
                           "images": [image_b64]}],
                options={"temperature": 0.1, "num_predict": 300}
            )
            text = response["message"]["content"].strip()
            return text if text else None
        except Exception as error:
            print(f"   [WARN] Could not analyze figure {image_path}: {error}")
            return None

    def confirm_with_figures(self, observations):
        """Visual matcher: lets the user compare uncertain lines against the
        book's figure plates. Each confirmed figure is vision-analyzed and its
        description REPLACES the uncertain observation of that feature."""
        extra_observations = []
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            feature_name = obs.get("feature_name", "")
            confidence = obs.get("confidence", "")
            canonical = canonical_feature(feature_name)
            if canonical not in CANONICAL_LINE_PLATES:
                continue
            if confidence == "high":
                continue  # vision model is already sure
            prefix = CANONICAL_LINE_PLATES[canonical]
            # Sort by numeric grid position: Hand_2 before Hand_10.
            images = sorted(ILLUSTRATIONS_DIR.glob(f"{prefix}_Hand_*.png"),
                            key=lambda p: int(p.stem.rsplit("_", 1)[1]))
            if not images:
                continue

            label = canonical.replace("_", " ").title()
            print(f"\n📖 The vision model is not fully sure about your {label}.")
            print("   Opening the book's reference figures for comparison...")
            opened_any = False
            for image in images:
                try:
                    subprocess.run(["open", str(image)])
                    opened_any = True
                except Exception as error:
                    print(f"   ⚠️ Could not open images automatically ({error}).")
                    break
            if not opened_any:
                print("   Figures (open them manually):")
                for image in images:
                    number = int(image.stem.rsplit("_", 1)[1])
                    print(f"     {number}. {image}")

            try:
                answer = input(
                    f"   Which figure looks closest to YOUR palm? (1-{len(images)}, or Enter to skip): "
                ).strip()
            except EOFError:
                print("   (no input available; skipping)")
                continue
            if answer.isdigit():
                number = int(answer)
                matching = [p for p in images if p.stem.endswith(f"_{number}")]
                if not matching:
                    print(f"   ⚠️ Figure {number} does not exist. Skipping.")
                    continue
                chosen = matching[0]
                description = self.describe_reference_figure(chosen)
                if description:
                    print(f"   ✅ Confirmed figure {number}: {description[:80]}...")
                    extra_observations.append({
                        "feature_type": "line",
                        "feature_name": canonical,
                        "description": description,
                        "confidence": "high",
                        "source": f"user_confirmed_figure_{number}"
                    })
        return extra_observations

    def get_reference_images(self, matched_rules):
        """Returns paths to relevant reference images from the book."""
        reference_images = []
        seen_canonical = set()
        for rule in matched_rules:
            canonical = rule.get("canonical_feature", "")
            if canonical in CANONICAL_LINE_PLATES and canonical not in seen_canonical:
                seen_canonical.add(canonical)
                prefix = CANONICAL_LINE_PLATES[canonical]
                for img_path in sorted(
                    ILLUSTRATIONS_DIR.glob(f"{prefix}_Hand_*.png"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[1])
                ):
                    reference_images.append(str(img_path))
        return reference_images

    def read_palm(self, image_path, hand_type="auto", gender=None, interactive=True):
        """Main entry point: full pipeline from image to reading."""
        vision_result = self.analyze_image(image_path, hand_type)
        if not vision_result:
            return None
        observations = list(vision_result["observations"])
        resolved_hand = vision_result["hand_type"]

        # Visual matcher: confirmed figures REPLACE the uncertain observations
        # of the same feature. Skipped when stdin is not a terminal.
        if interactive and not sys.stdin.isatty():
            print("\n[INFO] Standard input is not a terminal: skipping the visual matcher.")
            interactive = False
        if interactive:
            confirmed = self.confirm_with_figures(observations)
            if confirmed:
                # Replace by canonical feature so "Heart Line" and "heart_line"
                # are treated as the same feature.
                confirmed_names = {canonical_feature(obs.get("feature_name", ""))
                                   for obs in confirmed}
                observations = [obs for obs in observations
                                if canonical_feature(obs.get("feature_name", ""))
                                not in confirmed_names]
                observations.extend(confirmed)

        matched_rules = self.retrieve_matching_rules(observations)
        if not matched_rules:
            print("\n[ERROR] No matching rules found in the knowledge base for the")
            print("   observed features. Cannot produce a grounded reading. Aborting.")
            return None

        # Render the FINAL observations (including user confirmations) so the
        # reading model sees what was actually used for retrieval.
        final_analysis = self._render_observations(observations)
        reading = self.generate_reading(
            final_analysis, matched_rules, resolved_hand, gender
        )
        unknown_citations, citation_status = self.verify_citations(reading, matched_rules)
        if citation_status != "all citations valid":
            print(f"   ⚠️ Citation check: {citation_status}"
                  + (f" -> {unknown_citations}" if unknown_citations else ""))
        reading = reading + READING_DISCLAIMER

        reference_images = self.get_reference_images(matched_rules)
        return {
            "reading": reading,
            "vision_analysis": vision_result["analysis"],
            "matched_rules": matched_rules,
            "reference_images": reference_images,
            "hand_type": resolved_hand,
            "gender": gender,
            "citation_status": citation_status
        }


def main():
    # Parse arguments first so --help works without Ollama / vector DB.
    parser = argparse.ArgumentParser(description="Samudrik Shastra Palm Reader")
    parser.add_argument("image", help="Path to palm image")
    parser.add_argument("--hand", choices=["left", "right", "auto"],
                        default="auto", help="Hand type (default: auto-detect)")
    parser.add_argument("--gender", choices=["male", "female"], default=None,
                        help="Person's gender (classical convention: men read the "
                             "right hand, women the left)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip the visual matcher questions")
    parser.add_argument("--output", help="Save reading to file")
    parser.add_argument("--verbose", action="store_true", help="Show full vision analysis")
    args = parser.parse_args()

    engine = PalmReaderEngine()
    engine.initialize()

    result = engine.read_palm(args.image, args.hand, args.gender,
                              interactive=not args.no_interactive)

    if result:
        print("\n" + "=" * 60)
        print(result["reading"])
        print("=" * 60)

        if result["reference_images"]:
            print(f"\nReference diagrams from the book:")
            for img in result["reference_images"][:6]:
                print(f"   {img}")

        if args.verbose:
            print("\n--- Raw Vision Analysis ---")
            print(result["vision_analysis"])

        if args.output:
            output_path = Path(args.output)
            output_path.write_text(result["reading"], encoding="utf-8")
            print(f"\nReading saved to: {output_path}")
    else:
        print("[ERROR] Analysis failed.")


if __name__ == "__main__":
    main()
