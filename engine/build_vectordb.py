#!/usr/bin/env python3
"""
Phase 2: Build ChromaDB Vector Store from Samudrik Shastra Knowledge Base
Indexes all extracted rules into a local vector database for RAG retrieval.
100% offline. No API keys needed.
"""

import json
import sys
import re
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("[ERROR] chromadb not installed. Run: pip3 install chromadb sentence-transformers")
    sys.exit(1)

KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.json"
VECTOR_DB_PATH = Path(__file__).parent / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "samudrik_shastra_rules"


def load_knowledge_base() -> dict:
    """Loads the structured JSON knowledge base."""
    if not KNOWLEDGE_BASE_PATH.exists():
        print(f"[ERROR] Knowledge base not found at: {KNOWLEDGE_BASE_PATH}")
        print("Run extract_knowledge.py first.")
        sys.exit(1)

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_str(value, default=""):
    """Returns default if value is None, otherwise the string form.
    ChromaDB 1.5.9 crashes on None metadata values."""
    return default if value is None else str(value)


def clean_int(value, default=0):
    """Returns default if value is None or not numeric, otherwise an int."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Words that point at a real physical feature. A rule whose observation just
# repeats its prediction is dropped UNLESS its observation mentions one of
# these, because such rules still carry retrievable content.
PHYSICAL_ANCHOR = re.compile(
    r"(line|mark|mount|finger|nail|hand|palm|thumb|wrist|triangle|star|island|"
    r"square|cross|circle|crescent|cone|grille|chain|fork|break|crease|vein|"
    r"shape|sole|foot|feet|toe|heel|joint|knuckle|phalanx|ray|flexion)", re.I)


def is_degenerate_rule(rule) -> bool:
    """True when the extraction put the whole statement in BOTH the
    observation and the prediction, leaving no physical feature to search
    on. Detects exact repeats, containment and near-paraphrases."""
    observation = str(rule.get("observation", "")).strip()
    prediction = str(rule.get("prediction", "")).strip()
    if not observation or not prediction:
        return False
    obs_words = set(observation.lower().split())
    pred_words = set(prediction.lower().split())
    jaccard = (
        len(obs_words & pred_words) / len(obs_words | pred_words)
        if obs_words and pred_words
        else 0.0
    )
    repetitive = (
        jaccard > 0.75
        or observation.lower() in prediction.lower()
        or prediction.lower() in observation.lower()
    )
    feature_name = str(rule.get("feature_name", "")).strip()
    feature_has_anchor = bool(PHYSICAL_ANCHOR.search(feature_name))
    obs_has_anchor = bool(PHYSICAL_ANCHOR.search(observation))
    return repetitive and not (feature_has_anchor or obs_has_anchor)


def canonical_feature(feature_name: str) -> str:
    """Maps a free-form feature name to a canonical category.

    The extraction LLM may call the same feature different things
    ("line of affection" vs "heart line"). Retrieval uses this canonical
    value to filter queries, so rules about the feet or face cannot be
    returned for a palm-line query.

    Check ORDER matters:
    - names containing the WORD "line"/"lines" (or "rekha") are line rules
      and are checked BEFORE fingers and mounts, so
      "heart_line_ending_below_index_finger" is a heart-line rule and
      "life_line_ending_at_mount_of_venus" is a life-line rule. The word
      boundary stops "inclined", "decline" and "outline" from matching.
    - fingers come next, so "saturn finger" is not the fate line
    - mounts come before the planet-line checks, so "mount of sun" maps to
      the mount, not sun_line
    - foot comes BEFORE marks, so "star on foot" is a foot rule
    - "ear" matches only as a whole word, so "near the wrist" is not a face rule
    - "head_shape" is a hand-shape rule, not head_line
    """
    name = str(feature_name).lower()

    # True when "line"/"lines" appears as a whole word (not inside "inclined").
    has_line_word = (bool(re.search(r"(^|_)lines?(_|$)", name))
                     or bool(re.search(r"\blines?\b", name))
                     or "rekha" in name)

    # A NAMED line that merely mentions a foot as its endpoint stays a line
    # rule (e.g. "life_line_ending_at_foot_of_palm").
    named_line = bool(re.search(r"(life|heart|head|fate|sun|health|marriage|apollo)_line", name))

    # 1. Foot reading comes BEFORE line rules so "foot_lines" and
    #    "lines_of_the_feet" land in the foot group. Any non-letter acts as
    #    a boundary, so spaces and hyphens work too, while "solemn" and
    #    "bigfoot" still don't match.
    if (re.search(r"(^|[^a-z])(foot|feet|soles?)([^a-z]|$)", name)
            or "padam" in name or "पैर" in name) and not named_line:
        return "foot_reading"

    # 1b. Specific named palm features that the generic checks would miss.
    phrase = name.replace("_", " ")
    if ("palm" in phrase and
            ("color" in phrase or "colour" in phrase or "texture" in phrase)):
        return "palm_texture"
    if "girdle of venus" in phrase:
        return "girdle_of_venus"
    if "mars line" in phrase or "plane of mars" in phrase or "mars plain" in phrase:
        return "mount_of_mars"
    if "line of ambition" in phrase:
        return "line_of_ambition"
    if "line of wealth" in phrase:
        return "line_of_wealth"
    if "line of service" in phrase:
        return "line_of_service"
    if "planetary friendship" in phrase or "planetary enmity" in phrase:
        return "mounts"
    # Sacred signs: the book reads wheel, conch and shell marks on the hand
    # as well as on the foot (foot-qualified names were caught by step 1).
    if re.search(r"(^|[^a-z])(wheel|conch|shell)([^a-z]|$)", name):
        return "special_marks"
    if "sign rising" in phrase or "rising sign" in phrase:
        return "physiognomy"

    # 2. Line rules win over fingers and mounts (line chapters are the core).
    if has_line_word:
        if "life" in name or "jivan" in name or "jeevan" in name or "जीवन" in name:
            return "life_line"
        if "affection" in name or "heart" in name or "hriday" in name or "हृदय" in name:
            return "heart_line"
        if ("mental" in name or "mastak" in name or "mastishk" in name or "मस्तक" in name
                or ("head" in name and "shape" not in name and "forehead" not in name)):
            return "head_line"
        if "fate" in name or "destiny" in name or "saturn" in name or "bhagya" in name or "भाग्य" in name:
            return "fate_line"
        if "apollo" in name or "sun" in name or "surya" in name or "सूर्य" in name:
            return "sun_line"
        if "health" in name or "mercury" in name or "swasthya" in name or "स्वास्थ्य" in name:
            return "health_line"
        if "marriage" in name or "vivah" in name or "विवाह" in name:
            return "marriage_line"
        if "progeny" in name or "children" in name:
            return "progeny_lines"
        if "bracelet" in name or "rascette" in name or "manibandh" in name:
            return "bracelet_lines"
        return "line"

    # 3. Fingers (the book names them after planets).
    if "finger" in name or "phalange" in name:
        return "hand_shape"

    # 4. Mounts (must precede the planet-line checks).
    if "mount" in name:
        for planet in ("jupiter", "saturn", "apollo", "sun", "mercury",
                       "venus", "moon", "mars"):
            if planet in name:
                return "mount_of_apollo" if planet == "sun" else f"mount_of_{planet}"
        return "mounts"

    if "life" in name or "jivan" in name or "jeevan" in name or "जीवन" in name:
        return "life_line"
    if "affection" in name or "heart" in name or "hriday" in name or "हृदय" in name:
        return "heart_line"
    if ("mental" in name or "mastak" in name or "mastishk" in name or "मस्तक" in name
            or ("head" in name and "shape" not in name and "forehead" not in name)):
        return "head_line"
    if "fate" in name or "destiny" in name or "saturn" in name or "bhagya" in name or "भाग्य" in name:
        return "fate_line"
    if "apollo" in name or "sun" in name or "surya" in name or "सूर्य" in name:
        return "sun_line"
    if "health" in name or "mercury" in name or "swasthya" in name or "स्वास्थ्य" in name:
        return "health_line"
    if "marriage" in name or "vivah" in name or "विवाह" in name:
        return "marriage_line"
    if "progeny" in name or "children" in name:
        return "progeny_lines"
    if "bracelet" in name or "rascette" in name or "manibandh" in name:
        return "bracelet_lines"
    if any(mark in name for mark in ("island", "square", "triangle", "star",
                                     "cross", "circle", "grille", "crescent", "cone")):
        return "special_marks"
    if any(re.search(r"\b" + re.escape(part) + r"\b", name) for part in
           ("face", "forehead", "nose", "eye", "eyes", "lip", "lips",
            "body", "mole", "ear", "ears")):
        return "physiognomy"
    if any(part in name for part in ("hand", "finger", "thumb", "nail", "shape")):
        return "hand_shape"
    return "other"


def build_searchable_text(rule: dict) -> str:
    """Creates a rich searchable text block from a single rule."""
    if not isinstance(rule, dict):
        return ""
    parts = []
    if rule.get("observation"):
        parts.append(f"Observation: {rule['observation']}")
    if rule.get("feature_name"):
        parts.append(f"Feature: {str(rule['feature_name']).replace('_', ' ')}")
    if rule.get("variation"):
        parts.append(f"Variation: {rule['variation']}")
    if rule.get("prediction"):
        parts.append(f"Prediction: {rule['prediction']}")
    return " | ".join(parts)


def build_vector_store():
    """Main function: loads rules and indexes them into ChromaDB."""
    print("=" * 60)
    print("🔍 Building ChromaDB Vector Store")
    print("=" * 60)

    knowledge_base = load_knowledge_base()
    rules = knowledge_base.get("rules", [])
    print(f"📚 Loaded {len(rules)} rules from knowledge base")

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

    # Use sentence-transformers for local embeddings (no API key)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Delete existing collection if it exists (fresh build)
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection: {COLLECTION_NAME}")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        # Cosine distance: measures meaning similarity, unlike L2 which let
        # unrelated queries pass the old 1.5 threshold.
        metadata={"hnsw:space": "cosine", "description": "Samudrik Shastra palmistry rules"}
    )
    print(f"📦 Created collection: {COLLECTION_NAME} (cosine space)")

    # Prepare batch data
    ids = []
    documents = []
    metadatas = []
    skipped_front_matter = 0
    skipped_degenerate = 0

    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            print(f"  [WARN] Skipping non-dict rule at index {index}")
            continue
        # Front matter (title, publisher, editorial, contents) yields junk
        # "rules" with wrong citations (e.g. the TOC page cites printed 205).
        pdf_page = clean_int(rule.get("pdf_page"), 0)
        if 0 < pdf_page <= 15:
            skipped_front_matter += 1
            continue
        # Degenerate rules (observation just repeats the prediction) carry
        # nothing searchable unless the observation names a physical feature.
        if is_degenerate_rule(rule):
            skipped_degenerate += 1
            continue
        searchable_text = build_searchable_text(rule)
        if not searchable_text.strip():
            continue

        rule_id = f"rule_{index:04d}"
        ids.append(rule_id)
        documents.append(searchable_text)
        metadatas.append({
            "rule_id": rule_id,
            "feature_type": clean_str(rule.get("feature_type"), "general"),
            "feature_name": clean_str(rule.get("feature_name"), "unknown"),
            "canonical_feature": canonical_feature(rule.get("feature_name")),
            "source_page": clean_int(rule.get("source_page"), 0),
            "pdf_page": pdf_page,
            "source_topic": clean_str(rule.get("source_topic"), "general"),
            "confidence": clean_str(rule.get("confidence"), "medium"),
            "observation": clean_str(rule.get("observation")),
            "prediction": clean_str(rule.get("prediction")),
            "variation": clean_str(rule.get("variation")),
        })

    # Insert in batches of 100
    batch_size = 100
    total_inserted = 0

    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end]
        )
        total_inserted += end - start
        print(f"  📥 Inserted {total_inserted}/{len(ids)} rules...")

    if skipped_front_matter:
        print(f"  🗑️  Skipped {skipped_front_matter} front-matter rules (pdf pages 1-15)")
    if skipped_degenerate:
        print(f"  🗑️  Skipped {skipped_degenerate} degenerate rules (observation = prediction)")

    # Near-duplicate scan: double-scanned pages produce rules with slightly
    # different wording but almost identical meaning. In cosine space those
    # sit extremely close together, so one batch query finds them.
    dup_ids = set()
    if documents:
        print("  🔎 Scanning for near-duplicate rules...")
        dup_results = collection.query(query_texts=documents, n_results=2)
        for i, row in enumerate(dup_results["ids"]):
            if len(row) > 1 and dup_results["distances"][i][1] < 0.03:
                # Flag only the lexicographically LARGER id of the pair, and
                # keep the set unique. When A and B are each other's nearest
                # neighbours, both queries flag the same larger id, so exactly
                # one copy survives (fixes mutual deletion and duplicate ids).
                dup_ids.add(max(row[0], row[1]))
        if dup_ids:
            collection.delete(ids=list(dup_ids))
            print(f"  🧹 Removed {len(dup_ids)} near-duplicate rules")
        else:
            print("  ✅ No near-duplicates found")

    print(f"\n{'=' * 60}")
    print(f"🎉 Vector store built successfully!")
    print(f"📊 Total rules indexed: {total_inserted - len(dup_ids)}")
    print(f"📁 Stored at: {VECTOR_DB_PATH}")
    print(f"🧠 Embedding model: {EMBEDDING_MODEL}")
    print(f"{'=' * 60}")

    # Quick test search
    print("\n🧪 Test search: 'heart line is long and deep'")
    results = collection.query(query_texts=["heart line is long and deep"], n_results=3)
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"  [{i+1}] {doc[:120]}...")
        print(f"       Page {metadata['source_page']} | {metadata['feature_name']} | "
              f"distance {results['distances'][0][i]:.3f}")

    return collection


if __name__ == "__main__":
    build_vector_store()
