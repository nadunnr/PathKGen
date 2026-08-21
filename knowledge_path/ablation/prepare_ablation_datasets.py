"""
prepare_ablation_datasets.py
============================
Generates ablation variant JSONs for the E1 experiment.

Reads WSI-Bench-{train,test}_Report_knowledge.json and emits four variants
into <DATASET_DIR>/ablation_kg/ by rewriting the `knowledge` field per sample:

  Condition    knowledge_mode   knowledge field written
  ──────────────────────────────────────────────────────────────────────
  full         full             original unchanged
  class_only   class_only       <knowledge>\nHistology: X\n</knowledge>
  kg_only      kg_only          <knowledge>\n<site + feature lines>\n</knowledge>
  site_only    site_only        <knowledge>\nfound in: Y\n</knowledge>

Usage:
    python prepare_ablation_datasets.py [--dry-run]
"""

import json, os, re, argparse, sys

# ---------------------------------------------------------------------------
# Paths  (resolve relative to repo root regardless of invocation dir)
# ---------------------------------------------------------------------------
REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE   = os.path.join(REPO_ROOT, "pipeline")
sys.path.insert(0, PIPELINE)
sys.path.insert(0, REPO_ROOT)

try:
    from paths_config import DATASET_DIR
    DATASET_DIR = os.path.abspath(DATASET_DIR)
except ImportError:
    DATASET_DIR = os.path.join(REPO_ROOT, "data", "dataset")

OUT_DIR = os.path.join(DATASET_DIR, "ablation_kg")

SPLITS = {
    "train": os.path.join(DATASET_DIR, "WSI-Bench-train_Report_knowledge.json"),
    "test":  os.path.join(DATASET_DIR, "WSI-Bench-test_Report_knowledge.json"),
}

# ---------------------------------------------------------------------------
# Parsing the <knowledge> block
# ---------------------------------------------------------------------------

def parse_knowledge(raw: str) -> dict:
    """
    Decompose the <knowledge>…</knowledge> block into named parts:
      histology  – value of the "Histology: …" line
      site       – full "found in: …" line (verbatim)
      features   – every other line ("has * :")
    """
    inner = re.sub(r"</?knowledge>", "", raw).strip()
    histology, site, features = None, None, []

    for line in inner.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Histology:"):
            histology = line[len("Histology:"):].strip()
        elif line.startswith("found in:"):
            site = line          # keep full "found in: Bronchus and lung"
        else:
            features.append(line)

    return {"histology": histology, "site": site, "features": features}


def build_knowledge(lines: list) -> str:
    """Wrap a list of text lines into a <knowledge> block."""
    return "<knowledge>\n" + "\n".join(lines) + "\n</knowledge>"


# ---------------------------------------------------------------------------
# Per-condition transformers
# ---------------------------------------------------------------------------

def tf_full(raw, p):
    """(a/d) full — unchanged"""
    return raw

def tf_class_only(raw, p):
    """(b) Histology class name only"""
    lines = [f"Histology: {p['histology']}"] if p["histology"] else []
    return build_knowledge(lines)

def tf_kg_only(raw, p):
    """(c) KG features with site, Histology class name withheld"""
    lines = ([p["site"]] if p["site"] else []) + p["features"]
    return build_knowledge(lines)

def tf_site_only(raw, p):
    """(d/site) Anatomical site only"""
    lines = [p["site"]] if p["site"] else []
    return build_knowledge(lines)


CONDITION_TRANSFORMS = {
    "full":       tf_full,
    "class_only": tf_class_only,
    "kg_only":    tf_kg_only,
    "site_only":  tf_site_only,
}

# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def transform_dataset(data: list, condition: str) -> list:
    transform = CONDITION_TRANSFORMS[condition]
    out, missing = [], 0
    for item in data:
        new_item = dict(item)
        raw = item.get("knowledge", "")
        if not raw or not raw.strip():
            missing += 1
            new_item["knowledge"] = ""
        else:
            parsed = parse_knowledge(raw)
            new_item["knowledge"] = transform(raw, parsed)
        out.append(new_item)
    if missing:
        print(f"    ⚠  {missing} samples had empty/missing knowledge fields")
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Prepare E1 ablation datasets")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show sample outputs without writing files")
    args = ap.parse_args()

    if not args.dry_run:
        os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("E1 Ablation Dataset Preparation")
    print(f"Input  : {DATASET_DIR}")
    print(f"Output : {OUT_DIR}")
    print("=" * 70)

    for split, src_path in SPLITS.items():
        print(f"\n── {split.upper()} SPLIT {'─'*40}")
        if not os.path.exists(src_path):
            print(f"  ✗ Not found: {src_path}")
            continue

        with open(src_path) as f:
            data = json.load(f)

        n_with = sum(1 for d in data if d.get("knowledge", "").strip())
        print(f"  Loaded {len(data)} samples  ({n_with} with knowledge)")

        for condition, transform in CONDITION_TRANSFORMS.items():
            transformed = transform_dataset(data, condition)
            out_name = f"WSI-Bench-{split}_Report_knowledge_{condition}.json"
            out_path = os.path.join(OUT_DIR, out_name)

            if args.dry_run:
                sample = next((d for d in data if d.get("knowledge", "").strip()), data[0])
                parsed = parse_knowledge(sample.get("knowledge", ""))
                result = transform(sample.get("knowledge", ""), parsed)
                print(f"\n  [{condition}] sample →  {repr(result[:130])}")
            else:
                with open(out_path, "w") as f:
                    json.dump(transformed, f, indent=2, ensure_ascii=False)
                size_kb = os.path.getsize(out_path) / 1024
                print(f"  ✓ [{condition:12s}] → {out_name}  ({size_kb:.0f} KB)")

    if not args.dry_run:
        print("\n" + "=" * 70)
        print(f"All variants in: {OUT_DIR}")
        print("=" * 70)


if __name__ == "__main__":
    main()
