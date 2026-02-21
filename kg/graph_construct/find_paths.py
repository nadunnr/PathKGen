"""
find_paths.py

For each histology in the dataset, find the shortest path between the histology CUI
and each of its associated morphological feature CUIs in the UMLS Neo4j graph.

Output format per morphology path:
  HistologyName(CUI) ----(rel1)----> Node1Name(CUI) ----(rel2)----> ... ----(relN)----> MorphologyName(CUI)

Plus per-histology statistics (relation type counts, path-length distribution,
most common relation sequences).

Sources:
  - Histology → CUI:  /home/nadun/wd/datasets/wsi_bench/kg_data/histology_cui.py  (CUI_MAP)
  - Morphology CUIs:  /home/nadun/wd/datasets/wsi_bench/kg_data/top_cuis.json

Output:
  <script_dir>/output/histology_morphology_paths.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# ─── path setup ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from graphs import Neo4jQuery  # noqa: E402

# ─── data files ───────────────────────────────────────────────────────────────
HISTOLOGY_CUI_PY = Path("/home/nadun/wd/datasets/wsi_bench/kg_data/histology_cui.py")
# TOP_CUIS_JSON    = Path("/home/nadun/wd/datasets/wsi_bench/kg_data/top_cuis/top_150_cuis.json")
TOP_CUIS_JSON    = Path("/home/nadun/wd/datasets/wsi_bench/kg_data/top_cuis/top_10_cuis.json")
OUTPUT_DIR         = Path(__file__).resolve().parent / "output"
OUT_VISUAL         = OUTPUT_DIR / "paths_visual.json"       # human-readable path strings
OUT_STRUCTURED     = OUTPUT_DIR / "paths_structured.json"   # nodes + relationships arrays
OUT_STATISTICS     = OUTPUT_DIR / "statistics.json"         # per-histology stats

# ─── Neo4j connection ─────────────────────────────────────────────────────────
NEO4J_URL      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "11111111"
NEO4J_DATABASE = "neo4jv8"

MAX_DEPTH = 5


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_histology_cui_map() -> dict:
    ns = {}
    exec(HISTOLOGY_CUI_PY.read_text(), ns)
    return ns["CUI_MAP"]


def load_top_cuis() -> dict:
    with open(TOP_CUIS_JSON) as f:
        return json.load(f)


def build_histology_key_map(cui_map: dict, top_cuis: dict) -> dict:
    """
    Return { histology_name: { histology_cui, morphology: {cui: name} } }
    guarding against slight key differences between CUI_MAP and top_cuis.json.
    """
    def _norm(s: str) -> str:
        return s.lower().replace("-", " ").replace("_", " ").strip()

    norm_to_tc_key = {_norm(k): k for k in top_cuis}

    result = {}
    for histo_name, histo_cui in cui_map.items():
        tc_key = None
        if histo_name in top_cuis:
            tc_key = histo_name
        else:
            histo_norm = _norm(histo_name)
            for norm_k, orig_k in norm_to_tc_key.items():
                if histo_norm in norm_k or norm_k in histo_norm:
                    tc_key = orig_k
                    break

        if tc_key is None:
            print(f"  [WARN] No morphology CUIs found for: {histo_name!r}")
            continue

        cuis  = top_cuis[tc_key]["cuis"]
        names = top_cuis[tc_key]["names"]
        result[histo_name] = {
            "histology_cui": histo_cui,
            "morphology": dict(zip(cuis, names)),
        }
    return result


def build_path_string(histo_name: str, histo_cui: str,
                      nodes: list, relationships: list) -> str:
    """
    Render a path as:
      HistoName(CUI) ----(rel1)----> Node1(CUI) ----(rel2)----> Node2(CUI) ...
    """
    parts = [f"{histo_name}({histo_cui})"]
    for rel, node in zip(relationships, nodes[1:]):
        rel_type = rel.get("type") or rel.get("rela") or "?"
        parts.append(f"----({rel_type})---->")
        parts.append(f"{node['name']}({node['cui']})")
    return " ".join(parts)


def compute_statistics(paths_list: list) -> dict:
    """
    Given a list of path dicts (or None for missing paths), compute:
      - total / found / not_found counts
      - path length distribution
      - relation type frequency (across all hop positions)
      - top-5 most common full relation-sequence patterns
    """
    found_paths = [p for p in paths_list if p is not None]
    not_found   = len(paths_list) - len(found_paths)

    length_dist   = Counter()
    rel_type_freq = Counter()
    seq_freq      = Counter()

    for p in found_paths:
        length_dist[p["length"]] += 1
        seq = []
        for rel in p["relationships"]:
            rt = rel.get("type") or rel.get("rela") or "?"
            rel_type_freq[rt] += 1
            seq.append(rt)
        seq_freq[" -> ".join(seq)] += 1

    return {
        "total_morphologies": len(paths_list),
        "paths_found":        len(found_paths),
        "paths_not_found":    not_found,
        "path_length_distribution": dict(sorted(length_dist.items())),
        "relation_type_frequency":  dict(rel_type_freq.most_common()),
        "top_relation_sequences":   dict(seq_freq.most_common(5)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main logic
# ══════════════════════════════════════════════════════════════════════════════

def find_and_format(db: Neo4jQuery, histology_map: dict) -> dict:
    output = {}

    for histo_name, info in tqdm(histology_map.items(), desc="Histologies"):
        histo_cui  = info["histology_cui"]
        morphology = info["morphology"]   # {morph_cui: morph_name}

        morph_entries = {}
        raw_paths     = []

        for morph_cui, morph_name in tqdm(
            morphology.items(),
            desc=f"  {histo_name[:40]}",
            leave=False,
        ):
            path = db.find_shortest_path(histo_cui, morph_cui, max_depth=MAX_DEPTH)
            raw_paths.append(path)

            if path is not None:
                path_str = build_path_string(
                    histo_name, histo_cui,
                    path["nodes"], path["relationships"]
                )
                morph_entries[morph_cui] = {
                    "morphology_name": morph_name,
                    "path_length":     path["length"],
                    "path_string":     path_str,
                    # keep structured data for downstream use
                    "nodes":           path["nodes"],
                    "relationships":   path["relationships"],
                }
            else:
                morph_entries[morph_cui] = {
                    "morphology_name": morph_name,
                    "path_length":     None,
                    "path_string":     None,
                    "nodes":           None,
                    "relationships":   None,
                }

        stats = compute_statistics(raw_paths)
        print(
            f"  {histo_name}: "
            f"{stats['paths_found']}/{stats['total_morphologies']} paths found"
        )

        output[histo_name] = {
            "histology_cui":    histo_cui,
            "morphology_paths": morph_entries,
            "statistics":       stats,
        }

    return output


def main():
    print("Loading histology CUI map …")
    cui_map = load_histology_cui_map()
    print(f"  {len(cui_map)} histology classes loaded.")

    print("Loading morphology CUIs …")
    top_cuis = load_top_cuis()
    print(f"  {len(top_cuis)} histology entries in top_cuis.json.")

    print("Resolving histology ↔ morphology mapping …")
    histology_map = build_histology_key_map(cui_map, top_cuis)
    print(f"  {len(histology_map)} histologies matched.\n")

    print(f"Connecting to Neo4j at {NEO4J_URL} …")
    db = Neo4jQuery(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, database=NEO4J_DATABASE)

    print(f"Finding shortest paths (max depth={MAX_DEPTH}) …\n")
    results = find_and_format(db, histology_map)
    db.close()

    # ── split into three output dicts ─────────────────────────────────────────
    visual     = {}   # { histology: { histology_cui, morphology_paths: {cui: {name, path_string}} } }
    structured = {}   # { histology: { histology_cui, morphology_paths: {cui: {name, nodes, relationships, length}} } }
    statistics = {}   # { histology: { histology_cui, statistics: {...} } }

    for histo, data in results.items():
        hcui  = data["histology_cui"]
        morph = data["morphology_paths"]
        stats = data["statistics"]

        visual[histo] = {
            "histology_cui": hcui,
            "morphology_paths": {
                cui: {
                    "morphology_name": m["morphology_name"],
                    "path_string":     m["path_string"],
                }
                for cui, m in morph.items()
            },
        }

        structured[histo] = {
            "histology_cui": hcui,
            "morphology_paths": {
                cui: {
                    "morphology_name": m["morphology_name"],
                    "path_length":     m["path_length"],
                    "nodes":           m["nodes"],
                    "relationships":   m["relationships"],
                }
                for cui, m in morph.items()
            },
        }

        statistics[histo] = {
            "histology_cui": hcui,
            "statistics":    stats,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, obj, label in [
        (OUT_VISUAL,     visual,     "Visual paths"),
        (OUT_STRUCTURED, structured, "Structured paths"),
        (OUT_STATISTICS, statistics, "Statistics"),
    ]:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"✓ {label:20s} → {path}")

    total_pairs = sum(len(v["morphology_paths"]) for v in results.values())
    total_found = sum(v["statistics"]["paths_found"] for v in results.values())
    print(f"\n✓ Done.  {total_found}/{total_pairs} total paths found.")


if __name__ == "__main__":
    main()
