"""
format_paths.py

Reads histology_morphology_paths.json (produced by find_paths.py) and:

1. Reformats each path as a human-readable chain:
       histology_name(histology_cui) ---[rel_type]---> node1_name(node1_cui) ---[rel_type]---> ... ---> morphology_name(morph_cui)

2. Saves two output files per histology:
   output/formatted/
       <histology_slug>.txt      – all formatted path strings for that histology
       <histology_slug>_stats.txt – per-histology statistics log

3. Saves a combined JSON:
   output/formatted/all_formatted_paths.json
       { histology: { morph_cui: { "morphology_name": ..., "path_str": ..., "length": N } } }
"""

import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict

# ─── I/O paths ────────────────────────────────────────────────────────────────
INPUT_JSON   = Path("/mnt/hdd4tb/vlm_histo/kg/graph_construct/output/histology_morphology_paths.json")
OUTPUT_DIR   = Path("/mnt/hdd4tb/vlm_histo/kg/graph_construct/output/formatted")
COMBINED_JSON = OUTPUT_DIR / "all_formatted_paths.json"
GLOBAL_STATS  = OUTPUT_DIR / "_global_stats.txt"


# ─── helpers ──────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Convert histology name to a safe filename slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def format_path(histo_name: str, histo_cui: str, path_data: dict) -> str:
    """
    Build the chain string from a path dict:
        histology_name(cui) ---[rel]---> node1_name(cui) ---[rel]---> ... ---> morph_name(cui)

    path_data has:
        nodes:         [ {cui, name}, ... ]   len = N+1
        relationships: [ {type, rela}, ... ]  len = N
    """
    nodes = path_data["nodes"]
    rels  = path_data["relationships"]

    # Build each segment: node ---[rel]--->
    parts = []
    for i, node in enumerate(nodes):
        label = f"{node['name']}({node['cui']})"
        parts.append(label)
        if i < len(rels):
            rel_type = rels[i]["rela"] or rels[i]["type"]  # prefer rela if set
            parts.append(f" ---[{rel_type}]---> ")

    return "".join(parts)


def compute_stats(histo_name: str, morph_paths: dict) -> str:
    """
    Returns a formatted statistics block for one histology.

    Covers:
     - total morphology features / paths found / not found
     - path length distribution
     - most common relation types (across all path edges)
     - most common relation patterns (full sequence)
     - intermediate nodes seen most often
    """
    total        = len(morph_paths)
    found        = [v for v in morph_paths.values() if v["path"] is not None]
    not_found    = [v for v in morph_paths.values() if v["path"] is None]

    length_dist  = Counter()
    rel_counter  = Counter()   # individual relation types
    pattern_ctr  = Counter()   # full relation-type sequence as a string
    inter_nodes  = Counter()   # intermediate node names (exclude first & last)

    for v in found:
        path = v["path"]
        nodes = path["nodes"]
        rels  = path["relationships"]
        length_dist[path["length"]] += 1

        rel_seq = []
        for r in rels:
            rt = r["rela"] or r["type"]
            rel_counter[rt] += 1
            rel_seq.append(rt)
        pattern_ctr[" → ".join(rel_seq)] += 1

        # Intermediate nodes: all except first (histology) and last (morphology)
        for node in nodes[1:-1]:
            inter_nodes[node["name"]] += 1

    lines = [
        "=" * 70,
        f"HISTOLOGY: {histo_name}",
        "=" * 70,
        f"  Morphology features :  {total}",
        f"  Paths found         :  {len(found)}",
        f"  Not found           :  {len(not_found)}",
    ]

    if not_found:
        lines.append("\n  Missing paths for:")
        for v in not_found:
            lines.append(f"    - {v['morphology_name']}")

    # ── path length distribution ──────────────────────────────────────────────
    lines.append("\n  Path length distribution:")
    for length in sorted(length_dist):
        bar = "█" * length_dist[length]
        lines.append(f"    {length} hops : {bar} ({length_dist[length]})")

    # ── top relation types ────────────────────────────────────────────────────
    lines.append("\n  Top relation types (across all edges):")
    for rel, cnt in rel_counter.most_common(10):
        lines.append(f"    {cnt:3d}x  {rel}")

    # ── top relation patterns ─────────────────────────────────────────────────
    lines.append("\n  Most common path patterns (relation sequences):")
    for pattern, cnt in pattern_ctr.most_common(5):
        lines.append(f"    {cnt:3d}x  [ {pattern} ]")

    # ── top intermediate nodes ────────────────────────────────────────────────
    lines.append("\n  Most common intermediate nodes:")
    for node_name, cnt in inter_nodes.most_common(10):
        lines.append(f"    {cnt:3d}x  {node_name}")

    lines.append("")
    return "\n".join(lines)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT_JSON} …")
    with open(INPUT_JSON) as f:
        data = json.load(f)
    print(f"  {len(data)} histologies loaded.\n")

    all_formatted   = {}
    global_rel_ctr  = Counter()
    global_len_dist = Counter()
    global_inter    = Counter()
    global_patterns = Counter()
    global_totals   = {"total": 0, "found": 0, "not_found": 0}

    for histo_name, histo_data in data.items():
        histo_cui    = histo_data["histology_cui"]
        morph_paths  = histo_data["morphology_paths"]   # { morph_cui: {morphology_name, path} }

        slug = slugify(histo_name)
        paths_file = OUTPUT_DIR / f"{slug}.txt"
        stats_file = OUTPUT_DIR / f"{slug}_stats.txt"

        formatted_morph = {}
        path_lines      = [f"{'='*70}", f"HISTOLOGY: {histo_name}  ({histo_cui})", f"{'='*70}", ""]

        for morph_cui, entry in morph_paths.items():
            morph_name = entry["morphology_name"]
            path       = entry["path"]

            if path is None:
                path_str = f"[NO PATH FOUND]  {morph_name}({morph_cui})"
            else:
                path_str = format_path(histo_name, histo_cui, path)

            formatted_morph[morph_cui] = {
                "morphology_name": morph_name,
                "path_str": path_str,
                "length": path["length"] if path else None,
            }
            path_lines.append(f"→ {morph_name} ({morph_cui})")
            path_lines.append(f"  {path_str}")
            path_lines.append("")

        # Write formatted paths file
        paths_file.write_text("\n".join(path_lines), encoding="utf-8")

        # Compute and write stats
        stats_text = compute_stats(histo_name, morph_paths)
        stats_file.write_text(stats_text, encoding="utf-8")

        # Accumulate global stats
        found_cnt     = sum(1 for v in morph_paths.values() if v["path"] is not None)
        not_found_cnt = len(morph_paths) - found_cnt
        global_totals["total"]     += len(morph_paths)
        global_totals["found"]     += found_cnt
        global_totals["not_found"] += not_found_cnt

        for v in morph_paths.values():
            if v["path"]:
                p = v["path"]
                global_len_dist[p["length"]] += 1
                seq = []
                for r in p["relationships"]:
                    rt = r["rela"] or r["type"]
                    global_rel_ctr[rt] += 1
                    seq.append(rt)
                global_patterns[" → ".join(seq)] += 1
                for node in p["nodes"][1:-1]:
                    global_inter[node["name"]] += 1

        all_formatted[histo_name] = {
            "histology_cui": histo_cui,
            "morphology_paths": formatted_morph,
        }

        print(f"  {histo_name}: {found_cnt}/{len(morph_paths)} paths written → {paths_file.name}")

    # ── Save combined JSON ────────────────────────────────────────────────────
    with open(COMBINED_JSON, "w") as f:
        json.dump(all_formatted, f, indent=2)
    print(f"\n✓ Combined JSON saved to: {COMBINED_JSON}")

    # ── Write global stats ────────────────────────────────────────────────────
    glines = [
        "=" * 70,
        "GLOBAL STATISTICS ACROSS ALL HISTOLOGIES",
        "=" * 70,
        f"  Total histologies       : {len(data)}",
        f"  Total morphology pairs  : {global_totals['total']}",
        f"  Paths found             : {global_totals['found']}",
        f"  Paths not found         : {global_totals['not_found']}",
        f"  Coverage                : {global_totals['found']/max(global_totals['total'],1)*100:.1f}%",
        "",
        "  Global path length distribution:",
    ]
    for length in sorted(global_len_dist):
        bar = "█" * global_len_dist[length]
        glines.append(f"    {length} hops : {bar} ({global_len_dist[length]})")

    glines.append("\n  Top relation types globally:")
    for rel, cnt in global_rel_ctr.most_common(15):
        glines.append(f"    {cnt:4d}x  {rel}")

    glines.append("\n  Top path patterns globally:")
    for pattern, cnt in global_patterns.most_common(10):
        glines.append(f"    {cnt:4d}x  [ {pattern} ]")

    glines.append("\n  Top intermediate nodes globally:")
    for node_name, cnt in global_inter.most_common(15):
        glines.append(f"    {cnt:4d}x  {node_name}")

    GLOBAL_STATS.write_text("\n".join(glines), encoding="utf-8")
    print(f"✓ Global stats saved to: {GLOBAL_STATS}")
    print(f"✓ Per-histology files in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
