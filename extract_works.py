#!/usr/bin/env python3
"""
Step 1: Extract author-paper links and citation inversion from works TSV.

For each article-type work with known authors:
  - Emit (author_id, paper_id, year, n_authors) → for productivity/career age
  - Emit (cited_paper_id, citing_paper_id)       → for citation inversion

Writes two files to OUTPUT_DIR:
  author_paper.tsv      : author_id  paper_id  year  n_authors
  citations.tsv         : cited_id   citing_id

Filters to authors in brokerage_roles.csv only to keep output manageable.

Usage:
    python extract_works.py
"""

import os
import ast
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────

WORKS_FILE   = "/N/scratch/gpanayio/openalex-pre/works_core+basic+authorship+ids+funding+concepts+references+mesh.tsv"
ROLES_FILE   = "/N/slate/gpanayio/scisci-roles/obj/brokerage_roles.csv"
OUTPUT_DIR   = "/N/slate/gpanayio/scisci-roles/obj"
SCRATCH_DIR  = "/N/scratch/gpanayio/scisci-roles"

# Column indices (0-based) — confirmed from header
COL_ID       = 0
COL_TYPE     = 2
COL_YEAR     = 3
COL_AUTHORS  = 11   # authorships:author:id
COL_REFS     = 23   # referenced_works

BATCH_SIZE   = 1_000_000   # log every N lines

# ── HELPERS ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_ids(cell):
    """Parse '[123,456,789]' or '[]' into a list of strings."""
    cell = cell.strip()
    if cell == "[]" or not cell:
        return []
    # strip brackets and split
    return cell[1:-1].split(",")


# ── LOAD TARGET AUTHORS ───────────────────────────────────────────────────────

def load_target_authors(path):
    log(f"Loading target authors from {path}...")
    authors = set()
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            author_id = line.split(",")[0].strip()
            if author_id:
                authors.add(author_id)
    log(f"  Loaded {len(authors):,} target authors")
    return authors


# ── MAIN EXTRACTION ───────────────────────────────────────────────────────────

def extract(target_authors):
    ap_path  = os.path.join(SCRATCH_DIR, "author_paper.tsv")
    cit_path = os.path.join(SCRATCH_DIR, "citations.tsv")

    log(f"Writing author-paper links to {ap_path}")
    log(f"Writing citation pairs to     {cit_path}")
    log(f"Streaming {WORKS_FILE} ...")

    n_lines = n_kept = n_ap = n_cit = 0

    with open(WORKS_FILE) as fin, \
         open(ap_path,  "w") as fap, \
         open(cit_path, "w") as fcit:

        fap.write("author_id\tpaper_id\tyear\tn_authors\n")
        fcit.write("cited_id\tciting_id\n")

        next(fin)  # skip header

        for line in fin:
            n_lines += 1
            if n_lines % BATCH_SIZE == 0:
                log(f"  {n_lines/1e6:.0f}M lines | kept {n_kept:,} papers | "
                    f"{n_ap:,} author-paper | {n_cit:,} citation pairs")

            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(COL_AUTHORS, COL_REFS):
                continue

            # Only articles, skip grants/paratexts
            if parts[COL_TYPE] != "article":
                continue

            authors = parse_ids(parts[COL_AUTHORS])
            if not authors:
                continue

            # Check if any author is in our target set
            our_authors = [a for a in authors if a in target_authors]
            if not our_authors:
                continue

            paper_id = parts[COL_ID]
            year     = parts[COL_YEAR].strip()
            n_auth   = len(authors)
            refs     = parse_ids(parts[COL_REFS])

            n_kept += 1

            # author-paper rows
            for a in our_authors:
                fap.write(f"{a}\t{paper_id}\t{year}\t{n_auth}\n")
                n_ap += 1

            # citation inversion: this paper cites each ref
            # → each ref receives a citation from this paper
            for ref in refs:
                fcit.write(f"{ref}\t{paper_id}\n")
                n_cit += 1

    log(f"Done. {n_lines/1e6:.1f}M lines processed.")
    log(f"  Papers kept       : {n_kept:,}")
    log(f"  Author-paper rows : {n_ap:,}")
    log(f"  Citation pairs    : {n_cit:,}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    target_authors = load_target_authors(ROLES_FILE)
    extract(target_authors)


if __name__ == "__main__":
    main()
