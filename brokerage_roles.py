"""
Brokerage role analysis for multilayer disciplinary networks.

Edgelist formats (no header):
    collaboration layer : src<TAB>dst<TAB>weight   (integer weights = co-author count)
    similarity layer    : src<SPACE>dst<SPACE>weight (float weights)

Discipline is inferred from filename:
    filtered_collaboration_layer_{disc}.edgelist
    filtered_author_similarity_layer_{disc}.edgelist

Roles (Yu et al. + liaison):
    Coordinator   : low collab diversity, low similarity diversity
    Gatekeeper    : high collab diversity, low similarity diversity
    Representative: low collab diversity, high similarity diversity
    Liaison       : high collab diversity, high similarity diversity

Memory strategy: edgelists are loaded one discipline at a time, reduced to
a compact (author, discipline, count) table, then freed before the next load.
Peak memory is dominated by a single edgelist at a time, not all of them.
"""

import gc
import os
import time
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR     = "/N/slate/gpanayio/scisci-roles"
EDGELIST_DIR = "/N/slate/gpanayio/scisci-gatekeepers/obj"
OUTPUT_DIR   = os.path.join(BASE_DIR, "obj")
SCRATCH_DIR  = "/N/scratch/gpanayio/scisci-roles-counts"  # cached per-disc neighbor counts

DISCIPLINES = ["CS", "Biology", "Math", "Physics", "Sociology", "Economics"]  # Linguistics excluded — layers not ready yet

SIMILARITY_AVAILABLE = {
    "CS":        True,
    "Biology":   True,
    "Math":      True,
    "Physics":   True,
    "Sociology": True,
    "Economics": True,
}

MIN_COLLABORATORS = 5   # minimum distinct neighbors in collab layer
MIN_SIM_NEIGHBORS = 5   # minimum distinct neighbors in similarity layer

# None = within-discipline percentile cutoff (see BLAU_PERCENTILE below)
# or set a fixed float e.g. 0.3
BLAU_THRESHOLD  = None   # None = use percentile; or set a fixed float e.g. 0.3
BLAU_PERCENTILE = 0.80   # top 20th percentile

# ── HELPERS ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def blau_from_counts(group):
    """
    Compute Blau index from a group of (neighbor_disc, count) rows.
    Uses edge counts as weights so repeated edges count proportionally.
    """
    total = group["count"].sum()
    p = group["count"] / total
    return float(1 - (p ** 2).sum())


def stream_edgelist_to_neighbor_counts(path, sep, disc, chunksize=5_000_000):
    """
    Stream a (potentially huge) edgelist in chunks and accumulate it directly
    into a compact (author, neighbor_disc, count) table, without ever holding
    the full edgelist — or even one full chunk's src+dst doubled copy — in
    memory at the same time as the running total.

    This replaces the old load-everything-then-reduce approach, which could
    OOM on large similarity layers (e.g. Physics: 150M+ edges).
    """
    log(f"  Streaming {os.path.basename(path)}...")
    running = None  # accumulating (author, neighbor_disc) -> count, as a Series
    n_edges = 0

    for chunk in pd.read_csv(
        path, sep=sep, header=None,
        names=["src", "dst", "weight"],
        dtype={"src": str, "dst": str, "weight": float},
        chunksize=chunksize,
    ):
        n_edges += len(chunk)
        both = pd.concat([chunk["src"], chunk["dst"]], ignore_index=True)
        chunk_counts = both.value_counts()
        del both, chunk
        if running is None:
            running = chunk_counts
        else:
            running = running.add(chunk_counts, fill_value=0)
        del chunk_counts

    log(f"  Done — {n_edges:,} edges streamed")

    if running is None:
        return pd.DataFrame(columns=["author", "neighbor_disc", "count"])

    counts = running.rename("count").rename_axis("author").reset_index()
    counts["neighbor_disc"] = disc
    counts["count"] = counts["count"].astype("int64")
    return counts[["author", "neighbor_disc", "count"]]


# ── BUILD NEIGHBOR COUNTS (one discipline at a time) ─────────────────────────

def build_neighbor_counts(layer_type, min_neighbors):
    """
    Load each discipline edgelist, reduce to (author, disc, count), free edgelist.
    Caches the compact per-discipline count tables to SCRATCH_DIR as parquet
    files so reruns skip the expensive streaming step entirely.
    Returns a single stacked counts table across all disciplines.
    """
    log(f"Building neighbor counts for {layer_type} layer...")
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    parts = []

    for disc in DISCIPLINES:
        if layer_type == "sim" and not SIMILARITY_AVAILABLE.get(disc, False):
            log(f"  Skipping {disc} (similarity unavailable)")
            continue

        cache_path = os.path.join(SCRATCH_DIR, f"counts_{layer_type}_{disc}.parquet")

        if os.path.exists(cache_path):
            log(f"  {disc}: loading cached counts from {cache_path}")
            counts = pd.read_parquet(cache_path)
            log(f"  {disc}: {len(counts):,} author-discipline pairs (from cache)")
        else:
            if layer_type == "collab":
                path = os.path.join(EDGELIST_DIR, f"filtered_collaboration_layer_{disc}.edgelist")
                sep  = "\t"
            else:
                path = os.path.join(EDGELIST_DIR, f"filtered_author_similarity_layer_{disc}.edgelist")
                sep  = " "

            log(f"  {disc}: streaming from {os.path.basename(path)}...")
            counts = stream_edgelist_to_neighbor_counts(path, sep, disc)
            log(f"  {disc}: {len(counts):,} author-discipline pairs")

            counts.to_parquet(cache_path, index=False)
            log(f"  {disc}: counts cached to {cache_path}")

        parts.append(counts)
        gc.collect()

    log("  Concatenating all disciplines...")
    all_counts = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    # Filter by minimum neighbor count
    author_totals = all_counts.groupby("author")["count"].sum()
    valid = author_totals[author_totals >= min_neighbors].index
    all_counts = all_counts[all_counts["author"].isin(valid)]
    log(f"  Retained {len(valid):,} authors with >= {min_neighbors} neighbors")

    return all_counts


# ── BUILD AUTHOR → PRIMARY DISCIPLINE MAPPING ─────────────────────────────────

def build_author_discipline(collab_counts):
    """
    Derive primary discipline and all-disciplines set from the
    already-compact collab counts table (no reload needed).
    """
    log("Building author → discipline mapping...")

    primary = (
        collab_counts.sort_values("count", ascending=False)
        .drop_duplicates(subset="author", keep="first")
        [["author", "neighbor_disc"]]
        .rename(columns={"neighbor_disc": "primary_discipline"})
        .set_index("author")
    )

    all_discs = (
        collab_counts.groupby("author")["neighbor_disc"]
        .apply(set)
        .rename("all_disciplines")
    )

    result = primary.join(all_discs)
    n_cross = (result["all_disciplines"].apply(len) > 1).sum()
    log(f"  Total unique authors: {len(result):,}  ({n_cross:,} appear in >1 discipline)")
    return result


# ── COMPUTE BLAU INDICES ──────────────────────────────────────────────────────

def compute_blau(counts, layer_name):
    log(f"Computing Blau indices for {layer_name} layer...")
    blau = (
        counts.groupby("author")
        .apply(blau_from_counts, include_groups=False)
        .rename(f"blau_{layer_name}")
    )
    log(f"  Done — {len(blau):,} authors scored")
    return blau


# ── ROLE ASSIGNMENT ───────────────────────────────────────────────────────────

def assign_roles(blau_collab, blau_sim, author_discipline):
    log("Assigning roles...")

    df = author_discipline.join(blau_collab, how="left").join(blau_sim, how="left")

    has_both = df["blau_collab"].notna() & df["blau_sim"].notna()
    log(f"  Authors with both scores : {has_both.sum():,}")
    log(f"  Authors missing sim score: {(~has_both).sum():,}")

    df["collab_diverse"] = np.nan
    df["sim_diverse"]    = np.nan

    if BLAU_THRESHOLD is None:
        log(f"  Thresholds: within-discipline top {100*(1-BLAU_PERCENTILE):.0f}th percentile")
        df.loc[has_both, "collab_diverse"] = (
            df[has_both].groupby("primary_discipline")["blau_collab"]
            .transform(lambda x: x > x.quantile(BLAU_PERCENTILE))
        )
        df.loc[has_both, "sim_diverse"] = (
            df[has_both].groupby("primary_discipline")["blau_sim"]
            .transform(lambda x: x > x.quantile(BLAU_PERCENTILE))
        )
    else:
        log(f"  Thresholds: fixed at {BLAU_THRESHOLD}")
        df.loc[has_both, "collab_diverse"] = df.loc[has_both, "blau_collab"] > BLAU_THRESHOLD
        df.loc[has_both, "sim_diverse"]    = df.loc[has_both, "blau_sim"]    > BLAU_THRESHOLD

    def _role(row):
        if pd.isna(row["collab_diverse"]) or pd.isna(row["sim_diverse"]):
            return "unassigned"
        c, s = bool(row["collab_diverse"]), bool(row["sim_diverse"])
        if   not c and not s: return "coordinator"
        elif     c and not s: return "gatekeeper"
        elif not c and     s: return "representative"
        else:                 return "liaison"

    log("  Applying role labels...")
    df["role"] = df.apply(_role, axis=1)
    n_assigned = (df["role"] != "unassigned").sum()
    log(f"  Done — {n_assigned:,} authors assigned a role")
    return df


# ── SUMMARIZE ─────────────────────────────────────────────────────────────────

def summarize(df):
    assigned = df[df["role"] != "unassigned"]

    print("\n── Role distribution (overall) ──────────────────────────")
    print(assigned["role"].value_counts().to_string())

    print("\n── Role distribution by primary discipline ──────────────")
    pivot = (
        assigned.groupby(["primary_discipline", "role"])
        .size().unstack(fill_value=0)
    )
    print(pivot.to_string())

    print("\n── Mean Blau scores by role ─────────────────────────────")
    print(
        assigned.groupby("role")[["blau_collab", "blau_sim"]]
        .mean().round(3).to_string()
    )

    print("\n── Cross-disciplinary authors (appear in >1 discipline) ─")
    cross = df[df["all_disciplines"].apply(len) > 1]
    print(f"  Total: {len(cross):,}")
    if len(cross):
        print(cross.groupby("primary_discipline")["role"].value_counts().to_string())


def check_files_exist():
    """Verify all expected edgelist files exist before doing any heavy work."""
    log("Checking that all expected edgelist files exist...")
    missing = []
    for disc in DISCIPLINES:
        collab_path = os.path.join(EDGELIST_DIR, f"filtered_collaboration_layer_{disc}.edgelist")
        if not os.path.exists(collab_path):
            missing.append(collab_path)

        if SIMILARITY_AVAILABLE.get(disc, False):
            sim_path = os.path.join(EDGELIST_DIR, f"filtered_author_similarity_layer_{disc}.edgelist")
            if not os.path.exists(sim_path):
                missing.append(sim_path)

    if missing:
        log("  Missing files:")
        for m in missing:
            log(f"    {m}")
        raise FileNotFoundError(
            f"{len(missing)} expected edgelist file(s) not found. "
            "Either wait for the pipeline to finish, or set SIMILARITY_AVAILABLE[disc] = False "
            "for disciplines whose similarity layer isn't ready yet."
        )
    log("  All expected files present.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("=== Brokerage role analysis starting ===")

    check_files_exist()

    # Load and immediately reduce each edgelist — never hold >1 full edgelist
    collab_counts = build_neighbor_counts("collab", MIN_COLLABORATORS)
    author_discipline = build_author_discipline(collab_counts)

    blau_collab = compute_blau(collab_counts, "collab")
    del collab_counts
    gc.collect()

    sim_counts  = build_neighbor_counts("sim", MIN_SIM_NEIGHBORS)
    blau_sim    = compute_blau(sim_counts, "sim")
    del sim_counts
    gc.collect()

    results = assign_roles(blau_collab, blau_sim, author_discipline)
    summarize(results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "brokerage_roles.csv")
    log(f"Saving results to {out_path}...")
    results.to_csv(out_path)

    log(f"=== Done in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
