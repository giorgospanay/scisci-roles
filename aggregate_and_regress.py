#!/usr/bin/env python3
"""
Step 2: Aggregate paper-level data to author-level stats and run regression.

Requires outputs from extract_works.py:
    author_paper.tsv   : author_id  paper_id  year  n_authors
    citations.tsv      : cited_id   citing_id

Computes per author:
    - works_count       : total papers
    - career_age        : 2024 - first publication year
    - mean_team_size    : average co-authors per paper
    - total_citations   : sum of incoming citations
    - h_index           : standard h-index
    - median_citations  : median citations per paper

Then merges with brokerage_roles.csv and runs OLS regressions:
    log(h_index)          ~ role + controls + discipline FE
    log(total_citations)  ~ role + controls + discipline FE
    log(median_citations) ~ role + controls + discipline FE

Saves:
    author_stats.csv         : author-level stats
    regression_results.txt   : model summaries
    figures/fig_coefplot.png : coefficient plot

Usage:
    python aggregate_and_regress.py
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "/N/slate/gpanayio/scisci-roles/obj"
SCRATCH_DIR = "/N/scratch/gpanayio/scisci-roles"
ROLES_FILE = os.path.join(OUTPUT_DIR, "brokerage_roles.csv")
AP_FILE    = os.path.join(SCRATCH_DIR, "author_paper.tsv")
CIT_FILE   = os.path.join(SCRATCH_DIR, "citations.tsv")

CURRENT_YEAR = 2024
MIN_PAPERS   = 3    # minimum papers to include author in regression

# ── HELPERS ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def h_index(citation_counts):
    """Compute h-index from an array of citation counts."""
    counts = np.sort(citation_counts)[::-1]
    h = 0
    for i, c in enumerate(counts, 1):
        if c >= i:
            h = i
        else:
            break
    return h


# ── LOAD & AGGREGATE ──────────────────────────────────────────────────────────

def load_author_paper():
    log(f"Loading author-paper links...")
    ap = pd.read_csv(AP_FILE, sep="\t",
                     dtype={"author_id": str, "paper_id": str,
                            "year": "Int64", "n_authors": "Int64"})
    log(f"  {len(ap):,} rows, {ap['author_id'].nunique():,} authors, "
        f"{ap['paper_id'].nunique():,} papers")
    return ap


def load_citations(relevant_papers):
    """
    Load citations.tsv filtered to papers we care about.
    relevant_papers: set of paper_ids authored by our target authors.
    """
    log("Loading citations (filtering to relevant papers)...")
    chunks = []
    chunksize = 5_000_000
    n_total = n_kept = 0

    for chunk in pd.read_csv(
        CIT_FILE, sep="\t",
        dtype={"cited_id": str, "citing_id": str},
        chunksize=chunksize
    ):
        n_total += len(chunk)
        chunk = chunk[chunk["cited_id"].isin(relevant_papers)]
        n_kept += len(chunk)
        chunks.append(chunk)
        if n_total % (chunksize * 5) == 0:
            log(f"  {n_total/1e6:.0f}M citation rows read, {n_kept:,} kept...")

    citations = pd.concat(chunks, ignore_index=True)
    log(f"  Done — {n_total/1e6:.1f}M total, {len(citations):,} kept")
    return citations


def compute_author_stats(ap, citations):
    log("Computing per-author productivity stats...")

    # Works count, career age, mean team size
    prod = ap.groupby("author_id").agg(
        works_count   = ("paper_id",   "count"),
        first_year    = ("year",        "min"),
        mean_team_size= ("n_authors",   "mean"),
    ).reset_index()
    prod["career_age"] = CURRENT_YEAR - prod["first_year"]

    log(f"  Productivity stats for {len(prod):,} authors")

    # Citations per paper
    log("Computing citations per paper...")
    cit_per_paper = (
        citations.groupby("cited_id").size()
        .rename("citations")
        .reset_index()
        .rename(columns={"cited_id": "paper_id"})
    )

    # Merge citations onto author-paper, fill missing with 0
    ap_cit = ap[["author_id", "paper_id"]].merge(
        cit_per_paper, on="paper_id", how="left"
    )
    ap_cit["citations"] = ap_cit["citations"].fillna(0).astype(int)

    log("Computing h-index and median citations per author...")
    def author_citation_stats(grp):
        counts = grp["citations"].values
        return pd.Series({
            "total_citations":  int(counts.sum()),
            "median_citations": float(np.median(counts)),
            "h_index":          h_index(counts),
        })

    cit_stats = (
        ap_cit.groupby("author_id")
        .apply(author_citation_stats)
        .reset_index()
    )
    log(f"  Citation stats for {len(cit_stats):,} authors")

    stats = prod.merge(cit_stats, on="author_id", how="left")
    return stats


# ── REGRESSION ────────────────────────────────────────────────────────────────

def run_regression(stats, roles):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        log("statsmodels not found — install with: pip install statsmodels")
        return None

    log("Merging stats with roles...")
    df = roles.merge(stats, left_index=True, right_on="author_id", how="inner")
    df = df[df["role"].isin(["coordinator", "gatekeeper", "representative", "liaison"])]
    df = df[df["works_count"] >= MIN_PAPERS]

    log(f"  Regression sample: {len(df):,} authors")

    # Log-transform outcomes (add 1 to avoid log(0))
    df["log_h"]      = np.log1p(df["h_index"])
    df["log_total"]  = np.log1p(df["total_citations"])
    df["log_median"] = np.log1p(df["median_citations"])
    df["log_works"]  = np.log1p(df["works_count"])

    # coordinator is reference category
    df["role"] = pd.Categorical(
        df["role"],
        categories=["coordinator", "gatekeeper", "representative", "liaison"]
    )

    results = {}
    outcomes = [
        ("log_h",      "log(h-index)"),
        ("log_total",  "log(total citations)"),
        ("log_median", "log(median citations per paper)"),
    ]

    formula_base = (
        "{outcome} ~ C(role) + log_works + career_age + mean_team_size"
        " + C(primary_discipline)"
    )

    result_lines = []
    for outcome, label in outcomes:
        log(f"  Fitting: {label}...")
        formula = formula_base.format(outcome=outcome)
        model   = smf.ols(formula, data=df).fit(cov_type="HC3")
        results[outcome] = model
        result_lines.append(f"\n{'='*70}")
        result_lines.append(f"Outcome: {label}")
        result_lines.append(f"N = {int(model.nobs):,}   R² = {model.rsquared:.3f}")
        result_lines.append(model.summary2().tables[1].to_string())

    out_path = os.path.join(OUTPUT_DIR, "regression_results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(result_lines))
    log(f"  Regression results saved to {out_path}")

    return results


# ── COEFFICIENT PLOT ──────────────────────────────────────────────────────────

def plot_coefplot(results):
    outcomes = {
        "log_h":      "log(h-index)",
        "log_total":  "log(total citations)",
        "log_median": "log(median citations / paper)",
    }
    roles    = ["gatekeeper", "representative", "liaison"]
    role_labels = {
        "gatekeeper":    "Gatekeeper",
        "representative":"Representative",
        "liaison":       "Liaison",
    }
    colors = {
        "gatekeeper":    "#DD8452",
        "representative":"#55A868",
        "liaison":       "#C44E52",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.suptitle(
        "Role effect on citation impact (ref: Coordinator)",
        fontweight="bold", y=1.02
    )

    for ax, (outcome, label) in zip(axes, outcomes.items()):
        model = results[outcome]
        for i, role in enumerate(roles):
            param = f"C(role)[T.{role}]"
            if param not in model.params:
                continue
            coef = model.params[param]
            ci   = model.conf_int().loc[param]
            ax.errorbar(
                coef, i,
                xerr=[[coef - ci[0]], [ci[1] - coef]],
                fmt="o", color=colors[role],
                capsize=4, linewidth=1.5, markersize=6,
                label=role_labels[role]
            )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.set_yticks(range(len(roles)))
        ax.set_yticklabels([role_labels[r] for r in roles])
        ax.set_xlabel("Coefficient (vs coordinator)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, "fig_coefplot.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    log(f"  Coefficient plot saved to {out}")
    plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = load_author_paper()

    relevant_papers = set(ap["paper_id"].unique())
    citations = load_citations(relevant_papers)

    stats = compute_author_stats(ap, citations)

    # Save author stats
    stats_path = os.path.join(OUTPUT_DIR, "author_stats.csv")
    stats.to_csv(stats_path, index=False)
    log(f"Author stats saved to {stats_path}")

    # Load roles
    log("Loading roles...")
    roles = pd.read_csv(ROLES_FILE, index_col=0)
    roles.columns = [
        "primary_discipline", "all_disciplines",
        "blau_collab", "blau_sim",
        "collab_diverse", "sim_diverse", "role"
    ]

    results = run_regression(stats, roles)
    if results:
        plot_coefplot(results)

    log("=== Done ===")


if __name__ == "__main__":
    main()
