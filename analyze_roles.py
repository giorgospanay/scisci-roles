"""
Analysis and visualization of brokerage role assignment results.

Usage:
    python analyze_roles.py brokerage_roles.csv

Produces:
    - Console summary (counts, percentages, Blau score distributions)
    - figures/fig1_role_distribution.png
    - figures/fig2_roles_by_discipline.png
    - figures/fig3_blau_distributions.png
    - figures/fig4_discipline_combinations.png
"""

import os
import sys
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_FILE  = sys.argv[1] if len(sys.argv) > 1 else "obj/brokerage_roles.csv"
FIGURE_DIR  = "figures"

ROLE_ORDER  = ["coordinator", "gatekeeper", "representative", "liaison", "unassigned"]
ROLE_COLORS = {
    "coordinator":   "#4C72B0",
    "gatekeeper":    "#DD8452",
    "representative":"#55A868",
    "liaison":       "#C44E52",
    "unassigned":    "#cccccc",
}

DISC_ORDER = ["CS", "Biology", "Math", "Physics", "Sociology", "Economics", "Linguistics"]

# ── LOAD & CLEAN ──────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path, index_col=0)
    df.columns = [
        "primary_discipline", "all_disciplines",
        "blau_collab", "blau_sim",
        "collab_diverse", "sim_diverse",
        "role"
    ]

    # Parse set stored as string e.g. "{'CS', 'Biology'}"
    def parse_set(s):
        try:
            return frozenset(ast.literal_eval(s))
        except Exception:
            return frozenset()

    df["all_disciplines"] = df["all_disciplines"].apply(parse_set)
    df["n_disciplines"]   = df["all_disciplines"].apply(len)
    df["disc_combo"]      = df["all_disciplines"].apply(
        lambda s: " + ".join(sorted(s)) if s else "unknown"
    )
    return df

# ── CONSOLE SUMMARY ───────────────────────────────────────────────────────────

def print_summary(df):
    assigned = df[df["role"] != "unassigned"]
    total    = len(df)
    n_assign = len(assigned)

    print("=" * 60)
    print("BROKERAGE ROLE ANALYSIS — SUMMARY")
    print("=" * 60)
    print(f"\nTotal authors in file : {total:,}")
    print(f"  Assigned a role     : {n_assign:,}  ({100*n_assign/total:.1f}%)")
    print(f"  Unassigned          : {total-n_assign:,}  ({100*(total-n_assign)/total:.1f}%)")

    print("\n── Role counts (assigned authors only) ──────────────────")
    for role in [r for r in ROLE_ORDER if r != "unassigned"]:
        n = (assigned["role"] == role).sum()
        print(f"  {role:<16} {n:>8,}   {100*n/n_assign:5.1f}%")

    print("\n── Role counts by primary discipline ────────────────────")
    pivot = (
        assigned.groupby(["primary_discipline", "role"])
        .size().unstack(fill_value=0)
    )
    # add pct rows
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    for disc in pivot.index:
        print(f"\n  {disc}")
        for role in [c for c in ROLE_ORDER if c in pivot.columns]:
            n = pivot.loc[disc, role]
            p = pct.loc[disc, role]
            print(f"    {role:<16} {n:>7,}   {p:5.1f}%")

    print("\n── Discipline combinations ───────────────────────────────")
    combo_counts = df["disc_combo"].value_counts()
    for combo, n in combo_counts.items():
        print(f"  {combo:<30} {n:>8,}   {100*n/total:5.1f}%")

    print("\n── Blau score statistics by role ────────────────────────")
    print(
        assigned.groupby("role")[["blau_collab", "blau_sim"]]
        .describe().round(3).to_string()
    )
    print()

# ── FIGURES ───────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
        "figure.dpi":       150,
    })

def fig1_role_distribution(df):
    """Overall role distribution — horizontal bar chart."""
    assigned = df[df["role"] != "unassigned"]
    counts   = assigned["role"].value_counts().reindex(
        [r for r in ROLE_ORDER if r != "unassigned"]
    ).fillna(0)
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(
        counts.index, pcts.values,
        color=[ROLE_COLORS[r] for r in counts.index],
        edgecolor="white", height=0.6,
    )
    for bar, (role, pct) in zip(bars, pcts.items()):
        n = int(counts[role])
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%  (n={n:,})", va="center", fontsize=10)

    ax.set_xlabel("% of assigned authors")
    ax.set_title("Role distribution across all disciplines", fontweight="bold", pad=12)
    ax.set_xlim(0, pcts.max() * 1.35)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig

def fig2_roles_by_discipline(df):
    """Stacked 100% bar chart — role mix per discipline."""
    assigned = df[df["role"] != "unassigned"]
    roles    = [r for r in ROLE_ORDER if r != "unassigned"]
    discs    = [d for d in DISC_ORDER if d in assigned["primary_discipline"].unique()]

    pivot = (
        assigned.groupby(["primary_discipline", "role"])
        .size().unstack(fill_value=0)
        .reindex(columns=roles, fill_value=0)
    )
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(pct))
    for role in roles:
        vals = pct[role].values if role in pct.columns else np.zeros(len(pct))
        ax.bar(pct.index, vals, bottom=bottom,
               label=role, color=ROLE_COLORS[role], edgecolor="white", width=0.55)
        bottom += vals

    ax.set_ylabel("% of authors")
    ax.set_title("Role distribution by primary discipline", fontweight="bold", pad=12)
    ax.set_ylim(0, 115)
    fig.subplots_adjust(top=0.78)
    fig.legend(
        [plt.Rectangle((0,0),1,1,color=ROLE_COLORS[r]) for r in roles],
        [r.capitalize() for r in roles],
        loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.97),
        framealpha=0.9, fontsize=9
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()
    return fig

def fig3_blau_distributions(df):
    """KDE of Blau scores split by role for both layers."""
    assigned = df[(df["role"] != "unassigned") &
                  df["blau_collab"].notna() & df["blau_sim"].notna()]
    roles    = [r for r in ROLE_ORDER if r != "unassigned"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, col, label in zip(
        axes,
        ["blau_collab", "blau_sim"],
        ["Collaboration layer (Blau)", "Similarity layer (Blau)"]
    ):
        for role in roles:
            sub = assigned[assigned["role"] == role][col].dropna()
            if len(sub) < 10:
                continue
            sns.kdeplot(sub, ax=ax, label=role, color=ROLE_COLORS[role],
                        fill=True, alpha=0.15, linewidth=1.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Blau score distributions by role", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig

def fig4_discipline_combinations(df):
    """Discipline span by role — single-discipline authors excluded."""
    assigned = df[(df["role"] != "unassigned") & (df["n_disciplines"] > 1)]
    roles    = [r for r in ROLE_ORDER if r != "unassigned"]
    n_discs  = sorted(assigned["n_disciplines"].unique())

    fig, ax = plt.subplots(figsize=(7, 4))
    x      = np.arange(len(n_discs))
    width  = 0.8 / len(roles)

    for i, role in enumerate(roles):
        sub    = assigned[assigned["role"] == role]
        counts = sub["n_disciplines"].value_counts().reindex(n_discs, fill_value=0)
        pcts   = counts / len(assigned) * 100
        ax.bar(x + i * width, pcts.values, width=width * 0.9,
               label=role, color=ROLE_COLORS[role], edgecolor="white")

    ax.set_xticks(x + width * (len(roles) - 1) / 2)
    ax.set_xticklabels([f"{n} discipline{'s' if n>1 else ''}" for n in n_discs])
    ax.set_ylabel("% of all assigned authors")
    ax.set_title("Discipline span by role", fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    fig.subplots_adjust(top=0.78)
    fig.legend(
        [plt.Rectangle((0,0),1,1,color=ROLE_COLORS[r]) for r in roles],
        [r.capitalize() for r in roles],
        loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.97),
        framealpha=0.9, fontsize=9
    )
    fig.tight_layout()
    return fig

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_FILE}...")
    df = load(INPUT_FILE)

    print_summary(df)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    set_style()

    figs = [
        ("fig1_role_distribution",    fig1_role_distribution(df)),
        ("fig2_roles_by_discipline",  fig2_roles_by_discipline(df)),
        ("fig3_blau_distributions",   fig3_blau_distributions(df)),
        ("fig4_discipline_combinations", fig4_discipline_combinations(df)),
    ]

    for name, fig in figs:
        path = os.path.join(FIGURE_DIR, f"{name}.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)

    print("\nDone.")

if __name__ == "__main__":
    main()
