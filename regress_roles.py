import os
"""
Standalone regression: roles predicting citation impact.

Usage:
    python regress_roles.py
    python regress_roles.py --roles brokerage_roles.csv --stats author_stats.csv

Outputs (saved to current directory):
    regression_results.txt
    fig_coefplot.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────

ROLES_FILE = "obj/brokerage_roles.csv"
STATS_FILE = "obj/author_stats.csv"
MIN_PAPERS = 3

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_data(roles_path, stats_path):
    print(f"Loading roles from {roles_path}...")
    roles = pd.read_csv(roles_path, index_col=0)
    roles.columns = [
        "primary_discipline", "all_disciplines",
        "blau_collab", "blau_sim",
        "collab_diverse", "sim_diverse", "role"
    ]
    print(f"  {len(roles):,} authors")

    print(f"Loading stats from {stats_path}...")
    stats = pd.read_csv(stats_path, dtype={"author_id": str})
    print(f"  {len(stats):,} authors")

    print("Merging...")
    roles.index = roles.index.astype(str)
    stats["author_id"] = stats["author_id"].astype(str)
    df = roles.merge(stats, left_index=True, right_on="author_id", how="inner")
    df = df[df["role"].isin(["coordinator", "gatekeeper", "representative", "liaison"])]
    df = df[df["works_count"] >= MIN_PAPERS]
    df = df[df["h_index"].notna() & df["total_citations"].notna()]

    print(f"  Regression sample: {len(df):,} authors")
    print(f"  Role counts:\n{df['role'].value_counts().to_string()}")
    return df

# ── REGRESSION ────────────────────────────────────────────────────────────────

def run_regression(df):
    import statsmodels.formula.api as smf

    # Log-transform outcomes and key predictors
    df["log_h"]         = np.log1p(df["h_index"])
    df["log_total"]     = np.log1p(df["total_citations"])
    df["log_median"]    = np.log1p(df["median_citations"])
    df["log_works"]     = np.log1p(df["works_count"])

    # coordinator = reference category
    df["role"] = pd.Categorical(
        df["role"],
        categories=["coordinator", "gatekeeper", "representative", "liaison"]
    )

    outcomes = [
        ("log_h",      "log(h-index)"),
        ("log_total",  "log(total citations)"),
        ("log_median", "log(median citations per paper)"),
    ]

    formula_base = (
        "{outcome} ~ C(role) + log_works + career_age + mean_team_size"
        " + C(primary_discipline)"
    )

    results      = {}
    result_lines = []

    for outcome, label in outcomes:
        print(f"  Fitting: {label}...")
        formula = formula_base.format(outcome=outcome)
        model   = smf.ols(formula, data=df).fit(cov_type="HC3")
        results[outcome] = model

        result_lines.append(f"\n{'='*70}")
        result_lines.append(f"Outcome : {label}")
        result_lines.append(f"N = {int(model.nobs):,}   R² = {model.rsquared:.3f}   "
                            f"Adj. R² = {model.rsquared_adj:.3f}")
        result_lines.append(model.summary2().tables[1].to_string())

    with open("regression_results.txt", "w") as f:
        f.write("\n".join(result_lines))
    print("  Saved regression_results.txt")

    return results, df

# ── COEFFICIENT PLOT ──────────────────────────────────────────────────────────

def plot_coefplot(results):
    outcomes = {
        "log_h":      "log(h-index)",
        "log_total":  "log(total citations)",
        "log_median": "log(median cit. / paper)",
    }
    roles = ["gatekeeper", "representative", "liaison"]
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
        "Role effect on citation impact  (reference: Coordinator)",
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
                capsize=4, linewidth=1.5, markersize=7,
                label=role_labels[role]
            )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.set_yticks(range(len(roles)))
        ax.set_yticklabels([role_labels[r] for r in roles])
        ax.set_xlabel("Coefficient vs coordinator")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(os.path.join("figures", "fig_coefplot.png"), bbox_inches="tight", dpi=150)
    print("  Saved figures/fig_coefplot.png")
    plt.close(fig)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", default=ROLES_FILE)
    parser.add_argument("--stats", default=STATS_FILE)
    args = parser.parse_args()

    df = load_data(args.roles, args.stats)

    print("\nRunning regressions...")
    results, df = run_regression(df)

    print("Plotting coefficients...")
    plot_coefplot(results)

    print("\nDone. Outputs saved to current directory.")

if __name__ == "__main__":
    main()
