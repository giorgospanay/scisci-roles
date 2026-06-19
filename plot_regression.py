import os
"""
Plots and tables for brokerage role regression analysis.

Requires:
    obj/brokerage_roles.csv
    obj/author_stats.csv

Outputs (saved to current directory):
    table_descriptive.csv        : descriptive stats by role
    table_regression.txt         : formatted regression table
    fig_violin_impact.png        : impact distributions by role
    fig_partial_blau.png         : partial regression: Blau scores vs impact

Usage:
    python plot_regression.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.formula.api as smf

# ── CONFIG ────────────────────────────────────────────────────────────────────

ROLES_FILE  = "obj/brokerage_roles.csv"
STATS_FILE  = "obj/author_stats.csv"
OUTPUT_DIR  = "figures"
MIN_PAPERS  = 3

ROLE_ORDER  = ["coordinator", "gatekeeper", "representative", "liaison"]
ROLE_LABELS = {
    "coordinator":    "Coordinator",
    "gatekeeper":     "Gatekeeper",
    "representative": "Representative",
    "liaison":        "Liaison",
}
ROLE_COLORS = {
    "coordinator":    "#4C72B0",
    "gatekeeper":     "#DD8452",
    "representative": "#55A868",
    "liaison":        "#C44E52",
}

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    roles = pd.read_csv(ROLES_FILE, index_col=0)
    roles.columns = [
        "primary_discipline", "all_disciplines",
        "blau_collab", "blau_sim",
        "collab_diverse", "sim_diverse", "role"
    ]
    stats = pd.read_csv(STATS_FILE, dtype={"author_id": str})

    roles.index = roles.index.astype(str)
    stats["author_id"] = stats["author_id"].astype(str)

    df = roles.merge(stats, left_index=True, right_on="author_id", how="inner")
    df = df[df["role"].isin(ROLE_ORDER)]
    df = df[df["works_count"] >= MIN_PAPERS]
    df = df[df["h_index"].notna() & df["total_citations"].notna()]

    df["log_h"]      = np.log1p(df["h_index"])
    df["log_total"]  = np.log1p(df["total_citations"])
    df["log_median"] = np.log1p(df["median_citations"])
    df["log_works"]  = np.log1p(df["works_count"])
    df["role"]       = pd.Categorical(df["role"], categories=ROLE_ORDER)

    print(f"  Sample: {len(df):,} authors")
    return df


# ── DESCRIPTIVE STATS TABLE ───────────────────────────────────────────────────

def table_descriptive(df):
    print("Computing descriptive stats table...")
    vars_ = {
        "h_index":          "h-index",
        "total_citations":  "Total citations",
        "median_citations": "Median cit. / paper",
        "works_count":      "Works count",
        "career_age":       "Career age (years)",
        "mean_team_size":   "Mean team size",
        "blau_collab":      "Blau (collab)",
        "blau_sim":         "Blau (similarity)",
    }

    rows = []
    for var, label in vars_.items():
        for role in ROLE_ORDER:
            sub = df[df["role"] == role][var].dropna()
            rows.append({
                "Variable": label,
                "Role":     ROLE_LABELS[role],
                "N":        f"{len(sub):,}",
                "Mean":     f"{sub.mean():.2f}",
                "SD":       f"{sub.std():.2f}",
                "Median":   f"{sub.median():.2f}",
            })

    tbl = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tbl.to_csv(os.path.join(OUTPUT_DIR, "table_descriptive.csv"), index=False)
    print(f"  Saved {os.path.join(OUTPUT_DIR, 'table_descriptive.csv')}")
    return tbl


# ── REGRESSION TABLE ──────────────────────────────────────────────────────────

def table_regression(df):
    print("Running regressions for table...")
    formula_base = (
        "{outcome} ~ C(role) + log_works + career_age + mean_team_size"
        " + C(primary_discipline)"
    )
    outcomes = [
        ("log_h",      "log(h-index)"),
        ("log_total",  "log(total cit.)"),
        ("log_median", "log(median cit.)"),
    ]
    models = {}
    for outcome, _ in outcomes:
        models[outcome] = smf.ols(
            formula_base.format(outcome=outcome), data=df
        ).fit(cov_type="HC3")

    # Role params only (discipline FE suppressed for brevity)
    role_params = [
        ("C(role)[T.gatekeeper]",     "Gatekeeper"),
        ("C(role)[T.representative]", "Representative"),
        ("C(role)[T.liaison]",        "Liaison"),
        ("log_works",                 "log(works count)"),
        ("career_age",                "Career age"),
        ("mean_team_size",            "Mean team size"),
    ]

    lines = []
    header = f"{'':28s}" + "".join(f"  {l:>18s}" for _, l in outcomes)
    lines.append(header)
    lines.append("-" * (28 + 21 * len(outcomes)))

    for param, label in role_params:
        row = f"{label:28s}"
        for outcome, _ in outcomes:
            m = models[outcome]
            if param not in m.params:
                row += f"  {'':>18s}"
                continue
            coef = m.params[param]
            se   = m.bse[param]
            pval = m.pvalues[param]
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            row += f"  {coef:>8.3f}{stars:<3s} ({se:.3f})  "
        lines.append(row)

    lines.append("-" * (28 + 21 * len(outcomes)))
    lines.append(f"{'Discipline FE':28s}" + "  " + "  ".join(["Yes"] * len(outcomes)))
    lines.append(f"{'N':28s}" + "".join(
        f"  {int(models[o].nobs):>18,}" for o, _ in outcomes
    ))
    lines.append(f"{'R²':28s}" + "".join(
        f"  {models[o].rsquared:>18.3f}" for o, _ in outcomes
    ))
    lines.append(f"{'Adj. R²':28s}" + "".join(
        f"  {models[o].rsquared_adj:>18.3f}" for o, _ in outcomes
    ))
    lines.append("\nNote: HC3 robust SE in parentheses. * p<0.05  ** p<0.01  *** p<0.001")
    lines.append("Reference category: Coordinator.")

    with open(os.path.join(OUTPUT_DIR, "table_regression.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {os.path.join(OUTPUT_DIR, 'table_regression.txt')}")
    return models


# ── FIG 1: VIOLIN PLOT ────────────────────────────────────────────────────────

def fig_violin(df):
    print("Plotting violin figure...")
    outcomes = [
        ("log_h",      "log(h-index)"),
        ("log_total",  "log(total citations)"),
        ("log_median", "log(median cit. / paper)"),
    ]

    fig = plt.figure(figsize=(14, 5))
    # Leave room at top for legend
    gs = gridspec.GridSpec(1, 3, figure=fig, top=0.78)

    for i, (outcome, label) in enumerate(outcomes):
        ax = fig.add_subplot(gs[i])
        data_by_role = [
            df[df["role"] == role][outcome].dropna().values
            for role in ROLE_ORDER
        ]
        parts = ax.violinplot(
            data_by_role, positions=range(len(ROLE_ORDER)),
            showmedians=True, showextrema=False
        )
        for j, (body, role) in enumerate(zip(parts["bodies"], ROLE_ORDER)):
            body.set_facecolor(ROLE_COLORS[role])
            body.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(len(ROLE_ORDER)))
        ax.set_xticklabels(
            [ROLE_LABELS[r] for r in ROLE_ORDER],
            rotation=20, ha="right", fontsize=9
        )
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Value" if i == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Legend on top
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ROLE_COLORS[r], alpha=0.75)
        for r in ROLE_ORDER
    ]
    fig.legend(
        handles, [ROLE_LABELS[r] for r in ROLE_ORDER],
        loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.97),
        framealpha=0.9, fontsize=10
    )
    fig.suptitle(
        "Citation impact distributions by brokerage role",
        fontweight="bold", y=1.04
    )

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_violin_impact.png"), bbox_inches="tight", dpi=150)
    print(f"  Saved {os.path.join(OUTPUT_DIR, 'fig_violin_impact.png')}")
    plt.close(fig)


# ── FIG 2: PARTIAL REGRESSION BLAU VS IMPACT ─────────────────────────────────

def fig_partial_blau(df):
    """
    Partial regression plots: Blau scores vs log(h-index),
    after partialling out controls. One panel per layer.
    Uses a sample for speed (scatter at 14M points is unreadable anyway).
    """
    print("Plotting partial regression figure...")

    SAMPLE = 50_000
    sample = df.sample(min(SAMPLE, len(df)), random_state=42)

    formula_controls = (
        "log_h ~ log_works + career_age + mean_team_size + C(primary_discipline)"
    )
    # Residualize outcome and each Blau score on controls
    resid_y       = smf.ols(formula_controls, data=sample).fit().resid
    resid_collab  = smf.ols(
        "blau_collab ~ log_works + career_age + mean_team_size + C(primary_discipline)",
        data=sample
    ).fit().resid
    resid_sim     = smf.ols(
        "blau_sim ~ log_works + career_age + mean_team_size + C(primary_discipline)",
        data=sample
    ).fit().resid

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    fig.subplots_adjust(top=0.78)

    panels = [
        (resid_collab, "Blau (collaboration layer)", sample["role"]),
        (resid_sim,    "Blau (similarity layer)",    sample["role"]),
    ]

    for ax, (resid_x, xlabel, roles_s) in zip(axes, panels):
        for role in ROLE_ORDER:
            mask = roles_s == role
            ax.scatter(
                resid_x[mask], resid_y[mask],
                color=ROLE_COLORS[role], alpha=0.15,
                s=4, label=ROLE_LABELS[role], rasterized=True
            )
        # Overall regression line
        m, b = np.polyfit(resid_x, resid_y, 1)
        xr = np.linspace(resid_x.min(), resid_x.max(), 200)
        ax.plot(xr, m * xr + b, color="black", linewidth=1.5, linestyle="--")

        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_xlabel(f"Residual {xlabel}", fontsize=10)
        ax.set_ylabel("Residual log(h-index)" if ax == axes[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, linestyle="--")

    # Legend on top
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=ROLE_COLORS[r], markersize=8)
        for r in ROLE_ORDER
    ]
    fig.legend(
        handles, [ROLE_LABELS[r] for r in ROLE_ORDER],
        loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.97),
        framealpha=0.9, fontsize=10
    )
    fig.suptitle(
        "Partial regression: Blau diversity vs log(h-index)",
        fontweight="bold", y=1.05
    )

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_partial_blau.png"), bbox_inches="tight", dpi=150)
    print(f"  Saved {os.path.join(OUTPUT_DIR, 'fig_partial_blau.png')}")
    plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    table_descriptive(df)
    models = table_regression(df)
    table_regression_latex(models)
    fig_violin(df)
    fig_partial_blau(df)
    print("\nDone.")


if __name__ == "__main__":
    main()


# ── LATEX REGRESSION TABLE ────────────────────────────────────────────────────

def table_regression_latex(models):
    print("Writing LaTeX regression table...")

    outcomes = [
        ("log_h",      r"\log(\text{h-index})"),
        ("log_total",  r"\log(\text{total cit.})"),
        ("log_median", r"\log(\text{median cit.})"),
    ]
    role_params = [
        ("C(role)[T.gatekeeper]",     "Gatekeeper"),
        ("C(role)[T.representative]", "Representative"),
        ("C(role)[T.liaison]",        "Liaison"),
        ("log_works",                 r"$\log(\text{works count})$"),
        ("career_age",                "Career age"),
        ("mean_team_size",            "Mean team size"),
    ]

    def fmt(coef, se, pval):
        stars = (
            "^{***}" if pval < 0.001 else
            "^{**}"  if pval < 0.01  else
            "^{*}"   if pval < 0.05  else ""
        )
        return f"${coef:.3f}{stars}$ & $({se:.3f})$"

    col_spec = "l" + "rr" * len(outcomes)
    header_top = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{${l}$}}" for _, l in outcomes
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(outcomes))
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Brokerage roles and citation impact (OLS). Reference category: Coordinator.}",
        r"\label{tab:regression}",
        r"\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f" & {header_top} \\\\",
        cmidrules,
        " & " + " & ".join([r"\textit{Coef.} & \textit{(SE)}"] * len(outcomes)) + r" \\",
        r"\midrule",
    ]

    # Role dummies — bold separator
    lines.append(r"\multicolumn{" + str(1 + 2*len(outcomes)) + r"}{l}{\textit{Role (ref: Coordinator)}} \\")
    for param, label in role_params[:3]:
        row_parts = []
        for outcome, _ in outcomes:
            m = models[outcome]
            if param not in m.params:
                row_parts.append(" & ")
                continue
            row_parts.append(fmt(m.params[param], m.bse[param], m.pvalues[param]))
        lines.append(f"\\quad {label} & " + " & ".join(row_parts) + r" \\")

    # Controls
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(1 + 2*len(outcomes)) + r"}{l}{\textit{Controls}} \\")
    for param, label in role_params[3:]:
        row_parts = []
        for outcome, _ in outcomes:
            m = models[outcome]
            if param not in m.params:
                row_parts.append(" & ")
                continue
            row_parts.append(fmt(m.params[param], m.bse[param], m.pvalues[param]))
        lines.append(f"\\quad {label} & " + " & ".join(row_parts) + r" \\")

    # Footer
    lines += [
        r"\midrule",
        "\\quad Discipline FE & " + " & ".join(["\\multicolumn{2}{c}{Yes}"] * len(outcomes)) + r" \\",
        "\\quad $N$ & " + " & ".join(
            [f"\\multicolumn{{2}}{{c}}{{{int(models[o].nobs):,}}}" for o, _ in outcomes]
        ) + r" \\",
        "\\quad $R^2$ & " + " & ".join(
            [f"\\multicolumn{{2}}{{c}}{{{models[o].rsquared:.3f}}}" for o, _ in outcomes]
        ) + r" \\",
        "\\quad Adj.\\ $R^2$ & " + " & ".join(
            [f"\\multicolumn{{2}}{{c}}{{{models[o].rsquared_adj:.3f}}}" for o, _ in outcomes]
        ) + r" \\",
        r"\bottomrule",
        r"\multicolumn{" + str(1 + 2*len(outcomes)) + r"}{l}{\footnotesize HC3 robust SE. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path = os.path.join(OUTPUT_DIR, "table_regression.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")
