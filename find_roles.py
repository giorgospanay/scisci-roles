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

Physics similarity layer is optional — authors without it are kept in
the output as 'unassigned' until the layer becomes available.
"""

import os
import time
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR     = "/N/slate/gpanayio/scisci-roles"
EDGELIST_DIR = "/N/slate/gpanayio/scisci-gatekeepers/obj"
OUTPUT_DIR   = os.path.join(BASE_DIR, "obj")

DISCIPLINES = ["CS", "Biology", "Math", "Physics"]

# Flip Physics to True once your queue finishes
SIMILARITY_AVAILABLE = {
	"CS":      True,
	"Biology": True,
	"Math":    True,
	"Physics": False,
}

MIN_COLLABORATORS = 5   # minimum distinct neighbors in collab layer
MIN_SIM_NEIGHBORS = 5   # minimum distinct neighbors in similarity layer

# None = within-discipline top tercile (recommended)
# or set a fixed float e.g. 0.3
BLAU_THRESHOLD = None

# ── HELPERS ───────────────────────────────────────────────────────────────────

def log(msg):
	print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def blau_index(s):
	"""Blau's diversity index: 1 - sum(p_i^2). Range [0, 1)."""
	p = s.value_counts(normalize=True)
	return float(1 - (p ** 2).sum())


def load_edgelist(path, sep):
	log(f"  Reading {os.path.basename(path)}...")
	df = pd.read_csv(
		path, sep=sep, header=None,
		names=["src", "dst", "weight"],
		dtype={"src": str, "dst": str, "weight": float},
	)
	log(f"  Done — {len(df):,} edges")
	return df


# ── LOAD ALL LAYERS ───────────────────────────────────────────────────────────

def load_all_layers():
	collab_layers = {}
	sim_layers    = {}

	for disc in DISCIPLINES:
		log(f"Loading {disc} collaboration layer...")
		path = os.path.join(EDGELIST_DIR, f"filtered_collaboration_layer_{disc}.edgelist")
		collab_layers[disc] = load_edgelist(path, sep="\t")

		if SIMILARITY_AVAILABLE[disc]:
			log(f"Loading {disc} similarity layer...")
			path = os.path.join(EDGELIST_DIR, f"filtered_author_similarity_layer_{disc}.edgelist")
			sim_layers[disc] = load_edgelist(path, sep=" ")
		else:
			log(f"Skipping {disc} similarity layer (marked unavailable)")
			sim_layers[disc] = None

	return collab_layers, sim_layers


# ── BUILD AUTHOR → PRIMARY DISCIPLINE MAPPING ─────────────────────────────────

def build_author_discipline(collab_layers):
	"""
	Primary discipline = the layer where the author has the most edges.
	Separately records all disciplines an author appears in.
	"""
	log("Building author → discipline mapping...")
	records = []
	for disc, df in collab_layers.items():
		counts = (
			pd.concat([
				df[["src"]].rename(columns={"src": "author"}),
				df[["dst"]].rename(columns={"dst": "author"}),
			])
			.groupby("author").size().rename("n_edges")
			.reset_index()
		)
		counts["discipline"] = disc
		records.append(counts)
		log(f"  {disc}: {len(counts):,} authors")

	all_records = pd.concat(records, ignore_index=True)

	primary = (
		all_records.sort_values("n_edges", ascending=False)
		.drop_duplicates(subset="author", keep="first")
		[["author", "discipline"]]
		.rename(columns={"discipline": "primary_discipline"})
		.set_index("author")
	)

	all_discs = (
		all_records.groupby("author")["discipline"]
		.apply(set)
		.rename("all_disciplines")
	)

	result = primary.join(all_discs)
	n_cross = (result["all_disciplines"].apply(len) > 1).sum()
	log(f"  Total unique authors: {len(result):,}  ({n_cross:,} appear in >1 discipline)")
	return result


# ── COMPUTE BLAU INDICES ──────────────────────────────────────────────────────

def compute_blau(layers, layer_name, min_neighbors):
	"""
	Stack all edges across disciplines into a long table where each row is
	(author, neighbor_discipline). Blau index = diversity of that distribution.
	"""
	log(f"Computing Blau indices for {layer_name} layer...")

	records = []
	for disc, df in layers.items():
		if df is None:
			continue
		log(f"  Stacking {disc}...")
		for col in ["src", "dst"]:
			side = df[[col]].rename(columns={col: "author"}).copy()
			side["neighbor_disc"] = disc
			records.append(side)

	log("  Concatenating...")
	long = pd.concat(records, ignore_index=True)
	log(f"  Total rows: {len(long):,}")

	log(f"  Filtering to authors with >= {min_neighbors} neighbors...")
	counts = long.groupby("author").size()
	valid  = counts[counts >= min_neighbors].index
	long   = long[long["author"].isin(valid)]
	log(f"  Retained {len(valid):,} authors, {len(long):,} rows")

	log("  Computing Blau index per author...")
	blau = (
		long.groupby("author")["neighbor_disc"]
		.apply(blau_index)
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
		log("  Thresholds: within-discipline top tercile")
		df.loc[has_both, "collab_diverse"] = (
			df[has_both].groupby("primary_discipline")["blau_collab"]
			.transform(lambda x: x > x.quantile(2/3))
		)
		df.loc[has_both, "sim_diverse"] = (
			df[has_both].groupby("primary_discipline")["blau_sim"]
			.transform(lambda x: x > x.quantile(2/3))
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


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
	t0 = time.time()
	log("=== Brokerage role analysis starting ===")

	collab_layers, sim_layers = load_all_layers()
	author_discipline = build_author_discipline(collab_layers)

	blau_collab = compute_blau(collab_layers, "collab", MIN_COLLABORATORS)
	blau_sim    = compute_blau(sim_layers,    "sim",    MIN_SIM_NEIGHBORS)

	results = assign_roles(blau_collab, blau_sim, author_discipline)
	summarize(results)

	os.makedirs(OUTPUT_DIR, exist_ok=True)
	out_path = os.path.join(OUTPUT_DIR, "brokerage_roles.csv")
	log(f"Saving results to {out_path}...")
	results.to_csv(out_path)

	log(f"=== Done in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
	main()