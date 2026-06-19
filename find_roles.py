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

DISCIPLINES = ["CS", "Biology", "Math", "Physics", "Sociology", "Economics", "Linguistics"]

SIMILARITY_AVAILABLE = {
	"CS":          True,
	"Biology":     True,
	"Math":        True,
	"Physics":     True,
	"Sociology":   True,
	"Economics":   True,
	"Linguistics": True,
}

MIN_COLLABORATORS = 5   # minimum distinct neighbors in collab layer
MIN_SIM_NEIGHBORS = 5   # minimum distinct neighbors in similarity layer

# None = within-discipline top tercile (recommended)
# or set a fixed float e.g. 0.3
BLAU_THRESHOLD = None

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


def load_edgelist(path, sep):
	log(f"  Reading {os.path.basename(path)}...")
	df = pd.read_csv(
		path, sep=sep, header=None,
		names=["src", "dst", "weight"],
		dtype={"src": str, "dst": str, "weight": float},
	)
	log(f"  Done — {len(df):,} edges")
	return df


def edgelist_to_neighbor_counts(df, disc):
	"""
	Reduce a full edgelist to a compact table:
		author | neighbor_disc | count
	where count = number of edges to neighbors in that discipline.
	Frees the edgelist immediately after reduction.
	"""
	src = df[["src"]].rename(columns={"src": "author"})
	dst = df[["dst"]].rename(columns={"dst": "author"})
	both = pd.concat([src, dst], ignore_index=True)
	both["neighbor_disc"] = disc
	counts = both.groupby(["author", "neighbor_disc"]).size().reset_index(name="count")
	return counts


# ── BUILD NEIGHBOR COUNTS (one discipline at a time) ─────────────────────────

def build_neighbor_counts(layer_type, min_neighbors):
	"""
	Load each discipline edgelist, reduce to (author, disc, count), free edgelist.
	Returns a single stacked counts table across all disciplines.
	"""
	log(f"Building neighbor counts for {layer_type} layer...")
	parts = []

	for disc in DISCIPLINES:
		if layer_type == "sim" and not SIMILARITY_AVAILABLE.get(disc, False):
			log(f"  Skipping {disc} (similarity unavailable)")
			continue

		if layer_type == "collab":
			path = os.path.join(EDGELIST_DIR, f"filtered_collaboration_layer_{disc}.edgelist")
			sep  = "\t"
		else:
			path = os.path.join(EDGELIST_DIR, f"filtered_author_similarity_layer_{disc}.edgelist")
			sep  = " "

		log(f"  Loading {disc}...")
		df = load_edgelist(path, sep)

		log(f"  Reducing to neighbor counts...")
		counts = edgelist_to_neighbor_counts(df, disc)
		log(f"  {disc}: {len(counts):,} author-discipline pairs")

		parts.append(counts)

		# Free the full edgelist immediately
		del df
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
		.apply(blau_from_counts)
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