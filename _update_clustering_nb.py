"""Rewrite sections 4-6 of 04_user_clustering.ipynb per the clustering fix plan."""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/04_user_clustering.ipynb")


def src(text: str) -> list[str]:
    """Notebook source as list of lines ending with \\n (except possibly last)."""
    if not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)
    return lines


cells: dict[int, tuple[str, str]] = {}

cells[0] = (
    "markdown",
    """# User clustering and personas

Split users 80/20, compute train-only movie `rating_std`, build preference-weighted user profiles (`rating - user_mean`), apply relative genre affinity, compare Full vs Genre-only K-Means (PCA 90%), choose `k` in 4–6 by interpretability, and save persona assignments.
""",
)

cells[2] = (
    "code",
    """from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path("..")
PROCESSED = ROOT / "data" / "processed"

CHUNKSIZE = 2_000_000
RANDOM_STATE = 42
SILHOUETTE_MAX_USERS = 50_000
MIN_USER_RATINGS = 5
GENRE_MEAN_CLIP = 1e-6
CANDIDATE_K = [4, 5, 6]
NICHE_GENRES = [
    "genre_horror",
    "genre_comedy",
    "genre_romance",
    "genre_action",
    "genre_drama",
    "genre_thriller",
    "genre_science_fiction",
    "genre_documentary",
]

movies = pd.read_csv(PROCESSED / "movies_features.csv")
feature_movie_ids = set(movies["netflix_movie_id"].astype(int))

GENRE_COLS = [c for c in movies.columns if c.startswith("genre_")]
PROFILE_BASE_COLS = GENRE_COLS + [
    "budget_log",
    "revenue_log",
    "runtime",
    "movie_age",
    "lang_en",
]

print(f"Feature movies: {len(feature_movie_ids):,}")
print(f"Profile base columns: {len(PROFILE_BASE_COLS)}")
movies.head(2)
""",
)

cells[8] = (
    "markdown",
    """## 4. Clustering feature matrix (per user)

Two-pass profiles: (1) per-user mean rating, (2) preference-weighted mean of movie features with `w = rating - user_mean`, normalized by `sum(|w|)`. Drop users with fewer than 5 ratings or `sum(|w|) == 0`. Then overwrite `genre_*` with relative affinity vs train means. Production columns and `rating_std` stay as preference-weighted means.
""",
)

cells[9] = (
    "code",
    '''PROFILE_COLS = PROFILE_BASE_COLS + ["rating_std"]
movie_feat = movies_with_std.set_index("netflix_movie_id")[PROFILE_COLS]
all_split_users = train_user_set | test_user_set


def accumulate_user_means(users: set) -> tuple[pd.Series, pd.Series]:
    """Pass 1: rating count and sum per customer_id."""
    count = None
    total = None
    for chunk in pd.read_csv(ratings_path, usecols=usecols, chunksize=CHUNKSIZE):
        chunk = chunk[
            chunk["netflix_movie_id"].isin(feature_movie_ids)
            & chunk["customer_id"].isin(users)
        ]
        if chunk.empty:
            continue
        part = chunk.groupby("customer_id", sort=False)["rating"].agg(["count", "sum"])
        if count is None:
            count = part["count"]
            total = part["sum"]
        else:
            count = count.add(part["count"], fill_value=0)
            total = total.add(part["sum"], fill_value=0)
    user_mean = total / count
    return count.astype(int), user_mean


print("Pass 1: user means...")
user_n_ratings, user_mean = accumulate_user_means(all_split_users)
print(f"Users with means: {len(user_mean):,}")


def accumulate_preference_profiles(
    user_sets: dict[str, set], user_mean: pd.Series
) -> dict[str, pd.DataFrame]:
    """Pass 2: sum(w * features) / sum(|w|) with w = rating - user_mean."""
    weighted_sum = {name: None for name in user_sets}
    abs_w_sum = {name: None for name in user_sets}
    all_users = set().union(*user_sets.values())

    for chunk in pd.read_csv(ratings_path, usecols=usecols, chunksize=CHUNKSIZE):
        chunk = chunk[
            chunk["netflix_movie_id"].isin(feature_movie_ids)
            & chunk["customer_id"].isin(all_users)
        ]
        if chunk.empty:
            continue

        feats = chunk.join(movie_feat, on="netflix_movie_id")
        feats = feats.join(user_mean.rename("user_mean"), on="customer_id")
        w = feats["rating"].astype(float) - feats["user_mean"]
        weighted = feats[PROFILE_COLS].multiply(w, axis=0)
        weighted["customer_id"] = feats["customer_id"].values
        weighted["_abs_w"] = w.abs().values

        for name, users in user_sets.items():
            part_users = weighted[weighted["customer_id"].isin(users)]
            if part_users.empty:
                continue
            part = part_users.groupby("customer_id", sort=False).sum()
            if weighted_sum[name] is None:
                weighted_sum[name] = part[PROFILE_COLS]
                abs_w_sum[name] = part["_abs_w"]
            else:
                weighted_sum[name] = weighted_sum[name].add(
                    part[PROFILE_COLS], fill_value=0.0
                )
                abs_w_sum[name] = abs_w_sum[name].add(part["_abs_w"], fill_value=0.0)

    out = {}
    for name in user_sets:
        denom = abs_w_sum[name].replace(0, np.nan)
        profiles = weighted_sum[name].div(denom, axis=0)
        profiles.index.name = "customer_id"
        out[name] = profiles
    return out


print("Pass 2: preference-weighted profiles...")
profiles = accumulate_preference_profiles(
    {"train": train_user_set, "test": test_user_set}, user_mean
)
profiles_train = profiles["train"]
profiles_test = profiles["test"]

# Drop unstable users: too few ratings or zero preference mass
keep_train = profiles_train.index.intersection(
    user_n_ratings[user_n_ratings >= MIN_USER_RATINGS].index
)
keep_test = profiles_test.index.intersection(
    user_n_ratings[user_n_ratings >= MIN_USER_RATINGS].index
)
profiles_train = profiles_train.loc[keep_train].dropna(how="any")
profiles_test = profiles_test.loc[keep_test].dropna(how="any")
# sum(|w|)==0 already yields NaN rows via replace(0, nan)

print(f"Train profiles: {profiles_train.shape}")
print(f"Test profiles:  {profiles_test.shape}")
profiles_train.head(2)
''',
)

cells[10] = (
    "code",
    """# Relative genre affinity (in-place overwrite; train means only)
genre_global = profiles_train[GENRE_COLS].mean().clip(lower=GENRE_MEAN_CLIP)
profiles_train = profiles_train.copy()
profiles_test = profiles_test.copy()
profiles_train[GENRE_COLS] = profiles_train[GENRE_COLS].div(genre_global, axis=1)
profiles_test[GENRE_COLS] = profiles_test[GENRE_COLS].div(genre_global, axis=1)

print("Genre global means (clipped) — sample:")
print(genre_global[[c for c in NICHE_GENRES if c in genre_global.index]].round(4))
print("Relative genre means (train) ≈ 1 by construction:")
print(profiles_train[GENRE_COLS].mean().describe().round(4))
""",
)

# Insert new cells after 10 by expanding the notebook: we'll replace 11+ and insert.
# Strategy: set cells 11+ to new content and insert extra cells as needed.

cells[11] = (
    "markdown",
    """## 5. Scale, PCA, and A/B clustering

StandardScaler (train-fit) then PCA retaining 90% variance. Compare **Full** (`PROFILE_COLS`) vs **Genre-only** (`GENRE_COLS`) with the same K-Means protocol. Prefer `k` in {4,5,6}; silhouette on a ≤50k subsample guides the pick within that range.
""",
)

cells[12] = (
    "code",
    """def scale_and_pca(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]
) -> dict:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[cols])
    X_test_scaled = scaler.transform(test_df[cols])
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    scaled_train = pd.DataFrame(X_train_scaled, index=train_df.index, columns=cols)
    scaled_test = pd.DataFrame(X_test_scaled, index=test_df.index, columns=cols)
    return {
        "scaler": scaler,
        "pca": pca,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "scaled_train": scaled_train,
        "scaled_test": scaled_test,
        "n_components": int(pca.n_components_),
        "explained": float(pca.explained_variance_ratio_.sum()),
    }


def silhouette_subsample(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if n > SILHOUETTE_MAX_USERS:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=SILHOUETTE_MAX_USERS, replace=False)
        return X[idx]
    return X


def sweep_k(X: np.ndarray, ks: list[int]) -> tuple[list[float], list[float]]:
    X_sil = silhouette_subsample(X)
    silhouettes, inertias = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_sil)
        silhouettes.append(silhouette_score(X_sil, labels))
        inertias.append(km.inertia_)
        print(f"  k={k}: silhouette={silhouettes[-1]:.4f}, inertia={inertias[-1]:,.0f}")
    return silhouettes, inertias


def cluster_summary(
    profiles: pd.DataFrame, labels: np.ndarray, include_production: bool
) -> pd.DataFrame:
    summary = profiles.copy()
    summary["persona_id"] = labels
    top_genres = (
        summary.groupby("persona_id")[GENRE_COLS]
        .mean()
        .apply(lambda row: row.nlargest(3).index.tolist(), axis=1)
    )
    mean_aggs = {}
    niche_present = [c for c in NICHE_GENRES if c in summary.columns]
    for c in niche_present:
        mean_aggs[f"mean_{c}"] = (c, "mean")
    if include_production:
        for c, name in [
            ("budget_log", "mean_budget_log"),
            ("revenue_log", "mean_revenue_log"),
            ("runtime", "mean_runtime"),
            ("movie_age", "mean_movie_age"),
            ("lang_en", "mean_lang_en"),
            ("rating_std", "mean_rating_std"),
        ]:
            if c in summary.columns:
                mean_aggs[name] = (c, "mean")
    out = summary.groupby("persona_id").agg(**mean_aggs)
    out.insert(0, "n_users", summary.groupby("persona_id").size())
    out["top_genres"] = top_genres
    out["pct"] = (out["n_users"] / out["n_users"].sum() * 100).round(1)
    return out


VARIANTS = {
    "full": PROFILE_COLS,
    "genre_only": GENRE_COLS,
}

prep = {}
for name, cols in VARIANTS.items():
    print(f"\\n=== Preparing variant: {name} ({len(cols)} cols) ===")
    prep[name] = scale_and_pca(profiles_train, profiles_test, cols)
    print(
        f"PCA components: {prep[name]['n_components']} "
        f"(explained={prep[name]['explained']:.3f})"
    )
""",
)

cells[13] = (
    "code",
    """ks = list(range(2, 11))
sweep_results = {}

for name in VARIANTS:
    print(f"\\n=== Silhouette sweep: {name} ===")
    sils, inerts = sweep_k(prep[name]["X_train_pca"], ks)
    sweep_results[name] = {"ks": ks, "silhouettes": sils, "inertias": inerts}
    best_sil_k = ks[int(np.argmax(sils))]
    print(f"Max-silhouette k (reference only): {best_sil_k}")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for i, name in enumerate(VARIANTS):
    r = sweep_results[name]
    axes[0, i].plot(r["ks"], r["silhouettes"], marker="o")
    axes[0, i].set_title(f"Silhouette — {name}")
    axes[0, i].set_xlabel("k")
    axes[0, i].set_ylabel("Mean silhouette")
    axes[1, i].plot(r["ks"], r["inertias"], marker="o")
    axes[1, i].set_title(f"Elbow — {name}")
    axes[1, i].set_xlabel("k")
    axes[1, i].set_ylabel("Inertia")
plt.tight_layout()
plt.show()
""",
)

cells[14] = (
    "code",
    """# Fit candidate k in {4,5,6} for both variants; summarize train personas
candidate_fits = {}

for name in VARIANTS:
    print(f"\\n=== Candidate fits: {name} ===")
    candidate_fits[name] = {}
    X_tr = prep[name]["X_train_pca"]
    for k in CANDIDATE_K:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_tr)
        # silhouette on subsample for this k
        X_sil = silhouette_subsample(X_tr)
        # map: refit labels on subsample for score
        km_sil = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        sil = silhouette_score(X_sil, km_sil.fit_predict(X_sil))
        include_prod = name == "full"
        summary = cluster_summary(profiles_train, labels, include_production=include_prod)
        candidate_fits[name][k] = {
            "kmeans": km,
            "labels": labels,
            "silhouette": sil,
            "summary": summary,
        }
        print(f"\\nk={k}, silhouette={sil:.4f}")
        display_cols = ["n_users", "pct", "top_genres"] + [
            c for c in summary.columns if c.startswith("mean_genre_")
        ][:4]
        print(summary[display_cols].to_string())
""",
)

cells[15] = (
    "code",
    '''def pick_k_for_variant(name: str) -> int:
    """Best silhouette among CANDIDATE_K (interpretability range)."""
    scores = {k: candidate_fits[name][k]["silhouette"] for k in CANDIDATE_K}
    return max(scores, key=scores.get)


def min_cluster_pct(summary: pd.DataFrame) -> float:
    return float(summary["pct"].min())


# Decision: prefer clearer niche separation, then silhouette, then Full if tied
chosen = {}
for name in VARIANTS:
    k = pick_k_for_variant(name)
    chosen[name] = {
        "k": k,
        "silhouette": candidate_fits[name][k]["silhouette"],
        "summary": candidate_fits[name][k]["summary"],
        "min_pct": min_cluster_pct(candidate_fits[name][k]["summary"]),
    }
    print(
        f"{name}: preferred k={k}, "
        f"silhouette={chosen[name]['silhouette']:.4f}, "
        f"min cluster pct={chosen[name]['min_pct']:.1f}"
    )

# Heuristic: count how often niche genres appear in top_genres (excl. drama-only blandness)
def niche_score(summary: pd.DataFrame) -> float:
    niche = {
        "genre_horror",
        "genre_documentary",
        "genre_animation",
        "genre_western",
        "genre_war",
        "genre_science_fiction",
        "genre_family",
        "genre_fantasy",
        "genre_romance",
        "genre_comedy",
        "genre_action",
        "genre_thriller",
    }
    score = 0.0
    tops = summary["top_genres"].tolist()
    # diversity of first-ranked genre across clusters
    firsts = [t[0] if t else None for t in tops]
    score += len(set(firsts))
    for t in tops:
        score += sum(1 for g in t if g in niche)
    # penalize very tiny clusters
    if summary["pct"].min() < 5:
        score -= 1.0
    return score


for name in VARIANTS:
    chosen[name]["niche_score"] = niche_score(chosen[name]["summary"])
    print(f"{name}: niche_score={chosen[name]['niche_score']:.1f}")

full_ns = chosen["full"]["niche_score"]
genre_ns = chosen["genre_only"]["niche_score"]
full_sil = chosen["full"]["silhouette"]
genre_sil = chosen["genre_only"]["silhouette"]

if genre_ns > full_ns + 0.5:
    winning_variant = "genre_only"
    win_reason = f"clearer personas (niche_score {genre_ns:.1f} > {full_ns:.1f})"
elif full_ns > genre_ns + 0.5:
    winning_variant = "full"
    win_reason = f"clearer personas (niche_score {full_ns:.1f} > {genre_ns:.1f})"
elif genre_sil > full_sil + 0.01 and chosen["genre_only"]["min_pct"] >= 5:
    winning_variant = "genre_only"
    win_reason = f"higher silhouette ({genre_sil:.4f} > {full_sil:.4f}) with balanced sizes"
else:
    winning_variant = "full"
    win_reason = "tie / default to Full (proposal: metadata + genres)"

best_k = chosen[winning_variant]["k"]
print(f"\\nWinner: {winning_variant} with k={best_k} — {win_reason}")
''',
)

cells[16] = (
    "markdown",
    """## 5b. Final fit on winning variant

Refit K-Means on all train PCA rows for the winning feature set and `k`; assign test with `.predict`. Cluster summary uses relative genre profiles (and production means if Full won).
""",
)

cells[17] = (
    "code",
    """win = prep[winning_variant]
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
train_labels = kmeans.fit_predict(win["X_train_pca"])
test_labels = kmeans.predict(win["X_test_pca"])

personas_train = pd.DataFrame(
    {
        "customer_id": profiles_train.index,
        "persona_id": train_labels,
        "split": "train",
    }
)
personas_test = pd.DataFrame(
    {
        "customer_id": profiles_test.index,
        "persona_id": test_labels,
        "split": "test",
    }
)
user_personas = pd.concat([personas_train, personas_test], ignore_index=True)

print(user_personas["persona_id"].value_counts().sort_index())
print(f"Winning variant: {winning_variant}, k={best_k}")

final_summary = cluster_summary(
    profiles_train,
    train_labels,
    include_production=(winning_variant == "full"),
)
final_summary
""",
)

# Cells 18 and 19 for save section — need to insert beyond current 17
# Current notebook has 18 cells (0-17). We'll expand.

cells[18] = (
    "markdown",
    """## 6. Save clustering outputs

Write `user_personas.csv`, scaled profiles for the winning variant, and `clustering_meta.json`. Raw data under `data/raw/` is untouched.
""",
)

cells[19] = (
    "code",
    """import json

user_personas.to_csv(PROCESSED / "user_personas.csv", index=False)

scaled_all = pd.concat(
    [
        win["scaled_train"].assign(split="train").reset_index(),
        win["scaled_test"].assign(split="test").reset_index(),
    ],
    ignore_index=True,
)
scaled_all.to_csv(PROCESSED / "user_profiles_scaled.csv", index=False)

meta = {
    "winning_variant": winning_variant,
    "win_reason": win_reason,
    "best_k": int(best_k),
    "pca_n_components": int(win["n_components"]),
    "pca_explained_variance": float(win["explained"]),
    "n_train_users": int(len(profiles_train)),
    "n_test_users": int(len(profiles_test)),
    "min_user_ratings": MIN_USER_RATINGS,
    "candidate_k_silhouettes": {
        name: {str(k): float(candidate_fits[name][k]["silhouette"]) for k in CANDIDATE_K}
        for name in VARIANTS
    },
    "niche_scores": {name: float(chosen[name]["niche_score"]) for name in VARIANTS},
}
with open(PROCESSED / "clustering_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Saved user_personas.csv: {user_personas.shape}")
print(f"Saved user_profiles_scaled.csv: {scaled_all.shape}")
print(f"Saved clustering_meta.json: {meta}")
print("Done.")
""",
)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    existing = nb["cells"]

    # Keep cells 0-7 structure; replace 0,2 and from 8 onward
    # Build new cell list: 0-7 from existing (with 0,2 replaced), then 8-19 from cells dict
    new_cells = []
    for i in range(8):
        if i in cells:
            kind, text = cells[i]
            new_cells.append(
                {
                    "cell_type": kind,
                    "metadata": {},
                    "source": src(text),
                    **(
                        {"outputs": [], "execution_count": None}
                        if kind == "code"
                        else {}
                    ),
                }
            )
        else:
            # preserve markdown/code cells 1,3,4,5,6,7 but clear outputs on code
            c = existing[i]
            cell = {
                "cell_type": c["cell_type"],
                "metadata": c.get("metadata", {}),
                "source": c["source"],
            }
            if c["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            new_cells.append(cell)

    for i in range(8, 20):
        kind, text = cells[i]
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": src(text),
        }
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        new_cells.append(cell)

    nb["cells"] = new_cells
    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {NB_PATH} with {len(new_cells)} cells")


if __name__ == "__main__":
    main()
