# Handoff — EDA session (movies features + ratings)

## Project

Predict / personalize Netflix movie ratings using TMDB production metadata and user persona clustering.

- Course rules: `docs/00_course_requirements.md`
- Proposal & RQs: `docs/01_proposal.md`
- Agent guide: `AGENTS.md`
- `data/raw/` is immutable; write outputs only under `data/processed/`

## Completed this session

Built and executed [`notebooks/05_eda.ipynb`](notebooks/05_eda.ipynb) on:

- [`data/processed/movies_features.csv`](data/processed/movies_features.csv) (~1,322 movies × 81 cols; previously ~294 with person one-hots)
- [`data/processed/ratings.csv`](data/processed/ratings.csv) (~57M rows)

**Also done:** preprocess film-count redesign in [`notebooks/03_preprocess_movies.ipynb`](notebooks/03_preprocess_movies.ipynb) and EDA refresh to match.

**Deferred:** user personas / user profiles EDA.

**Design choice:** EDA on `movies_features` (model-ready), not `movies_metadata` (overlapping with preprocess diagnostics in notebook 03).

### Notebook sections (current)

1. Continuous histograms: `movie_age`, `runtime`, `budget_log`, `log_roi`, four `*_film_count_log` (`*_scaled` skipped)
2. Binary family bar charts: language, decade, genre, country, company
3. Film-count diagnostics: `describe()` + % zero (missing director / empty billing slot)
4. Per-movie `avg_rating`, `rating_std`, `n_ratings` + Spearman vs features
5. Ratings: global 1–5 histogram; user buckets for count / std / mean

## Key EDA findings (original — pre film-count redesign)

These motivated the preprocess change; person one-hots are no longer in `movies_features`.

- **Director one-hots fail:** ~79% of movies had all `director_*` = 0 (~21% coverage). `MIN_DIRECTOR_FILMS = 5` → 41 sparse director columns.
- **Actor one-hots weak:** ~33% all-zero; 186 sparse columns.
- **`year` and `movie_age`:** perfect collinearity (`corr = -1`). Keep one only → kept `movie_age`.
- **`log_roi`:** defined as `revenue_log - budget_log` (= `log1p(revenue) - log1p(budget)`). Strongest Spearman with `avg_rating` (~0.44), but linearly dependent with budget/revenue logs — do not keep the full triangle for modeling → kept `budget_log` + `log_roi`.
- Genres + continuous production features dominate signal; named director/actor flags barely appeared in top correlations.

## Key EDA findings (after refresh)

- Shape: **1,322 × 81**; no person one-hots; film-count logs nearly fully populated (zeros only for empty `actor_2` / `actor_3` slots, ~0.2% each).
- **`log_roi`** still strongest Spearman with `avg_rating` (~0.44); then `runtime`, `movie_age`, `genre_drama`.
- **`director_film_count_log`** appears in the top correlations with `avg_rating` (~0.17) — usable signal vs the old sparse one-hots.
- With **`rating_std`**: higher `runtime`, `actor_1_film_count_log`, and `director_film_count_log` associate with *lower* polarization; horror/comedy/decade_2000 with higher.
- Decade dummies now included in binary EDA / Spearman (were in CSV before but not plotted).

## Preprocess direction (implemented)

Replaced sparse person one-hots with film-count features in notebook 03:

- `director_film_count_log = log1p(in-dataset director frequency)`; missing → 0
- Actor counts from **full cast** across movies; mapped onto top-3 billing slots as `actor_{1,2,3}_film_count_log`
- Dropped `year` / `year_scaled`; kept `movie_age`
- Kept `budget_log` + `log_roi`; dropped `revenue_log` from model columns

## Likely next steps

1. Align [`notebooks/04_user_clustering.ipynb`](notebooks/04_user_clustering.ipynb): `PROFILE_BASE_COLS` still lists `revenue_log` (missing from current `movies_features`); switch to `log_roi` (and optionally film-count cols) before re-running personas.
2. Re-run clustering / regenerate `user_profiles_scaled.csv` after that fix.
3. Optional later: top-K named-person one-hots + `has_director` if RQ3 needs named effects.

## Important files

| Path | Role |
|------|------|
| `notebooks/03_preprocess_movies.ipynb` | Feature engineering (film counts, encodings) |
| `notebooks/04_user_clustering.ipynb` | Personas, preference-weighted profiles |
| `notebooks/05_eda.ipynb` | EDA (refreshed for 81-col features) |
| `data/processed/movies_features.csv` | Model-ready movie table |
| `data/processed/ratings.csv` | User–movie ratings |
| `data/processed/user_personas.csv` | `customer_id`, `persona_id`, `split` only |
| `data/processed/user_profiles_scaled.csv` | Scaled preference-weighted profiles |

## Working rules for the next agent

- Do not overwrite `data/raw/`.
- Stay within the four research questions; no scope creep.
- Prefer a clear notebook with short section descriptions; comment only non-obvious logic.
- Commit only when the user asks.
