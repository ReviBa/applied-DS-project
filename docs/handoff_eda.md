# Handoff — movie cleaning, EDA, and features

## Project

Predict / personalize Netflix movie ratings using TMDB production metadata and user persona clustering.

- Course rules: `docs/00_course_requirements.md`
- Proposal & RQs: `docs/01_proposal.md`
- Agent guide: `AGENTS.md`
- `data/raw/` is immutable; write outputs only under `data/processed/`

## Pipeline

1. [`notebooks/01_combine_movie_metadata.ipynb`](notebooks/01_combine_movie_metadata.ipynb) writes `movies_metadata.csv` (1,806 movies).
2. [`notebooks/02_extract_ratings.ipynb`](notebooks/02_extract_ratings.ipynb) writes `ratings.csv` from that allow-list. Do not re-extract ratings for the movie filter below.
3. [`notebooks/03_clean_movies.ipynb`](notebooks/03_clean_movies.ipynb) writes `movies_cleaned.csv` (1,806 rows). Parsed name lists, raw numbers, and log copies. No row drops.
4. [`notebooks/04_eda_movies.ipynb`](notebooks/04_eda_movies.ipynb) explores that table plus ratings, including per-user rating buckets.
5. [`notebooks/05_movie_features.ipynb`](notebooks/05_movie_features.ipynb) writes `movies_features.csv`.

`notebooks/06_user_features.ipynb` and `notebooks/07_user_clustering.ipynb` still expect the previous feature columns. Do not re-run them until they are adapted. `user_features.csv` is unchanged.

## Feature table

`data/processed/movies_features.csv`: **1,314 rows × 48 columns**.

Dropped before encoding: budget or revenue under $1,000 (includes zeros), or runtime missing or 0 (492 movies).

Kept:

- `runtime`
- `movie_age_log = log1p(2005 - year)`
- `budget_log = log(budget)`
- `log_roi = log(revenue) - log(budget)`
- 18 genre columns. `Foreign` is dropped. `Documentary` is kept.
- Top 15 companies after alias merge, plus `company_other`
- `us_only`, `us_coproduction`, `non_us`
- `lang_en`, `is_multilingual`

Not in the file: `year`, raw age, raw budget, raw revenue, decade dummies, scaled columns, `n_ratings`.

## People decision

From notebook 04, the strongest |Spearman| with `avg_rating` is the actor **track record** over the first **10** billed actors (0.315). Exposure and track records use other movies' ratings, so this file stores `director` and `cast_top10` only. Compute the scores after the train/test split, from training movies only.

## Next step

Adapt user features and clustering to these columns. Do not overwrite `movies_features.csv` with the old film-count encodings.
