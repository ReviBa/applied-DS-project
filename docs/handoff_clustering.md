# Handoff — user-feature clustering

## Project

Segment Netflix users into personas from `data/processed/user_features.csv`, built in `notebooks/06_user_features.ipynb`.

- Course rules: `docs/00_course_requirements.md`
- Agent guide: `AGENTS.md`
- `data/raw/` is immutable; write outputs only under `data/processed/`

This step clusters **types of users**, including how much they rate and how they use the 1–5 scale, as well as which movies they like. Follow the column list below. 

Build the cluster matrix from `user_features.csv` in a new notebook `notebooks/07_user_clustering.ipynb`.

## Input


| File                                 | Role                                                                              |
| ------------------------------------ | --------------------------------------------------------------------------------- |
| `data/processed/user_features.csv`   | One row per user. 442,043 users with `n_ratings >= 5` and `n_high >= 1`.          |
| `data/processed/movies_features.csv` | Movie features already averaged into the user table. Do not re-join for this fit. |
| `data/processed/ratings.csv`         | Needed only if movie averages are recomputed.                                     |


Taste columns in the user table are means over that user's ratings of 4 or 5. Behavior columns use every rating.

## Who to cluster

Filter at fit time to `n_high >= 5` (431,528 users among those with `n_ratings >= 5`). Do not rewrite `user_features.csv`; it stays at `n_high >= 1`. `n_high` is a cutoff, not a cluster feature.

## Cluster columns

Scale every column before K-Means (standardize). After scaling, each column has one vote, so duplicates steal the split.

### Include

Habits:

- `log1p(n_ratings)` — how much they rate. Use the log; the raw count is too skewed.
- `mean_rating` — harsh vs generous. Spread is real (25th–75th about 3.36–3.96; std 0.47). Largest |Spearman| with a taste column is 0.21.
- `user_rating_std` — uses the whole 1–5 scale vs stays in a narrow band (std 0.24). Separate from the average.

Catalog (this is what makes a cluster nameable):

- all 19 `genre_*` columns. Keep overlapping pairs (animation/family 0.76, history/war 0.70, action/adventure 0.68); each pair still differs.
- `movie_age`, `runtime`, `budget_log`
- `director_film_count_log`
- one cast column: the mean of `actor_1_film_count_log`, `actor_2_film_count_log`, and `actor_3_film_count_log`

The three actor logs correlate 0.52–0.67. Three separate columns would give "famous cast" three votes.

### Leave out of the distance

Keep these columns in the saved user table. Use them as filters or as descriptions of each persona after the fit.


| Column                 | Why it stays out                                                                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_high`               | Same axis as `n_ratings`, and it is the sample size of the taste means. Filter only.                                                                 |
| `pct_high`, `pct_low`  | Restatements of `mean_rating`.                                                                                                                       |
| `genre_entropy`        | Spearman 0.71 with `n_ratings`. Mostly "rated enough films to cover genres."                                                                         |
| `log_roi`              | Correlation −0.45 with `budget_log`. Add only if the fit should separate hits vs flops.                                                              |
| `lang_en`              | Mean 0.98, std 0.04. Almost no separation.                                                                                                           |
| `movie_rating_std`     | Mean 0.99, std 0.03. Almost no separation.                                                                                                           |
| `mean_rating_vs_movie` | Close to generosity. Movie averages in this file use all ratings. Describe clusters with it only after those averages are recomputed on train users. |


Optional extra habit, only as a residual: `genre_entropy` after subtracting what `log(n_high)` predicts. That residual is specialist vs generalist. The raw entropy column is not.

`movie_age` rises with activity (Spearman 0.36 with `n_ratings`). Rare genre shares do too (`western` 0.47, `horror` 0.46, `music` 0.38). After the fit, check that a persona is not just the heavy-rater group.

## Weighting

Genres plus production columns are about 24 of ~27 features, so groups will be named by catalog, and the three habit columns will be a secondary split. That is the intended balance.

If "rates a lot" should stand equal with a genre, scale the three habit columns as one block and the content columns as another so each block has similar total weight, or reduce the genres with PCA before combining them with the habits. Do both only if the first fit is catalog-only and habits do not show up in the persona profiles.

## Fit

- Standardize the cluster columns on the users who pass `n_high >= 5`.
- K-Means. Choose `k` by interpretability in a small range (4–6); report silhouette and the smallest cluster's share.
- Name each persona from its mean profile: top genres, `budget_log`, `movie_age`, `runtime`, plus `n_ratings`, `mean_rating`, and `user_rating_std`.
- Save assignments under `data/processed/` (for example `user_personas.csv` with `customer_id`, `persona_id`, and the split if you hold out users). Do not overwrite `user_features.csv`.



## Leakage

`mean_rating_vs_movie`, and any movie `avg_rating` / `rating_std` joined from the full ratings file, use every user. Recompute those movie stats on train users only before they enter a model or a cluster description that will be used downstream.

## Working rules

- Prefer a clear notebook with short section descriptions. Comment only non-obvious logic.
- Indicative variable names.
- Commit only when asked.

