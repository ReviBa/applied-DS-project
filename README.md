# Netflix Movie Ratings - Persona Analysis and Rating Predictions

Applied Data Science project: predict a user’s rating of a movie that has no ratings yet, using TMDB production metadata and audience personas built from movies they already like.

The four research questions are in [docs/01_proposal.md](docs/01_proposal.md). Course format rules are in [docs/00_course_requirements.md](docs/00_course_requirements.md). The analysis itself is the eight notebooks under `notebooks/`, meant to be read and run from top to bottom.

## Setup

Python 3.9 or newer. From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Raw and processed tables are not in git. Download the two Kaggle datasets and place these files in `data/raw/`:


| File                                          | Dataset                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------- |
| `combined_data_1.txt` … `combined_data_4.txt` | [Netflix Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) |
| `movie_titles.csv`                            | same (no header; if the download is `movie_titles.txt`, rename it)              |
| `tmdb_5000_movies.csv`                        | [TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)           |
| `tmdb_5000_credits.csv`                       | same                                                                            |


`probe.txt` and `qualifying.txt` may come with the Netflix download. The notebooks leave them unused. Leave everything in `data/raw/` as downloaded; notebooks write only to `data/processed/`.

The four `combined_data` files are about 2 GB. Notebook 02 writes `ratings.csv` (about 1.5 GB, about 50 million rows). Notebooks 06–08 hold that table in memory, so use a machine with 16 GB of RAM and a few extra GB of free disk.

## Run

Start Jupyter from the project root, with the virtual environment active:

```powershell
jupyter lab
```

Open the notebooks in the order below and run each one from top to bottom before starting the next. Paths are relative to the `notebooks/` folder (`Path("..")`). Jupyter uses that folder as the working directory when the file is opened from the file browser, which is what these notebooks expect.

Notebooks 02 and 03 both need only the output of 01, so either may run first. Notebook 04 needs both of those outputs. It writes nothing, so notebook 05 can run as soon as `movies_cleaned.csv` exists. From 05 onward each notebook needs the files written by the one before it.


| Notebook                          | What it does                                                                                                                                                                                                  | Writes to `data/processed/`                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `01_combine_movie_metadata.ipynb` | Match Netflix titles to TMDB on normalized title and year                                                                                                                                                     | `movies_metadata.csv`                                                                   |
| `02_extract_ratings.ipynb`        | Stream the Netflix files and keep ratings of matched movies. Drop users with fewer than 5 of those ratings                                                                                                    | `ratings.csv`                                                                           |
| `03_clean_movies.ipynb`           | Drop unused columns, remove movies with zero budget, revenue, or runtime, parse genre / company / country / cast / crew lists                                                                                 | `movies_cleaned.csv`                                                                    |
| `04_eda_movies.ipynb`             | Distributions, data quality, and which encodings are worth keeping. Read this before 05; it does not write a file                                                                                             | —                                                                                       |
| `05_movie_features.ipynb`         | Apply the filters and encodings chosen in 04 (drop amounts under $1,000, logs, genre and company indicators, director and top 5 actors)                                                                       | `movies_features.csv`                                                                   |
| `06_split_and_user_eda.ipynb`     | Split **movies** 70% train / 15% validation / 15% test, stratified, so validation and test titles are unseen (cold start). User EDA on training movies. Aggregates use training movies only                   | `movie_split.csv`, `movie_stats_train.csv`, `track_records.csv`, `user_stats_train.csv` |
| `07_user_clustering.ipynb`        | K-means personas for users with at least 20 training-movie ratings, from genre and production taste. Compares genres alone, production features, and production features plus rating variance                 | `user_personas.csv`, `user_profiles.csv`, `persona_genre_scores.csv`                    |
| `08_rating_model.ipynb`           | Gradient-boosted trees in a ladder: movie features, then user rating habits, then persona label, persona distances, and the full taste profile. Test RMSE, bootstrap intervals, and SHAP stay in the notebook | —                                                                                       |


If `data/processed/` already contains the files a notebook reads, that notebook can be re-run on its own. A full reproduction starts at notebook 01 with only `data/raw/` filled in.