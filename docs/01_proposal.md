# Applied Data Science Project Proposal

**Decoding the Movie Personality: Predicting User Ratings through Production Metadata and Audience Persona Clustering**

## Overview and Motivation

Streaming platforms spend heavily to create and acquire movies. Classic recommendation systems (e.g. Netflix Prize style) rely mainly on collaborative filtering over anonymous user–item ratings. That works mathematically, but:

- it cannot recommend a newly released movie with no ratings (**cold start**), and
- it cannot explain *why* a user might like a film.

This project builds a **hybrid personalized model** that links viewer behavior with movie production features. We combine Netflix rating data with TMDB metadata (directors, top actors, budgets, runtimes, etc.).

## Data

Integration of two sources:

| Source | Role | Link |
|--------|------|------|
| Netflix Prize Dataset | Historical user ratings (1–5 stars), ~100M ratings, 17,770 titles (up to 2005) | https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data |
| TMDB 5000 Movie Dataset | Structural metadata for ~4,800 feature films (up to 2017) | https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata |

Local copies live under `data/raw/`.

## Research questions

1. **User persona clustering**  
   Can we segment streaming users into distinct behavioral personas from the structural metadata of movies they rate highly (e.g. budget scale, directors, genre profiles)?

2. **Polarization / variance factor**  
   Can rating variance isolate polarized / “cult” films, and does using variance as a movie feature improve clustering?

3. **Feature importance hierarchy**  
   Which TMDB semantic features have the highest weight when forecasting personalized scores? Does director track record matter more for some personas than others?

4. **Model evolution**  
   How much do predictions improve when moving from a simple baseline (global movie/user averages) to a model that includes user personality / personalization (lower prediction error)?
