# Agent guide — Applied DS Project

## Goal

Predict / personalize movie ratings by combining Netflix ratings with TMDB production metadata, including user persona clustering.

## Source of truth

- Course rules: `docs/00_course_requirements.md`
- Proposal & research questions: `docs/01_proposal.md`
- Do not change the research questions without updating `docs/01_proposal.md`.

## Layout

- `data/raw/` — original CSVs (do not overwrite)
- `data/processed/` — cleaned / joined outputs
- `notebooks/` — main Jupyter analysis (course deliverable)
- `src/` — reusable helper functions if needed

## Working rules

- Prefer a clear notebook with short section descriptions.
- Comment only non-obvious logic.
- Keep scope tied to the four research questions in the proposal.
- Current phase: **scaffold / data setup**.
