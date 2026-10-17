"""Téléchargement et chargement des données TMDB 5000 et MovieLens."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")

# Copie des CSV TMDB 5000 (Kaggle) déjà publiée dans l'ancien projet de cours.
TMDB_URL = (
    "https://raw.githubusercontent.com/GonoyaPedrus/Film_recommandation_model/"
    "main/recommandation_films.zip"
)
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

TMDB_FILES = ("tmdb_5000_movies.csv", "tmdb_5000_credits.csv")
MOVIELENS_FILES = ("ratings.csv", "links.csv")

TOP_CAST = 5


def _extract(url: str, members: tuple[str, ...], dest: Path) -> None:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        by_basename = {Path(n).name: n for n in zf.namelist() if not n.startswith("__MACOSX")}
        for name in members:
            if name not in by_basename:
                raise FileNotFoundError(f"{name} absent de l'archive {url}")
            (dest / name).write_bytes(zf.read(by_basename[name]))


def download(data_dir: Path = DATA_DIR, force: bool = False) -> None:
    """Télécharge TMDB 5000 dans data/tmdb et MovieLens small dans data/movielens."""
    for url, members, sub in (
        (TMDB_URL, TMDB_FILES, "tmdb"),
        (MOVIELENS_URL, MOVIELENS_FILES, "movielens"),
    ):
        dest = data_dir / sub
        if not force and all((dest / m).exists() for m in members):
            continue
        _extract(url, members, dest)


def _names(raw: str) -> list[str]:
    return [item["name"] for item in json.loads(raw)] if isinstance(raw, str) else []


def _director(raw: str) -> str:
    for member in json.loads(raw) if isinstance(raw, str) else []:
        if member.get("job") == "Director":
            return member["name"]
    return ""


def load_tmdb(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Un film par ligne, colonnes de listes déjà décodées depuis le JSON TMDB."""
    movies = pd.read_csv(data_dir / "tmdb" / "tmdb_5000_movies.csv")
    credits = pd.read_csv(data_dir / "tmdb" / "tmdb_5000_credits.csv")

    df = pd.DataFrame(
        {
            "tmdb_id": movies["id"].astype(int),
            "title": movies["title"],
            "original_title": movies["original_title"],
            "overview": movies["overview"].fillna(""),
            "genres": movies["genres"].map(_names),
            "keywords": movies["keywords"].map(_names),
            "companies": movies["production_companies"].map(_names),
            "year": pd.to_datetime(movies["release_date"], errors="coerce").dt.year,
            "popularity": movies["popularity"],
            "vote_count": movies["vote_count"],
        }
    )
    cred = pd.DataFrame(
        {
            "tmdb_id": credits["movie_id"].astype(int),
            "cast": credits["cast"].map(lambda raw: _names(raw)[:TOP_CAST]),
            "director": credits["crew"].map(_director),
        }
    )
    df = df.merge(cred, on="tmdb_id", how="left")
    df["cast"] = df["cast"].map(lambda v: v if isinstance(v, list) else [])
    df["director"] = df["director"].fillna("")
    return df.reset_index(drop=True)


def load_movielens(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Notes MovieLens avec l'identifiant TMDB de chaque film (films sans lien retirés)."""
    ratings = pd.read_csv(data_dir / "movielens" / "ratings.csv")
    links = pd.read_csv(data_dir / "movielens" / "links.csv").dropna(subset=["tmdbId"])
    links["tmdb_id"] = links["tmdbId"].astype(int)
    out = ratings.merge(links[["movieId", "tmdb_id"]], on="movieId", how="inner")
    return out.rename(columns={"userId": "user_id"})[["user_id", "tmdb_id", "rating", "timestamp"]]
