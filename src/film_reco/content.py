"""Moteur content-based : une similarité cosinus par champ, combinée avec des poids.

Chaque champ (synopsis, genres, mots-clés, cast, réalisateur) a son propre vectoriseur
TF-IDF et ses lignes normalisées. Le score entre deux films est donc la moyenne pondérée
des cosinus par champ, ce qui permet d'expliquer une recommandation champ par champ.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_WEIGHTS = {
    "overview": 1.0,
    "genres": 1.0,
    "keywords": 1.5,
    "cast": 0.75,
    "director": 0.75,
}


def _tags(values: Iterable[str]) -> list[str]:
    # « Science Fiction » -> « sciencefiction » : une étiquette reste un seul jeton.
    return [v.lower().replace(" ", "") for v in values if v]


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    # Les films exclus (score -inf) ne doivent jamais revenir, même si k dépasse le reste.
    top = np.argsort(-scores)[:k]
    return top[np.isfinite(scores[top])]


def _tag_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=lambda tags: tags)


@dataclass
class ContentRecommender:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def fit(self, movies: pd.DataFrame) -> ContentRecommender:
        unknown = set(self.weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"Champs inconnus : {sorted(unknown)}")
        self.movies = movies.reset_index(drop=True)
        self._index = {tid: i for i, tid in enumerate(self.movies["tmdb_id"])}

        blocks: dict[str, sparse.csr_matrix] = {}
        if "overview" in self.weights:
            vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
            blocks["overview"] = vec.fit_transform(self.movies["overview"])
        for name in ("genres", "keywords", "cast"):
            if name in self.weights:
                docs = self.movies[name].map(_tags)
                blocks[name] = _tag_vectorizer().fit_transform(docs)
        if "director" in self.weights:
            docs = self.movies["director"].map(lambda d: _tags([d]))
            blocks["director"] = _tag_vectorizer().fit_transform(docs)

        self._blocks = {k: normalize(v).tocsr() for k, v in blocks.items()}
        self._total = sum(self.weights.values())
        return self

    def _scores(self, profile: dict[str, sparse.csr_matrix]) -> dict[str, np.ndarray]:
        return {
            k: np.asarray((m @ profile[k].T).todense()).ravel() for k, m in self._blocks.items()
        }

    def _combine(self, per_field: dict[str, np.ndarray]) -> np.ndarray:
        return sum(self.weights[k] * s for k, s in per_field.items()) / self._total

    def find(self, query: str, limit: int = 5) -> list[tuple[str, int, float]]:
        """Titres les plus proches d'une saisie approximative : (titre, tmdb_id, score)."""
        titles = self.movies["title"].tolist()
        hits = process.extract(
            query, titles, scorer=fuzz.ratio, processor=utils.default_process, limit=limit
        )
        return [(t, int(self.movies.at[i, "tmdb_id"]), s) for t, s, i in hits]

    def similar(self, tmdb_id: int, k: int = 10) -> pd.DataFrame:
        """Les k films les plus proches d'un film, avec le détail du score par champ."""
        i = self._index[tmdb_id]
        per_field = self._scores({b: m[i] for b, m in self._blocks.items()})
        scores = self._combine(per_field)
        scores[i] = -np.inf
        return self._top(scores, per_field, k)

    def recommend(
        self, liked: Iterable[int], k: int = 10, exclude: Iterable[int] = ()
    ) -> list[int]:
        """Top-k pour un profil = moyenne des vecteurs des films aimés."""
        rows = [self._index[t] for t in liked if t in self._index]
        if not rows:
            return []
        profile = {b: sparse.csr_matrix(m[rows].mean(axis=0)) for b, m in self._blocks.items()}
        scores = self._combine(self._scores(profile))
        banned = [self._index[t] for t in set(exclude) | set(liked) if t in self._index]
        scores[banned] = -np.inf
        top = _top_k(scores, k)
        return self.movies["tmdb_id"].iloc[top].tolist()

    def _top(self, scores: np.ndarray, per_field: dict[str, np.ndarray], k: int) -> pd.DataFrame:
        top = _top_k(scores, k)
        out = self.movies.loc[top, ["tmdb_id", "title", "year"]].copy()
        out["score"] = scores[top]
        for name, s in per_field.items():
            out[name] = s[top]
        return out.reset_index(drop=True)
