"""Évaluation hors ligne sur MovieLens : split temporel par utilisateur.

Pour chaque utilisateur, ses notes les plus récentes (fraction `test_frac`) forment le
test. Les films notés au moins `like_threshold` en train forment son profil, ceux du test
la vérité terrain. Les recommandations excluent tout ce que l'utilisateur a vu en train.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd

Recommender = Callable[[list[int], int, set[int]], list[int]]
# Construit un recommandeur à partir des notes d'entraînement (ignorées s'il n'en a pas besoin).
RecommenderFactory = Callable[[pd.DataFrame], Recommender]


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    coverage: float
    users: int


def temporal_split(
    ratings: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values(["user_id", "timestamp"], kind="mergesort")
    rank = ratings.groupby("user_id").cumcount()
    size = ratings.groupby("user_id")["tmdb_id"].transform("size")
    is_test = rank >= (size * (1 - test_frac)).round()
    return ratings[~is_test], ratings[is_test]


def evaluate(
    make_recommender: RecommenderFactory,
    ratings: pd.DataFrame,
    catalog: Iterable[int],
    k: int = 10,
    like_threshold: float = 4.0,
    test_frac: float = 0.2,
) -> Metrics:
    catalog = set(catalog)
    ratings = ratings[ratings["tmdb_id"].isin(catalog)]
    train, test = temporal_split(ratings, test_frac)
    liked_train = train[train["rating"] >= like_threshold].groupby("user_id")["tmdb_id"]
    liked_test = test[test["rating"] >= like_threshold].groupby("user_id")["tmdb_id"]
    seen = train.groupby("user_id")["tmdb_id"].apply(set)
    recommend = make_recommender(train)

    liked_train_sets = {u: list(s) for u, s in liked_train}
    precisions, recalls, shown = [], [], set()
    for user, truth in liked_test:
        profile = liked_train_sets.get(user)
        if not profile:
            continue
        recs = recommend(profile, k, seen.get(user, set()))
        hits = len(set(recs) & set(truth))
        precisions.append(hits / k)
        recalls.append(hits / len(set(truth)))
        shown.update(recs)

    n = len(precisions)
    return Metrics(
        precision=sum(precisions) / n if n else 0.0,
        recall=sum(recalls) / n if n else 0.0,
        coverage=len(shown) / len(catalog) if catalog else 0.0,
        users=n,
    )


def popularity_recommender(train_ratings: pd.DataFrame) -> Recommender:
    """Référence : les films les plus notés, hors films déjà vus."""
    ranking = train_ratings["tmdb_id"].value_counts().index.tolist()

    def recommend(_profile: list[int], k: int, seen: set[int]) -> list[int]:
        return [t for t in ranking if t not in seen][:k]

    return recommend
