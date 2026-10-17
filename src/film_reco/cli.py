"""Ligne de commande : film-reco download | similar | evaluate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from film_reco import data
from film_reco.content import ContentRecommender
from film_reco.evaluate import evaluate, popularity_recommender


def _similar(args: argparse.Namespace) -> None:
    model = ContentRecommender().fit(data.load_tmdb(args.data_dir))
    title, tmdb_id, score = model.find(args.title, limit=1)[0]
    if score < 80:
        print(f"« {args.title} » introuvable. Titres proches :")
        for t, _, s in model.find(args.title):
            print(f"  {t} ({s:.0f})")
        return
    print(f"Films proches de « {title} » :")
    with pd.option_context("display.width", 140, "display.precision", 3):
        print(model.similar(tmdb_id, k=args.k).to_string(index=False))


def _evaluate(args: argparse.Namespace) -> None:
    movies = data.load_tmdb(args.data_dir)
    ratings = data.load_movielens(args.data_dir)
    catalog = movies["tmdb_id"].tolist()

    model = ContentRecommender().fit(movies)
    candidates = {
        "popularité": popularity_recommender,
        "content-based": lambda _train: model.recommend,
    }
    print(f"{'modèle':<15} {'P@' + str(args.k):>8} {'R@' + str(args.k):>8} {'couverture':>11}")
    for name, rec in candidates.items():
        m = evaluate(rec, ratings, catalog, k=args.k, test_frac=args.test_frac)
        print(f"{name:<15} {m.precision:>8.4f} {m.recall:>8.4f} {m.coverage:>11.3f}")
    print(f"({m.users} utilisateurs évalués)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="film-reco")
    parser.add_argument("--data-dir", type=Path, default=data.DATA_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Télécharge TMDB 5000 et MovieLens small")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=lambda a: data.download(a.data_dir, a.force))

    p = sub.add_parser("similar", help="Films proches d'un titre")
    p.add_argument("title")
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(func=_similar)

    p = sub.add_parser("evaluate", help="Évaluation hors ligne sur MovieLens")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.set_defaults(func=_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
