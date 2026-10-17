import pandas as pd
import pytest

from film_reco.evaluate import evaluate, popularity_recommender, temporal_split


@pytest.fixture
def ratings() -> pd.DataFrame:
    rows = [
        # user 1 aime la SF : train 1, 2 puis test 3
        (1, 1, 5.0, 1),
        (1, 2, 4.5, 2),
        (1, 4, 2.0, 3),
        (1, 3, 5.0, 4),
        # user 2 aime la romance : train 4, 1 puis test 5
        (2, 4, 5.0, 1),
        (2, 1, 3.0, 2),
        (2, 2, 1.0, 3),
        (2, 5, 4.0, 4),
    ]
    return pd.DataFrame(rows, columns=["user_id", "tmdb_id", "rating", "timestamp"])


def test_temporal_split_keeps_latest_in_test(ratings):
    train, test = temporal_split(ratings, test_frac=0.25)
    assert sorted(test["tmdb_id"]) == [3, 5]
    assert len(train) == 6


def test_perfect_recommender_scores_one(ratings):
    truth = {1: 3, 2: 5}

    def oracle(profile, k, seen):
        user = 1 if 1 in profile and 2 in profile else 2
        return [truth[user]]

    m = evaluate(lambda _train: oracle, ratings, catalog=[1, 2, 3, 4, 5], k=1, test_frac=0.25)
    assert (m.precision, m.recall, m.users) == (1.0, 1.0, 2)
    assert m.coverage == pytest.approx(2 / 5)


def test_popularity_skips_seen(ratings):
    train, _ = temporal_split(ratings, test_frac=0.25)
    rec = popularity_recommender(train)
    assert rec([], 2, {1, 2}) == [4]


def test_popularity_is_trained_on_train_split_only(ratings):
    # Le film 3 n'est noté qu'en test : la référence ne doit pas le connaître.
    m = evaluate(popularity_recommender, ratings, catalog=[1, 2, 3, 4, 5], k=1, test_frac=0.25)
    assert m.precision == 0.0
