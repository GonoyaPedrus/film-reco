import pytest

from film_reco.content import ContentRecommender


def test_similar_ranks_same_universe_first(movies):
    model = ContentRecommender().fit(movies)
    out = model.similar(2, k=4)
    assert 2 not in out["tmdb_id"].tolist()
    assert set(out["tmdb_id"].head(2)) == {1, 3}
    assert out["score"].is_monotonic_decreasing


def test_similar_exposes_per_field_scores(movies):
    model = ContentRecommender().fit(movies)
    out = model.similar(4, k=1)
    assert out.at[0, "tmdb_id"] == 5
    assert out.at[0, "cast"] == pytest.approx(1.0)


def test_recommend_excludes_liked_and_seen(movies):
    model = ContentRecommender().fit(movies)
    recs = model.recommend([1], k=3, exclude=[2])
    assert recs[0] == 3
    assert 1 not in recs and 2 not in recs


def test_recommend_unknown_profile_is_empty(movies):
    assert ContentRecommender().fit(movies).recommend([999]) == []


def test_find_tolerates_typos(movies):
    model = ContentRecommender().fit(movies)
    title, tmdb_id, _ = model.find("notting hil", limit=1)[0]
    assert (title, tmdb_id) == ("Notting Hill", 4)


def test_unknown_weight_rejected(movies):
    with pytest.raises(ValueError):
        ContentRecommender(weights={"plot": 1.0}).fit(movies)


def test_excluded_films_never_pad_short_lists(movies):
    model = ContentRecommender().fit(movies)
    assert model.recommend([1, 2], k=10, exclude=[3, 4]) == [5]
    assert 2 not in model.similar(2, k=10)["tmdb_id"].tolist()
