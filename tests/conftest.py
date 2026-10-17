import pandas as pd
import pytest


@pytest.fixture
def movies() -> pd.DataFrame:
    rows = [
        (
            1,
            "Alien",
            "A crew in space meets a deadly alien creature",
            ["Science Fiction", "Horror"],
            ["space", "alien"],
            ["Sigourney Weaver"],
            "Ridley Scott",
        ),
        (
            2,
            "Aliens",
            "Marines fight alien creatures on a space colony",
            ["Science Fiction", "Action"],
            ["space", "alien", "marine"],
            ["Sigourney Weaver"],
            "James Cameron",
        ),
        (
            3,
            "Avatar",
            "A marine on an alien planet joins a tribe",
            ["Science Fiction", "Action"],
            ["alien", "marine", "tribe"],
            ["Sam Worthington"],
            "James Cameron",
        ),
        (
            4,
            "Notting Hill",
            "A bookseller falls in love with a movie star",
            ["Romance", "Comedy"],
            ["london", "love"],
            ["Hugh Grant"],
            "Roger Michell",
        ),
        (
            5,
            "Love Actually",
            "Love stories intertwine in London at Christmas",
            ["Romance", "Comedy"],
            ["london", "love", "christmas"],
            ["Hugh Grant"],
            "Richard Curtis",
        ),
    ]
    df = pd.DataFrame(
        rows, columns=["tmdb_id", "title", "overview", "genres", "keywords", "cast", "director"]
    )
    df["year"] = 2000
    return df
