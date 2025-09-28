from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


DEFAULT_K = 32
DEFAULT_INITIAL_RATING = 1200.0


@dataclass
class EloResult:
    r1_before: float
    r2_before: float
    r1_after: float
    r2_after: float
    delta: float


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / 400.0))


def update_ratings(
    r_winner: float,
    r_loser: float,
    *,
    k: int = DEFAULT_K,
    tie: bool = False,
) -> EloResult:
    e_winner = expected_score(r_winner, r_loser)
    e_loser = expected_score(r_loser, r_winner)

    if tie:
        s_winner, s_loser = 0.5, 0.5
    else:
        s_winner, s_loser = 1.0, 0.0

    r_w_after = r_winner + k * (s_winner - e_winner)
    r_l_after = r_loser + k * (s_loser - e_loser)

    return EloResult(
        r1_before=r_winner,
        r2_before=r_loser,
        r1_after=r_w_after,
        r2_after=r_l_after,
        delta=r_w_after - r_winner,
    )
