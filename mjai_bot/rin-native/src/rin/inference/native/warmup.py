"""Warm the native network using a synthetic, public, legal decision."""
from .session import Session


def warmup(predictor):
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
            "1p", "4p", "7p", "E"]
    session = Session(predictor, 0)
    return session.react([
        {"type": "start_game", "id": 0},
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
         "kyotaku": 0, "oya": 0, "dora_marker": "9s", "scores": [25000] * 4,
         "tehais": [hand, ["?"] * 13, ["?"] * 13, ["?"] * 13]},
        {"type": "tsumo", "actor": 0, "pai": "9p"},
    ])
