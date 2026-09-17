"""One isolated game: ingest actual events and never apply suggested actions."""
import json
from types import SimpleNamespace

import libriichi

from .metadata import ReviewableEngine


def public_event(raw, seat):
    """Defend the observation boundary even when replay input is uncensored."""
    event = dict(raw)
    if event["type"] == "start_kyoku":
        event["tehais"] = [list(hand) if actor == seat else ["?"] * 13
                           for actor, hand in enumerate(event["tehais"])]
    elif event["type"] == "tsumo" and event["actor"] != seat:
        event["pai"] = "?"
    # Optional replay annotations cannot become future policy inputs.
    for key in ("meta", "wall", "yama", "ura_markers", "uradora_markers"):
        event.pop(key, None)
    return event


class Session:
    def __init__(self, predictor, seat):
        if type(seat) is not int or seat not in range(4):
            raise ValueError("seat must be 0..3")
        self.seat = seat
        self.engine = ReviewableEngine(predictor=predictor, player_state_type=libriichi.state.PlayerState)
        self.engine.set_player_ids([seat])
        self.state = libriichi.state.PlayerState(seat)
        if not callable(getattr(self.state, "public_shanten", None)):
            raise RuntimeError("Libriichi lacks the required public_shanten input contract")
        self.events = []
        self.started = False
        self.ended = False

    def react(self, events):
        if (not isinstance(events, list) or not events
                or any(not isinstance(e, dict) or "type" not in e for e in events)):
            raise ValueError("Expected nonempty MJAI event array")
        if self.ended:
            raise ValueError("Game already ended; open a new session")
        reaction = {"type": "none"}
        for index, raw in enumerate(events):
            event = public_event(raw, self.seat)
            kind = event["type"]
            if kind == "start_game":
                if self.started:
                    raise ValueError("Duplicate start_game")
                if event.get("id", self.seat) != self.seat:
                    raise ValueError("start_game seat differs from the requested session seat")
                if event.get("num_players", 4) != 4:
                    raise ValueError("RIN Native supports four-player games only")
                event.setdefault("id", self.seat)
                event.setdefault("names", ["A", "B", "C", "D"])
                self.engine.start_game(0)
                self.started = True
            elif not self.started:
                raise ValueError("Missing start_game")
            else:
                memory = self.engine.memories[0]
                # Recommendations are counterfactual during review. Preserve an
                # atomic riichi discard only if the real next move declares it.
                if (memory.pending_riichi is not None and kind not in ("dora", "reach_accepted")
                        and not (kind == "reach" and event.get("actor") == self.seat)):
                    memory.pending_riichi = None
            self.state.update(json.dumps(event, separators=(",", ":")))
            if kind == "end_kyoku":
                self.events.append(event)
                self.engine.end_kyoku_with_log(0, json.dumps(self.events, separators=(",", ":")))
                self.events = []
            elif kind == "end_game":
                if self.events:
                    # Preserve any final public tail even for a truncated log.
                    self.engine._sync_memory(0, json.dumps(self.events, separators=(",", ":")))
                self.engine.end_game(0, ())
                self.ended = True
            elif kind != "start_game":
                self.events.append(event)
            if (index == len(events) - 1 and kind not in ("start_game", "end_kyoku", "end_game")
                    and self.state.last_cans.can_act):
                scene = SimpleNamespace(game_index=0, state=self.state,
                                        events_json=json.dumps(self.events, separators=(",", ":")))
                reaction = json.loads(self.engine.react_batch([scene])[0])
        return reaction
