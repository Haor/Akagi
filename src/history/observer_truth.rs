//! Offline-only labels from an original log, never from or fed into bot input.
//!
//! Hidden tiles remain unknown. Structural labels use the prefix ending at the
//! decision; only recorded settlement labels inspect the following outcome.

use anyhow::{bail, Result};
use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::parser::mjai_to_tid;
use riichienv_core::types::{Conditions, Meld, MeldType, Wind};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::local_review::LocalReviewDecision;
use crate::schema::MjaiEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverTruth {
    pub schema_version: String,
    pub event_index: usize,
    pub opponents: Vec<OpponentTruth>,
    pub candidates: Vec<CandidateTruth>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpponentTruth {
    pub seat: u8,
    pub tenpai: Option<bool>,
    pub waits: Option<Vec<String>>,
    pub furiten: Option<bool>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateTruth {
    pub candidate_index: usize,
    /// Same seat order as ObserverTruth.opponents, never absolute seat indices.
    pub ron: Vec<Option<bool>>,
    pub any_ron: Option<bool>,
    pub actual_discard: bool,
    pub deal_in_points: Option<i32>,
}

#[derive(Clone)]
struct Player {
    hand: Vec<String>,
    melds: Vec<Meld>,
    river: Vec<u8>,
    invalid: bool,
    reached: bool,
    temporary_furiten: Option<bool>,
    riichi_furiten: Option<bool>,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            hand: Vec::new(),
            melds: Vec::new(),
            river: Vec::new(),
            invalid: false,
            reached: false,
            temporary_furiten: Some(false),
            riichi_furiten: Some(false),
        }
    }
}

impl Player {
    fn remove(&mut self, tile: &str) {
        if let Some(index) = self.hand.iter().position(|item| item == tile) {
            self.hand.remove(index);
        } else if let Some(index) = self.hand.iter().position(|item| item == "?") {
            self.hand.remove(index);
        } else {
            self.invalid = true;
        }
    }

    fn evaluator(&self) -> std::result::Result<HandEvaluator, &'static str> {
        if self.invalid {
            return Err("invalid_hand");
        }
        if self.hand.is_empty() || self.hand.iter().any(|tile| tile == "?") {
            return Err("missing_hand");
        }
        if self.melds.len() > 4 || self.hand.len() + 3 * self.melds.len() != 13 {
            return Err("unsupported_state");
        }
        let tiles = self
            .hand
            .iter()
            .map(|tile| known_tile(tile).ok_or("invalid_hand"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut counts = [0_u8; 34];
        for tile in tiles
            .iter()
            .chain(self.melds.iter().flat_map(|meld| meld.tiles.iter()))
        {
            let count = &mut counts[(*tile / 4) as usize];
            *count += 1;
            if *count > 4 {
                return Err("invalid_hand");
            }
        }
        // HandEvaluator accepts only known 136-space tiles. Never construct it
        // with '?' sentinels, malformed counts, or a player holding a 14th tile.
        Ok(HandEvaluator::new(tiles, self.melds.clone()))
    }

    fn waits(&self) -> std::result::Result<Vec<u8>, &'static str> {
        let evaluator = self.evaluator()?;
        Ok(evaluator
            .get_waits_u8()
            .into_iter()
            // The library checks concealed counts only; meld copies must also
            // exclude structurally impossible fifth-copy waits.
            .filter(|tile| evaluator.full_hand.counts[*tile as usize] < 4)
            .collect())
    }

    fn furiten(&self, waits: &[u8]) -> Option<bool> {
        if waits.iter().any(|tile| self.river.contains(tile))
            || self.temporary_furiten == Some(true)
            || self.riichi_furiten == Some(true)
        {
            Some(true)
        } else if self.temporary_furiten.is_none() || self.riichi_furiten.is_none() {
            None
        } else {
            Some(false)
        }
    }
}

struct Prefix {
    players: [Player; 4],
    active: bool,
    supported: bool,
    dealer: u8,
    round_wind: u8,
    draws: usize,
    replacement_pending: bool,
    last_draw_replacement: bool,
    // Recorded MJAI has no explicit pass. Resolve opportunities only when a
    // subsequent actual move closes the window; a competing Hora is ambiguous.
    pending: Vec<(u8, Option<bool>)>,
}

impl Default for Prefix {
    fn default() -> Self {
        Self {
            players: std::array::from_fn(|_| Player::default()),
            active: false,
            supported: false,
            dealer: 0,
            round_wind: 0,
            draws: 0,
            replacement_pending: false,
            last_draw_replacement: false,
            pending: Vec::new(),
        }
    }
}

impl Prefix {
    fn ron(&self, seat: u8, tile: u8, chankan: bool) -> Option<bool> {
        if !self.active || !self.supported || self.draws > 70 {
            return None;
        }
        let player = self.players.get(seat as usize)?;
        let waits = player.waits().ok()?;
        if !waits.contains(&(tile / 4)) || player.furiten(&waits) == Some(true) {
            return Some(false);
        }
        let evaluator = player.evaluator().ok()?;
        let result = evaluator.calc(
            tile,
            Vec::new(),
            Vec::new(),
            Some(Conditions {
                tsumo: false,
                riichi: player.reached,
                // Extra han do not change eligibility once riichi is present.
                houtei: !chankan && self.draws == 70 && !self.last_draw_replacement,
                chankan,
                player_wind: Wind::from((seat + 4 - self.dealer) % 4),
                round_wind: Wind::from(self.round_wind),
                ..Conditions::default()
            }),
        );
        if !result.is_win {
            return Some(false); // A structural wait without a yaku cannot ron.
        }
        player.furiten(&waits).map(|furiten| !furiten)
    }

    fn close_window(&mut self, event: &MjaiEvent) {
        match event {
            MjaiEvent::Dora { .. } | MjaiEvent::ReachAccepted { .. } | MjaiEvent::None => return,
            MjaiEvent::Hora { .. }
            | MjaiEvent::Ryukyoku { .. }
            | MjaiEvent::EndKyoku
            | MjaiEvent::EndGame { .. }
            | MjaiEvent::StartGame { .. }
            | MjaiEvent::StartKyoku { .. } => {
                self.pending.clear();
                return;
            }
            _ => {}
        }
        for (seat, could_ron) in self.pending.drain(..) {
            let player = &mut self.players[seat as usize];
            if could_ron == Some(false) {
                continue;
            }
            let mark = could_ron.map(|_| true);
            if player.temporary_furiten != Some(true) {
                player.temporary_furiten = mark;
            }
            if player.reached && player.riichi_furiten != Some(true) {
                player.riichi_furiten = mark;
            }
        }
    }

    fn meld(
        &mut self,
        actor: u8,
        target: u8,
        called: Option<&str>,
        consumed: &[String],
        kind: MeldType,
    ) {
        let Some(player) = self.players.get_mut(actor as usize) else {
            self.supported = false;
            return;
        };
        let all = consumed.iter().map(String::as_str).chain(called);
        let Some(tiles) = all.map(known_tile).collect::<Option<Vec<_>>>() else {
            player.invalid = true;
            return;
        };
        let mut bases: Vec<u8> = tiles.iter().map(|tile| tile / 4).collect();
        bases.sort_unstable();
        let valid = match kind {
            MeldType::Chi => {
                bases.len() == 3
                    && bases[0] < 27
                    && bases[0] / 9 == bases[2] / 9
                    && bases[1] == bases[0] + 1
                    && bases[2] == bases[0] + 2
            }
            MeldType::Pon => bases.len() == 3 && bases.iter().all(|tile| *tile == bases[0]),
            MeldType::Daiminkan | MeldType::Ankan => {
                bases.len() == 4 && bases.iter().all(|tile| *tile == bases[0])
            }
            MeldType::Kakan => false,
        };
        if !valid || target >= 4 {
            player.invalid = true;
            return;
        }
        for tile in consumed {
            player.remove(tile);
        }
        player.melds.push(Meld::new(
            kind,
            tiles,
            kind != MeldType::Ankan,
            target as i8,
            called.and_then(known_tile),
        ));
    }

    fn apply(&mut self, event: &MjaiEvent) {
        self.close_window(event);
        match event {
            MjaiEvent::StartKyoku {
                tehais,
                num_players,
                bakaze,
                oya,
                ..
            } => {
                *self = Self::default();
                self.active = true;
                self.supported = *num_players == 4 && tehais.len() == 4 && *oya < 4;
                self.dealer = *oya;
                self.round_wind = match bakaze.as_str() {
                    "E" => 0,
                    "S" => 1,
                    "W" => 2,
                    "N" => 3,
                    _ => {
                        self.supported = false;
                        0
                    }
                };
                for (player, hand) in self.players.iter_mut().zip(tehais) {
                    player.hand = hand.clone();
                    player.invalid = hand.len() != 13
                        || hand
                            .iter()
                            .any(|tile| tile != "?" && known_tile(tile).is_none());
                }
            }
            MjaiEvent::Tsumo { actor, pai } => {
                if let Some(player) = self.players.get_mut(*actor as usize) {
                    player.hand.push(pai.clone());
                    player.temporary_furiten = Some(false);
                    player.invalid |= pai != "?" && known_tile(pai).is_none();
                    self.draws += 1;
                    self.last_draw_replacement = self.replacement_pending;
                    self.replacement_pending = false;
                } else {
                    self.supported = false;
                }
            }
            MjaiEvent::Dahai { actor, pai, .. } => {
                if let Some(player) = self.players.get_mut(*actor as usize) {
                    player.remove(pai);
                    if let Some(tile) = known_tile(pai) {
                        player.river.push(tile / 4);
                        self.pending = (0..4)
                            .filter(|seat| seat != actor)
                            .map(|seat| (seat, self.ron(seat, tile, false)))
                            .collect();
                    } else {
                        self.supported = false;
                    }
                } else {
                    self.supported = false;
                }
            }
            MjaiEvent::Chi {
                actor,
                target,
                pai,
                consumed,
            } => self.meld(*actor, *target, Some(pai), consumed, MeldType::Chi),
            MjaiEvent::Pon {
                actor,
                target,
                pai,
                consumed,
            } => self.meld(*actor, *target, Some(pai), consumed, MeldType::Pon),
            MjaiEvent::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => {
                self.meld(*actor, *target, Some(pai), consumed, MeldType::Daiminkan);
                self.replacement_pending = true;
            }
            MjaiEvent::Ankan { actor, consumed } => {
                self.meld(*actor, *actor, None, consumed, MeldType::Ankan);
                self.replacement_pending = true;
                // Robbing an ankan is ruleset-sensitive (kokushi only). Do not
                // fabricate a definite pass/eligibility result for that window.
                self.pending = (0..4)
                    .filter(|seat| seat != actor)
                    .map(|seat| (seat, None))
                    .collect();
            }
            MjaiEvent::Kakan {
                actor,
                pai,
                consumed,
            } => {
                if let (Some(player), Some(tile)) =
                    (self.players.get_mut(*actor as usize), known_tile(pai))
                {
                    player.remove(pai);
                    if let Some(meld) = player.melds.iter_mut().find(|meld| {
                        meld.meld_type == MeldType::Pon && meld.tiles[0] / 4 == tile / 4
                    }) {
                        if consumed.iter().all(|value| {
                            known_tile(value).is_some_and(|value| value / 4 == tile / 4)
                        }) {
                            meld.meld_type = MeldType::Kakan;
                            meld.tiles.push(tile);
                        } else {
                            player.invalid = true;
                        }
                    } else {
                        player.invalid = true;
                    }
                    self.pending = (0..4)
                        .filter(|seat| seat != actor)
                        .map(|seat| (seat, self.ron(seat, tile, true)))
                        .collect();
                } else {
                    self.supported = false;
                }
                self.replacement_pending = true;
            }
            MjaiEvent::Reach { actor, .. } | MjaiEvent::ReachAccepted { actor } => {
                if let Some(player) = self.players.get_mut(*actor as usize) {
                    player.reached = true;
                    if player.melds.iter().any(|meld| meld.opened) {
                        player.invalid = true;
                    }
                } else {
                    self.supported = false;
                }
            }
            MjaiEvent::Kita { .. } => self.supported = false,
            MjaiEvent::StartGame { .. } | MjaiEvent::EndKyoku | MjaiEvent::EndGame { .. } => {
                self.active = false
            }
            _ => {}
        }
    }

    fn opponent(&self, seat: u8) -> OpponentTruth {
        let player = &self.players[seat as usize];
        let waits = if self.active && self.supported {
            player.waits()
        } else {
            Err("unsupported_state")
        };
        match waits {
            Ok(waits) => {
                let furiten = player.furiten(&waits);
                OpponentTruth {
                    seat,
                    tenpai: Some(!waits.is_empty()),
                    waits: Some(waits.into_iter().map(tile_name).collect()),
                    furiten,
                    reason: furiten.is_none().then(|| "unsupported_state".into()),
                }
            }
            Err(reason) => OpponentTruth {
                seat,
                tenpai: None,
                waits: None,
                furiten: None,
                reason: Some(reason.into()),
            },
        }
    }
}

fn known_tile(text: &str) -> Option<u8> {
    // The upstream parser tolerates trailing text. Accept only canonical MJAI
    // spellings before letting a tile reach an array-indexed evaluator.
    let bytes = text.as_bytes();
    let suited = bytes.len() == 2
        && (b'1'..=b'9').contains(&bytes[0])
        && matches!(bytes[1], b'm' | b'p' | b's');
    if !suited
        && !matches!(
            text,
            "E" | "S" | "W" | "N" | "P" | "F" | "C" | "5mr" | "5pr" | "5sr"
        )
    {
        return None;
    }
    mjai_to_tid(text).filter(|tile| *tile < 136)
}

fn tile_name(tile: u8) -> String {
    if tile < 27 {
        format!("{}{}", tile % 9 + 1, ['m', 'p', 's'][(tile / 9) as usize])
    } else {
        ["E", "S", "W", "N", "P", "F", "C"][(tile - 27) as usize].into()
    }
}

struct Candidate {
    index: usize,
    tile: String,
    reach: bool,
    tsumogiri: Option<bool>,
}

fn candidates(meta: Option<&Value>, seat: u8) -> Vec<Candidate> {
    let Some(meta) = meta else {
        return Vec::new();
    };
    let Some(policy) = meta.get("candidates").and_then(Value::as_array) else {
        return Vec::new();
    };
    let selected_indices = meta
        .pointer("/observer/candidates")
        .and_then(Value::as_array);
    policy
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            if selected_indices.is_some_and(|values| {
                !values.iter().any(|value| {
                    value.get("candidate_index").and_then(Value::as_u64) == Some(index as u64)
                })
            }) {
                return None;
            }
            let action = value.get("action")?;
            let reach = action.get("type")?.as_str()? == "reach";
            let discard = if reach {
                value.get("continuation")?
            } else {
                action
            };
            if discard.get("type")?.as_str()? != "dahai"
                || discard.get("actor")?.as_u64()? != seat as u64
            {
                return None;
            }
            let tile = discard.get("pai")?.as_str()?;
            known_tile(tile)?;
            Some(Candidate {
                index,
                tile: tile.into(),
                reach,
                tsumogiri: discard.get("tsumogiri").and_then(Value::as_bool),
            })
        })
        .collect()
}

fn actual_discard(events: &[MjaiEvent], index: usize, seat: u8) -> Option<(usize, bool)> {
    let mut reach = false;
    for (position, event) in events.iter().enumerate().skip(index + 1) {
        match event {
            MjaiEvent::None | MjaiEvent::Dora { .. } | MjaiEvent::ReachAccepted { .. } => {}
            MjaiEvent::Reach { actor, .. } if *actor == seat => reach = true,
            MjaiEvent::Dahai { actor, .. } if *actor == seat => return Some((position, reach)),
            _ => return None,
        }
    }
    None
}

fn recorded_loss(events: &[MjaiEvent], discard: usize, seat: u8) -> Option<i32> {
    let mut loss = 0_i32;
    let mut won = false;
    let mut complete = true;
    let mut settlements = Vec::<Vec<i32>>::new();
    for event in events.iter().skip(discard + 1) {
        match event {
            MjaiEvent::None | MjaiEvent::ReachAccepted { .. } | MjaiEvent::Dora { .. } => {}
            MjaiEvent::Hora {
                actor,
                target,
                deltas,
                ..
            } if *target == seat && *actor != seat => {
                won = true;
                match deltas.as_ref().filter(|values| values.len() == 4) {
                    Some(values) if values[seat as usize] <= 0 => {
                        // Some sources repeat the combined settlement for every winner.
                        if !settlements.contains(values) {
                            loss = loss.checked_add(values[seat as usize].checked_neg()?)?;
                            settlements.push(values.clone());
                        }
                    }
                    _ => complete = false,
                }
            }
            MjaiEvent::Hora { .. } => return None,
            MjaiEvent::Tsumo { .. }
            | MjaiEvent::Chi { .. }
            | MjaiEvent::Pon { .. }
            | MjaiEvent::Daiminkan { .. }
            | MjaiEvent::Ryukyoku { .. } => {
                return if won {
                    complete.then_some(loss)
                } else {
                    Some(0)
                }
            }
            _ => return (won && complete).then_some(loss),
        }
    }
    (won && complete).then_some(loss)
}

/// Results preserve the supplied decision order, including repeated indices.
pub fn build_truth(
    events: &[MjaiEvent],
    decisions: &[LocalReviewDecision],
    seat: u8,
) -> Result<Vec<ObserverTruth>> {
    if seat >= 4 {
        bail!("observer truth requires a four-player seat");
    }
    if decisions
        .iter()
        .any(|decision| decision.event_index >= events.len())
    {
        bail!("observer truth decision index is outside the source log");
    }
    let mut positions: Vec<usize> = (0..decisions.len()).collect();
    positions.sort_by_key(|index| decisions[*index].event_index);
    let mut result: Vec<Option<ObserverTruth>> = vec![None; decisions.len()];
    let mut next = 0;
    let mut prefix = Prefix::default();
    for (event_index, event) in events.iter().enumerate() {
        if next == positions.len() {
            break;
        }
        prefix.apply(event);
        while next < positions.len() && decisions[positions[next]].event_index == event_index {
            let position = positions[next];
            let decision = &decisions[position];
            let seats: Vec<u8> = (1..4).map(|relative| (seat + relative) % 4).collect();
            let opponents = seats.iter().map(|seat| prefix.opponent(*seat)).collect();
            let actual = actual_discard(events, event_index, seat);
            let candidate_truth = candidates(decision.meta.as_ref(), seat).into_iter().map(|candidate| {
                let ron: Vec<Option<bool>> = seats.iter().map(|seat| known_tile(&candidate.tile).and_then(|tile| prefix.ron(*seat, tile, false))).collect();
                let any_ron = if ron.contains(&Some(true)) { Some(true) }
                    else if ron.iter().all(|value| *value == Some(false)) { Some(false) } else { None };
                let matching = actual.filter(|(index, reach)| {
                    matches!(&events[*index], MjaiEvent::Dahai { pai, tsumogiri, .. }
                        if pai == &candidate.tile && *reach == candidate.reach && candidate.tsumogiri.is_none_or(|value| value == *tsumogiri))
                });
                CandidateTruth { candidate_index: candidate.index, ron, any_ron,
                    actual_discard: matching.is_some(), deal_in_points: matching.and_then(|(index, _)| recorded_loss(events, index, seat)) }
            }).collect();
            result[position] = Some(ObserverTruth {
                schema_version: "akagi.observer-truth.v1".into(),
                event_index,
                opponents,
                candidates: candidate_truth,
            });
            next += 1;
        }
    }
    Ok(result
        .into_iter()
        .map(|value| value.expect("all decision indices were visited"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tiles(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    fn event(value: Value) -> MjaiEvent {
        serde_json::from_value(value).unwrap()
    }

    fn opening(hand: &str) -> MjaiEvent {
        event(
            json!({"type":"start_kyoku", "bakaze":"E", "dora_marker":"9m",
            "kyoku":1, "honba":0, "kyotaku":0, "oya":0, "scores":[25000,25000,25000,25000],
            "tehais":[vec!["?";13], tiles(hand), vec!["?";13], vec!["?";13]]}),
        )
    }

    fn decision(index: usize, candidates: Value) -> LocalReviewDecision {
        serde_json::from_value(json!({"event_index":index, "round":"E1", "turn":1,
            "hand":[], "trigger":{"type":"none"}, "actual":null, "recommended":{"type":"none"},
            "matches":null, "observer_truth":null, "meta":{"candidates":candidates}}))
        .unwrap()
    }

    fn discard_candidates() -> Value {
        json!([
            {"action":{"type":"dahai","actor":0,"pai":"6p","tsumogiri":true}},
            {"action":{"type":"dahai","actor":0,"pai":"3p","tsumogiri":false}}
        ])
    }

    const CLOSED: &str = "1m 2m 3m 1p 2p 3p 1s 2s 3s E E 4p 5p";

    fn ready() -> Prefix {
        let mut prefix = Prefix::default();
        prefix.apply(&opening(CLOSED));
        prefix
    }

    #[test]
    fn concealed_waits_include_red_five_base_type() {
        let mut prefix = Prefix::default();
        prefix.apply(&opening("1m 1m 2m 2m 3p 3p 4p 4p 6s 6s 7s 7s 5p"));
        let truth = prefix.opponent(1);
        assert_eq!(truth.tenpai, Some(true));
        assert!(truth.waits.unwrap().contains(&"5p".into()));
        assert_eq!(prefix.ron(1, known_tile("5pr").unwrap(), false), Some(true));
    }

    #[test]
    fn open_and_concealed_kan_waits_exclude_fifth_copies() {
        let mut prefix = ready();
        prefix.players[1].hand = tiles("1m 2p 3p 4p 5s 6s 7s 7p 8p 9p");
        prefix.players[1].melds = vec![Meld::new(
            MeldType::Pon,
            vec![known_tile("1m").unwrap(); 3],
            true,
            0,
            known_tile("1m"),
        )];
        assert_eq!(prefix.opponent(1).tenpai, Some(false));
        assert_eq!(prefix.opponent(1).waits, Some(Vec::new()));

        prefix.players[1].hand = tiles("2p 3p 4p 5s 6s 7s E E 4p 5p");
        prefix.players[1].melds = vec![Meld::new(
            MeldType::Ankan,
            vec![known_tile("1m").unwrap(); 4],
            false,
            1,
            None,
        )];
        assert_eq!(prefix.opponent(1).tenpai, Some(true));
        assert!(prefix.opponent(1).waits.unwrap().contains(&"6p".into()));
    }

    #[test]
    fn structural_wait_without_yaku_is_not_ron() {
        let mut prefix = ready();
        prefix.players[1].hand = tiles("2p 3p 4p 6p 7p 8p E E 4s 5s");
        prefix.players[1].melds = vec![Meld::new(
            MeldType::Chi,
            ["1m", "2m", "3m"]
                .map(|tile| known_tile(tile).unwrap())
                .to_vec(),
            true,
            0,
            known_tile("1m"),
        )];
        assert_eq!(prefix.opponent(1).tenpai, Some(true));
        assert_eq!(prefix.ron(1, known_tile("6s").unwrap(), false), Some(false));
        prefix.players[1].melds = vec![Meld::new(
            MeldType::Pon,
            vec![known_tile("P").unwrap(); 3],
            true,
            0,
            known_tile("P"),
        )];
        assert_eq!(prefix.ron(1, known_tile("6s").unwrap(), false), Some(true));
    }

    #[test]
    fn discard_furiten_covers_all_waits_and_pass_furiten_expires_on_draw() {
        let mut prefix = ready();
        prefix.players[1].river.push(known_tile("3p").unwrap() / 4);
        assert_eq!(prefix.opponent(1).furiten, Some(true));
        assert_eq!(prefix.ron(1, known_tile("6p").unwrap(), false), Some(false));

        let mut prefix = ready();
        prefix.apply(&event(
            json!({"type":"dahai","actor":0,"pai":"6p","tsumogiri":true}),
        ));
        prefix.apply(&event(json!({"type":"tsumo","actor":2,"pai":"?"})));
        assert_eq!(prefix.opponent(1).furiten, Some(true));
        prefix.apply(&event(json!({"type":"tsumo","actor":1,"pai":"9s"})));
        prefix.apply(&event(
            json!({"type":"dahai","actor":1,"pai":"9s","tsumogiri":true}),
        ));
        assert_eq!(prefix.opponent(1).furiten, Some(false));
    }

    #[test]
    fn passed_ron_after_riichi_remains_furiten_after_own_draw() {
        let mut prefix = ready();
        prefix.apply(&event(json!({"type":"reach_accepted","actor":1})));
        prefix.apply(&event(
            json!({"type":"dahai","actor":0,"pai":"6p","tsumogiri":true}),
        ));
        prefix.apply(&event(json!({"type":"tsumo","actor":2,"pai":"?"})));
        prefix.apply(&event(json!({"type":"tsumo","actor":1,"pai":"9s"})));
        prefix.apply(&event(
            json!({"type":"dahai","actor":1,"pai":"9s","tsumogiri":true}),
        ));
        assert_eq!(prefix.opponent(1).furiten, Some(true));
    }

    #[test]
    fn hidden_and_malformed_hands_remain_unknown() {
        let mut prefix = ready();
        let truth = prefix.opponent(2);
        assert_eq!(truth.tenpai, None);
        assert_eq!(truth.waits, None);
        assert_eq!(truth.furiten, None);
        assert_eq!(truth.reason.as_deref(), Some("missing_hand"));
        assert_eq!(prefix.ron(2, known_tile("6p").unwrap(), false), None);
        prefix.players[1].hand = vec!["1m".into(); 13];
        assert_eq!(prefix.opponent(1).reason.as_deref(), Some("invalid_hand"));
        prefix.players[1].hand[0] = "1mBAD".into();
        assert_eq!(prefix.opponent(1).reason.as_deref(), Some("invalid_hand"));
    }

    #[test]
    fn event_indices_preserve_order_and_future_draws_do_not_leak() {
        let events = vec![
            opening(CLOSED),
            event(json!({"type":"tsumo","actor":1,"pai":"9s"})),
            event(json!({"type":"dahai","actor":1,"pai":"1m","tsumogiri":false})),
        ];
        let decisions = vec![decision(2, json!([])), decision(0, json!([]))];
        let truth = build_truth(&events, &decisions, 0).unwrap();
        assert_eq!(truth[0].event_index, 2);
        assert_eq!(truth[1].event_index, 0);
        assert_eq!(truth[0].opponents[0].tenpai, Some(false));
        assert_eq!(truth[1].opponents[0].tenpai, Some(true));
        let prefix_only = build_truth(&events[..1], &decisions[1..], 0).unwrap();
        assert_eq!(
            serde_json::to_value(&truth[1]).unwrap(),
            serde_json::to_value(&prefix_only[0]).unwrap()
        );
    }

    #[test]
    fn actual_multi_ron_loss_is_separate_from_hypothetical_candidates() {
        let events = vec![
            opening(CLOSED),
            event(json!({"type":"tsumo","actor":0,"pai":"6p"})),
            event(json!({"type":"dahai","actor":0,"pai":"6p","tsumogiri":true})),
            event(json!({"type":"hora","actor":1,"target":0,"deltas":[-3900,3900,0,0]})),
            event(json!({"type":"hora","actor":2,"target":0,"deltas":[-2000,0,2000,0]})),
            MjaiEvent::EndKyoku,
        ];
        let truth = build_truth(&events, &[decision(1, discard_candidates())], 0).unwrap();
        assert!(truth[0].candidates[0].actual_discard);
        assert_eq!(truth[0].candidates[0].deal_in_points, Some(5900));
        assert!(!truth[0].candidates[1].actual_discard);
        assert_eq!(truth[0].candidates[1].deal_in_points, None);
        assert_eq!(truth[0].candidates[0].ron, vec![Some(true), None, None]);
        assert_eq!(truth[0].candidates[0].any_ron, Some(true));
    }

    #[test]
    fn repeated_combined_multi_ron_settlement_is_counted_once() {
        let events = vec![
            opening(CLOSED),
            event(json!({"type":"tsumo","actor":0,"pai":"6p"})),
            event(json!({"type":"dahai","actor":0,"pai":"6p","tsumogiri":true})),
            event(json!({"type":"hora","actor":1,"target":0,"deltas":[-9000,4000,0,5000]})),
            event(json!({"type":"hora","actor":3,"target":0,"deltas":[-9000,4000,0,5000]})),
            MjaiEvent::EndKyoku,
        ];
        let truth = build_truth(&events, &[decision(1, discard_candidates())], 0).unwrap();
        assert_eq!(truth[0].candidates[0].deal_in_points, Some(9000));
        assert_eq!(truth[0].candidates[1].deal_in_points, None);
    }

    #[test]
    fn zero_loss_requires_a_resolved_no_win_window_and_riichi_matches_its_own_candidate() {
        let mut events = vec![
            opening(CLOSED),
            event(json!({"type":"tsumo","actor":0,"pai":"6p"})),
            event(json!({"type":"reach","actor":0})),
            event(json!({"type":"dahai","actor":0,"pai":"6p","tsumogiri":true})),
            MjaiEvent::EndKyoku,
        ];
        let candidates = json!([
            {"action":{"type":"dahai","actor":0,"pai":"6p","tsumogiri":true}},
            {"action":{"type":"reach","actor":0},"continuation":{"type":"dahai","actor":0,"pai":"6p","tsumogiri":true}}
        ]);
        let decisions = vec![decision(1, candidates)];
        let truth = build_truth(&events, &decisions, 0).unwrap();
        assert!(!truth[0].candidates[0].actual_discard);
        assert!(truth[0].candidates[1].actual_discard);
        assert_eq!(truth[0].candidates[1].deal_in_points, None);
        events[4] = event(json!({"type":"tsumo","actor":1,"pai":"9s"}));
        assert_eq!(
            build_truth(&events, &decisions, 0).unwrap()[0].candidates[1].deal_in_points,
            Some(0)
        );
    }
}
