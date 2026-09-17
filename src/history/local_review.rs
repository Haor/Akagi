//! Offline review follows the recorded moves, never the model's suggestions.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::bot::BotRunner;
use crate::game_state::mahgen_view::MahgenView;
use crate::game_state::snapshot::GameStateSnapshot;
use crate::game_state::tracker::GameTracker;
use crate::schema::MjaiEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalReviewDecision {
    pub event_index: usize,
    pub round: String,
    pub turn: usize,
    pub hand: Vec<String>,
    pub trigger: MjaiEvent,
    pub actual: Option<MjaiEvent>,
    pub recommended: MjaiEvent,
    pub matches: Option<bool>,
    pub meta: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalReviewResult {
    pub history_id: String,
    pub bot: String,
    pub created_at: String,
    pub seat: u8,
    pub event_count: usize,
    pub compared: usize,
    pub matched: usize,
    pub events: Vec<MjaiEvent>,
    pub decisions: Vec<LocalReviewDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalReviewFrame {
    pub game: GameStateSnapshot,
    pub view: MahgenView,
}

/// Frame i shows the state immediately after applying recorded event i.
/// Lifecycle gaps retain their indices as None instead of borrowing a board
/// from a previous round. This tracker never subscribes to the live game bus.
pub fn build_frames(review: &LocalReviewResult) -> Result<Vec<Option<LocalReviewFrame>>> {
    if review.event_count != review.events.len() {
        bail!("local review event count does not match its recorded stream");
    }
    let num_players = match review.events.first() {
        Some(MjaiEvent::StartGame { num_players, .. }) => *num_players,
        _ => bail!("local review has no start_game event"),
    };
    // Reapply the public boundary even for old or imported cached results;
    // the saved review's seat owns this perspective, not a stale event id.
    let events = perspective(&review.events, review.seat, num_players)?;
    let mut tracker = GameTracker::new();
    let mut in_round = false;
    let mut frames = Vec::with_capacity(events.len());
    for (index, event) in events.iter().enumerate() {
        tracker
            .handle(event)
            .with_context(|| format!("local review frame failed at event {}", index + 1))?;
        match event {
            MjaiEvent::StartKyoku { .. } => in_round = true,
            MjaiEvent::StartGame { .. } | MjaiEvent::EndKyoku | MjaiEvent::EndGame { .. } => {
                in_round = false;
            }
            _ => {}
        }
        let frame = if in_round {
            tracker.snapshot().map(|mut game| {
                // The view encoder already hides other hands. Keep the raw
                // snapshot equally private for every frontend consumer.
                for player in &mut game.players {
                    if player.seat != review.seat {
                        player.tehai.fill("?".into());
                        if player.drawn_tile.is_some() {
                            player.drawn_tile = Some("?".into());
                        }
                    }
                }
                let view = MahgenView::from_snapshot(&game);
                LocalReviewFrame { game, view }
            })
        } else {
            None
        };
        frames.push(frame);
    }
    Ok(frames)
}

/// Validate before handing any event to the actor. An observer log cannot be
/// reviewed as though it contained a recorded player's hidden hand.
pub fn perspective(events: &[MjaiEvent], seat: u8, num_players: u8) -> Result<Vec<MjaiEvent>> {
    if num_players != 4 || seat >= num_players {
        bail!("local RIN review requires a four-player game with a recorded player seat");
    }
    if !matches!(events.first(), Some(MjaiEvent::StartGame { .. }))
        || !matches!(events.last(), Some(MjaiEvent::EndGame { .. }))
        || !events
            .iter()
            .any(|event| matches!(event, MjaiEvent::StartKyoku { .. }))
    {
        bail!("local review requires a complete start_game … end_game recording");
    }
    for event in events {
        match event {
            MjaiEvent::StartKyoku { tehais, .. }
                if tehais
                    .get(seat as usize)
                    .is_none_or(|hand| hand.len() != 13 || hand.iter().any(|p| p == "?")) =>
            {
                bail!("recording does not contain the recorded player's complete starting hand");
            }
            MjaiEvent::Tsumo { actor, pai } if *actor == seat && pai == "?" => {
                bail!("recording hides the recorded player's draw");
            }
            _ => {}
        }
    }
    crate::bot::native::build_api_events(events, seat, num_players)
        .into_iter()
        .map(|mut event| {
            if event["type"] == "start_game" {
                event["id"] = seat.into();
            }
            serde_json::from_value(event).context("decode censored review event")
        })
        .collect()
}

fn opens_window(event: &MjaiEvent, seat: u8) -> bool {
    match event {
        MjaiEvent::Tsumo { actor, .. }
        | MjaiEvent::Chi { actor, .. }
        | MjaiEvent::Pon { actor, .. }
        | MjaiEvent::Reach { actor, .. } => *actor == seat,
        MjaiEvent::Dahai { actor, .. }
        | MjaiEvent::Kakan { actor, .. }
        | MjaiEvent::Ankan { actor, .. } => *actor != seat,
        _ => false,
    }
}

/// A claim by another player can outrank ours, so absence of our claim in that
/// window is unknown, not proof we passed. Multiple winners remain visible.
fn actual_after(events: &[MjaiEvent], index: usize, seat: u8) -> Option<MjaiEvent> {
    let mut i = index + 1;
    while let Some(event) = events.get(i) {
        match event {
            MjaiEvent::ReachAccepted { .. } | MjaiEvent::Dora { .. } | MjaiEvent::None => {}
            MjaiEvent::Reach { actor, .. } if *actor == seat => {
                let pai = match events.get(i + 1) {
                    Some(MjaiEvent::Dahai { actor, pai, .. }) if *actor == seat => {
                        Some(pai.clone())
                    }
                    _ => None,
                };
                return Some(MjaiEvent::Reach { actor: seat, pai });
            }
            MjaiEvent::Dahai { actor, .. }
            | MjaiEvent::Chi { actor, .. }
            | MjaiEvent::Pon { actor, .. }
            | MjaiEvent::Daiminkan { actor, .. }
            | MjaiEvent::Kakan { actor, .. }
            | MjaiEvent::Ankan { actor, .. }
            | MjaiEvent::Kita { actor, .. } => return (*actor == seat).then(|| event.clone()),
            MjaiEvent::Hora { actor, .. } if *actor == seat => return Some(event.clone()),
            MjaiEvent::Hora { .. } => {
                return events[i..]
                    .iter()
                    .take_while(|e| matches!(e, MjaiEvent::Hora { .. }))
                    .find(|e| matches!(e, MjaiEvent::Hora { actor, .. } if *actor == seat))
                    .cloned();
            }
            MjaiEvent::Tsumo { .. } => return Some(MjaiEvent::None),
            // Exhaustive and abortive draws are not distinguishable in old logs.
            MjaiEvent::Ryukyoku { .. }
            | MjaiEvent::EndKyoku
            | MjaiEvent::EndGame { .. }
            | MjaiEvent::StartGame { .. }
            | MjaiEvent::StartKyoku { .. }
            | MjaiEvent::Reach { .. } => return None,
        }
        i += 1;
    }
    None
}

fn action_key(action: &MjaiEvent) -> Value {
    let mut value = serde_json::to_value(action).expect("MjaiEvent serializes");
    if let Some(object) = value.as_object_mut() {
        for key in ["tsumogiri", "deltas", "ura_markers"] {
            object.remove(key);
        }
        if let Some(Value::Array(consumed)) = object.get_mut("consumed") {
            consumed.sort_by_key(|tile| tile.as_str().unwrap_or("").to_string());
        }
    }
    value
}

fn same_action(actual: &MjaiEvent, recommended: &MjaiEvent) -> bool {
    if let (MjaiEvent::Reach { actor: a, pai: p }, MjaiEvent::Reach { actor: b, pai: q }) =
        (actual, recommended)
    {
        return a == b && (p.is_none() || q.is_none() || p == q);
    }
    action_key(actual) == action_key(recommended)
}

fn update_hand(hand: &mut Vec<String>, event: &MjaiEvent, seat: u8) {
    let mut remove = |tile: &str| {
        if let Some(i) = hand.iter().position(|p| p == tile) {
            hand.remove(i);
        }
    };
    match event {
        MjaiEvent::StartKyoku { tehais, .. } => *hand = tehais[seat as usize].clone(),
        MjaiEvent::Tsumo { actor, pai } if *actor == seat => hand.push(pai.clone()),
        MjaiEvent::Dahai { actor, pai, .. } | MjaiEvent::Kakan { actor, pai, .. }
            if *actor == seat =>
        {
            remove(pai)
        }
        MjaiEvent::Chi {
            actor, consumed, ..
        }
        | MjaiEvent::Pon {
            actor, consumed, ..
        } if *actor == seat => {
            for tile in consumed {
                remove(tile);
            }
        }
        MjaiEvent::Daiminkan {
            actor, consumed, ..
        } if *actor == seat => {
            for tile in consumed {
                remove(tile);
            }
        }
        MjaiEvent::Ankan {
            actor, consumed, ..
        } if *actor == seat => {
            for tile in consumed {
                remove(tile);
            }
        }
        _ => {}
    }
}

/// Public headless entry point used by both IPC and integration tests. Progress
/// receives processed/total events; no shared live tracker or bot is touched.
pub async fn replay(
    runner: &mut dyn BotRunner,
    events: &[MjaiEvent],
    seat: u8,
    num_players: u8,
    history_id: String,
    bot: String,
    mut progress: impl FnMut(usize, usize),
) -> Result<LocalReviewResult> {
    let visible = perspective(events, seat, num_players)?;
    let mut decisions = Vec::new();
    let mut round = String::new();
    let mut turn = 0;
    let mut hand = Vec::new();
    for (index, event) in visible.iter().enumerate() {
        if let MjaiEvent::StartKyoku {
            bakaze,
            kyoku,
            honba,
            ..
        } = event
        {
            round = format!("{bakaze}{kyoku} · {honba}");
            turn = 0;
        }
        if matches!(event, MjaiEvent::Tsumo { actor, .. } if *actor == seat) {
            turn += 1;
        }
        update_hand(&mut hand, event, seat);
        let mut response = runner
            .react(std::slice::from_ref(event))
            .await
            .with_context(|| format!("local review failed at event {}", index + 1))?;
        let continuation = response
            .meta
            .as_ref()
            .is_some_and(|meta| meta["continuation"] == true);
        if let MjaiEvent::Reach { pai, .. } = &mut response.action {
            if pai.is_none() {
                *pai = response
                    .meta
                    .as_ref()
                    .and_then(|meta| meta["candidates"].as_array())
                    .and_then(|candidates| {
                        candidates
                            .iter()
                            .find(|candidate| candidate["selected"] == true)
                    })
                    .and_then(|candidate| candidate["continuation"]["pai"].as_str())
                    .map(str::to_owned);
            }
        }
        let has_decision_meta = response
            .meta
            .as_ref()
            .is_some_and(|meta| meta["decision"] == true);
        if opens_window(event, seat)
            && !continuation
            && (!matches!(response.action, MjaiEvent::None) || has_decision_meta)
        {
            let actual = actual_after(&visible, index, seat);
            let matches = actual.as_ref().map(|a| same_action(a, &response.action));
            decisions.push(LocalReviewDecision {
                event_index: index,
                round: round.clone(),
                turn,
                hand: hand.clone(),
                trigger: event.clone(),
                actual,
                recommended: response.action,
                matches,
                meta: response.meta,
            });
        }
        if index % 20 == 0 || index + 1 == visible.len() {
            progress(index + 1, visible.len());
        }
    }
    Ok(LocalReviewResult {
        history_id,
        bot,
        created_at: chrono::Utc::now().to_rfc3339(),
        seat,
        event_count: visible.len(),
        compared: decisions.iter().filter(|d| d.matches.is_some()).count(),
        matched: decisions.iter().filter(|d| d.matches == Some(true)).count(),
        events: visible,
        decisions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::BotResponse;
    use async_trait::async_trait;
    use serde_json::json;

    fn ev(value: Value) -> MjaiEvent {
        serde_json::from_value(value).unwrap()
    }
    fn stream() -> Vec<MjaiEvent> {
        vec![
            ev(json!({"type":"start_game","names":["a","b","c","d"],"id":0})),
            ev(
                json!({"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":vec![25000;4],"dora_marker":"1p","tehais":vec![vec!["1m";13];4]}),
            ),
            ev(json!({"type":"tsumo","actor":1,"pai":"9p"})),
            ev(json!({"type":"dahai","actor":1,"pai":"9p","tsumogiri":true})),
            ev(json!({"type":"tsumo","actor":0,"pai":"2m"})),
            ev(json!({"type":"dahai","actor":0,"pai":"2m","tsumogiri":true})),
            MjaiEvent::EndKyoku,
            MjaiEvent::end_game(),
        ]
    }

    fn frame_review() -> LocalReviewResult {
        let hand = [
            "1m", "1m", "3m", "4m", "7m", "8m", "9m", "1p", "2p", "3p", "E", "E", "S",
        ];
        let opening = ev(json!({
            "type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,
            "oya":1,"scores":[25000,25000,25000,25000],"dora_marker":"2m",
            "tehais":[vec!["9s";13],vec!["9s";13],hand.to_vec(),vec!["9s";13]]
        }));
        let mut next_round = opening.clone();
        if let MjaiEvent::StartKyoku { kyoku, oya, .. } = &mut next_round {
            *kyoku = 2;
            *oya = 2;
        }
        let events = vec![
            // Deliberately inconsistent id: the result's seat is authoritative.
            ev(json!({"type":"start_game","names":["a","b","c","d"],"id":0})),
            opening,
            ev(json!({"type":"tsumo","actor":1,"pai":"1m"})),
            ev(json!({"type":"dahai","actor":1,"pai":"1m","tsumogiri":false})),
            ev(json!({"type":"pon","actor":2,"target":1,"pai":"1m","consumed":["1m","1m"]})),
            ev(json!({"type":"dahai","actor":2,"pai":"S","tsumogiri":false})),
            ev(json!({"type":"tsumo","actor":3,"pai":"9p"})),
            ev(json!({"type":"dahai","actor":3,"pai":"9p","tsumogiri":true})),
            ev(json!({"type":"ryukyoku","deltas":[0,0,0,0]})),
            MjaiEvent::EndKyoku,
            next_round,
            ev(json!({"type":"tsumo","actor":2,"pai":"5sr"})),
            ev(json!({"type":"dahai","actor":2,"pai":"5sr","tsumogiri":true})),
            MjaiEvent::EndKyoku,
            MjaiEvent::end_game(),
        ];
        LocalReviewResult {
            history_id: "frames".into(),
            bot: "rin-native".into(),
            created_at: String::new(),
            seat: 2,
            event_count: events.len(),
            compared: 0,
            matched: 0,
            events,
            decisions: vec![],
        }
    }

    #[test]
    fn frames_follow_event_indices_and_clear_the_board_between_rounds() {
        let review = frame_review();
        let original = serde_json::to_value(&review).unwrap();
        let frames = build_frames(&review).unwrap();
        assert_eq!(frames.len(), review.events.len());
        assert!(
            frames[0].is_none(),
            "start_game must not expose the tracker's synthetic initial deal"
        );
        let initial = frames[1].as_ref().unwrap();
        assert_eq!(initial.game.our_seat, Some(2));
        assert_eq!(initial.game.players[2].tehai.len(), 13);
        assert!(initial
            .game
            .players
            .iter()
            .all(|player| player.river.is_empty()));
        assert!(
            frames[8].is_some(),
            "keep the final round board through ryukyoku"
        );
        assert!(frames[9].is_none());
        let next = frames[10].as_ref().unwrap();
        assert_eq!(next.game.kyoku, 2);
        assert!(next
            .game
            .players
            .iter()
            .all(|player| player.river.is_empty() && player.melds.is_empty()));
        assert_eq!(frames[11].as_ref().unwrap().game.players[2].tehai.len(), 14);
        assert_eq!(frames[12].as_ref().unwrap().game.players[2].tehai.len(), 13);
        assert!(frames[13].is_none());
        assert!(frames[14].is_none());
        assert_eq!(serde_json::to_value(&review).unwrap(), original);
    }

    #[test]
    fn frames_reuse_called_river_and_meld_rendering_without_revealing_other_hands() {
        let frames = build_frames(&frame_review()).unwrap();
        let before_call = frames[3].as_ref().unwrap();
        assert!(!before_call.game.players[1].river[0].called);
        assert!(!before_call.view.players[1].river.is_empty());
        let after_call = frames[4].as_ref().unwrap();
        assert!(after_call.game.players[1].river[0].called);
        assert!(after_call.view.players[1].river.is_empty());
        assert_eq!(after_call.game.players[2].melds[0].from_who, 1);
        assert_eq!(after_call.view.players[2].melds.len(), 1);
        assert_eq!(after_call.game.players[2].tehai.len(), 11);
        assert_eq!(frames[5].as_ref().unwrap().game.players[2].tehai.len(), 10);
        for frame in frames.iter().flatten() {
            for player in &frame.game.players {
                if player.seat == 2 {
                    continue;
                }
                assert!(player.tehai.iter().all(|tile| tile == "?"));
                assert!(player.drawn_tile.as_deref().is_none_or(|tile| tile == "?"));
                let backs = frame.view.players[player.seat as usize]
                    .hand
                    .strip_suffix('z')
                    .unwrap();
                assert!(backs.chars().all(|tile| tile == '0'));
                assert_eq!(backs.len(), player.tehai.len());
            }
        }
    }

    #[test]
    fn frames_reject_broken_index_metadata_and_invalid_perspectives() {
        let mut review = frame_review();
        review.event_count -= 1;
        assert!(build_frames(&review).is_err());
        review.event_count += 1;
        review.seat = 4;
        assert!(build_frames(&review).is_err());
        review.seat = 2;
        if let MjaiEvent::StartKyoku { tehais, .. } = &mut review.events[1] {
            tehais[2].fill("?".into());
        }
        assert!(build_frames(&review).is_err());
    }

    #[test]
    fn projection_hides_other_hands_draws_and_predicted_reach_tile() {
        let mut events = stream();
        events.insert(4, ev(json!({"type":"reach","actor":1,"pai":"5mr"})));
        let visible = perspective(&events, 0, 4).unwrap();
        let MjaiEvent::StartKyoku { tehais, .. } = &visible[1] else {
            panic!()
        };
        assert_eq!(tehais[0], vec!["1m"; 13]);
        assert_eq!(tehais[1], vec!["?"; 13]);
        assert_eq!(visible[2], ev(json!({"type":"tsumo","actor":1,"pai":"?"})));
        assert_eq!(visible[4], ev(json!({"type":"reach","actor":1})));
        assert!(perspective(&events, 4, 4).is_err());
        assert!(perspective(&events, 0, 3).is_err());
    }

    #[test]
    fn mapping_distinguishes_pass_priority_and_riichi() {
        let mut events = stream();
        assert_eq!(actual_after(&events, 3, 0), Some(MjaiEvent::None));
        events[4] =
            ev(json!({"type":"pon","actor":2,"target":1,"pai":"9p","consumed":["9p","9p"]}));
        assert_eq!(actual_after(&events, 3, 0), None);
        events[4] = ev(json!({"type":"hora","actor":2,"target":1}));
        events[5] = ev(json!({"type":"hora","actor":0,"target":1}));
        assert_eq!(actual_after(&events, 3, 0), Some(events[5].clone()));
        events[4] = ev(json!({"type":"reach","actor":0}));
        events[5] = ev(json!({"type":"dahai","actor":0,"pai":"5mr","tsumogiri":false}));
        assert_eq!(
            actual_after(&events, 3, 0),
            Some(ev(json!({"type":"reach","actor":0,"pai":"5mr"})))
        );
        assert!(same_action(
            &events[5],
            &ev(json!({"type":"dahai","actor":0,"pai":"5mr","tsumogiri":true}))
        ));
        assert!(!same_action(
            &events[5],
            &ev(json!({"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}))
        ));
    }

    struct ScriptedBot {
        seen: Vec<MjaiEvent>,
    }
    #[async_trait]
    impl BotRunner for ScriptedBot {
        async fn react(&mut self, events: &[MjaiEvent]) -> Result<BotResponse> {
            self.seen.extend_from_slice(events);
            let event = events.last().unwrap();
            Ok(BotResponse {
                action: if matches!(event, MjaiEvent::Tsumo { actor: 0, .. }) {
                    ev(json!({"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}))
                } else {
                    MjaiEvent::None
                },
                meta: matches!(event, MjaiEvent::Dahai { actor: 1, .. })
                    .then(|| json!({"decision":true})),
            })
        }
        async fn reset(&mut self) -> Result<()> {
            bail!("review must not reset a live bot")
        }
    }

    #[tokio::test]
    async fn offline_replay_feeds_only_censored_recorded_events_and_keeps_passes() {
        let events = stream();
        let mut bot = ScriptedBot { seen: Vec::new() };
        let result = replay(
            &mut bot,
            &events,
            0,
            4,
            "game".into(),
            "test".into(),
            |_, _| {},
        )
        .await
        .unwrap();
        assert_eq!(bot.seen, perspective(&events, 0, 4).unwrap());
        assert_eq!(result.decisions.len(), 2);
        assert_eq!((result.compared, result.matched), (2, 1));
        assert_eq!(result.decisions[1].hand.len(), 14);
        assert!(result.decisions[0].matches.unwrap());
        assert!(!result.decisions[1].matches.unwrap());
    }
}
