//! Completed official records enrich a private offline source. No event is
//! published, and recorded public actions and event indices remain unchanged.

use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use prost_reflect::{DynamicMessage, SerializeOptions};
use serde_json::{json, Value};

use super::{
    parser::{MessageType, ParsedMessage, POOL},
    tile::{compare_pai, ms_to_mjai},
    MajsoulBridge,
};
use crate::schema::MjaiEvent;

fn decode(name: &str, data: &[u8]) -> Result<Value> {
    let descriptor = POOL
        .get_message_by_name(name)
        .with_context(|| format!("unknown record message {name}"))?;
    let message =
        DynamicMessage::decode(descriptor, data).with_context(|| format!("decode {name}"))?;
    let options = SerializeOptions::new()
        .stringify_64_bit_integers(false)
        .use_proto_field_name(true)
        .skip_default_fields(false);
    let mut serializer = serde_json::Serializer::new(Vec::new());
    message.serialize_with_options(&mut serializer, &options)?;
    Ok(serde_json::from_slice(&serializer.into_inner())?)
}

fn bytes(value: &Value) -> Result<Vec<u8>> {
    BASE64
        .decode(value.as_str().context("record bytes are missing")?)
        .context("decode record bytes")
}

fn records(data: &[u8]) -> Result<Vec<(String, Value)>> {
    ensure!(
        !data.is_empty() && data.len() <= 8 * 1024 * 1024,
        "record payload size is invalid"
    );
    let detail = match decode("lq.Wrapper", data) {
        Ok(wrapper)
            if matches!(
                wrapper["name"].as_str(),
                Some(".lq.GameDetailRecords" | "lq.GameDetailRecords")
            ) =>
        {
            decode("lq.GameDetailRecords", &bytes(&wrapper["data"])?)?
        }
        Ok(wrapper)
            if wrapper["name"]
                .as_str()
                .is_some_and(|name| name.starts_with(".lq.") || name.starts_with("lq.")) =>
        {
            bail!("record wrapper is not GameDetailRecords")
        }
        _ => decode("lq.GameDetailRecords", data)?,
    };
    let version = detail["version"].as_u64().unwrap_or(0);
    let encoded: Vec<&Value> = if version == 0 {
        detail["records"]
            .as_array()
            .context("record list is missing")?
            .iter()
            .collect()
    } else {
        ensure!(version >= 210715, "unsupported record version {version}");
        detail["actions"]
            .as_array()
            .context("record actions are missing")?
            .iter()
            .filter_map(|action| {
                action
                    .get("result")
                    .filter(|result| result.as_str().is_some_and(|text| !text.is_empty()))
            })
            .collect()
    };
    ensure!(
        !encoded.is_empty() && encoded.len() <= 50_000,
        "record action count is invalid"
    );
    encoded
        .into_iter()
        .enumerate()
        .map(|(index, encoded)| {
            let wrapper = decode("lq.Wrapper", &bytes(encoded)?)?;
            let name = wrapper["name"]
                .as_str()
                .context("record action name is missing")?
                .trim_start_matches('.');
            ensure!(
                name.starts_with("lq.Record"),
                "unsupported record action at {index}"
            );
            let payload = decode(name, &bytes(&wrapper["data"])?)?;
            Ok((name.trim_start_matches("lq.").to_owned(), payload))
        })
        .collect()
}

fn complete_hands(payload: &Value) -> Result<(Vec<Vec<String>>, String)> {
    let dealer = payload["ju"].as_u64().context("record dealer is missing")? as usize;
    ensure!(
        dealer < 4
            && payload["scores"]
                .as_array()
                .is_some_and(|scores| scores.len() == 4),
        "record is not a four-player round"
    );
    let mut hands = Vec::with_capacity(4);
    let mut opening = None;
    for seat in 0..4 {
        let field = format!("tiles{seat}");
        let raw = payload[&field]
            .as_array()
            .context("record starting hand is missing")?;
        ensure!(
            raw.len() == 13 + usize::from(seat == dealer),
            "record starting hand is incomplete"
        );
        let mut hand = raw
            .iter()
            .map(|tile| {
                ms_to_mjai(tile.as_str().context("record tile is invalid")?).map(str::to_owned)
            })
            .collect::<Result<Vec<_>>>()?;
        if seat == dealer {
            opening = hand.pop();
        }
        hand.sort_by(|a, b| compare_pai(a, b));
        hands.push(hand);
    }
    Ok((hands, opening.context("record opening draw is missing")?))
}

fn convert(data: &[u8], seat: u8) -> Result<Vec<MjaiEvent>> {
    let mut bridge = MajsoulBridge::new(None, None);
    bridge.seat = Some(seat);
    let mut events = Vec::new();
    let mut in_round = false;
    for (name, mut payload) in records(data)? {
        let action = match name.as_str() {
            "RecordNewRound" => "ActionNewRound",
            "RecordDealTile" => "ActionDealTile",
            "RecordDiscardTile" => "ActionDiscardTile",
            "RecordChiPengGang" => "ActionChiPengGang",
            "RecordAnGangAddGang" => "ActionAnGangAddGang",
            "RecordHule" => "ActionHule",
            "RecordNoTile" => "ActionNoTile",
            "RecordLiuJu" => "ActionLiuJu",
            _ => bail!("unsupported record action {name}"),
        };
        let hands = if action == "ActionNewRound" {
            ensure!(
                !in_round,
                "record starts a round before the previous one ended"
            );
            let hands = complete_hands(&payload)?;
            payload["tiles"] = payload[format!("tiles{seat}")].clone();
            if payload["doras"]
                .as_array()
                .is_none_or(|doras| doras.is_empty())
            {
                payload["doras"] = json!([payload["dora"]]);
            }
            in_round = true;
            Some(hands)
        } else {
            ensure!(in_round, "record action appears outside a round");
            None
        };
        if let Some(actor) = payload.get("seat").and_then(Value::as_u64) {
            ensure!(actor < 4, "record actor is outside the four-player table");
        }
        let draw = if action == "ActionDealTile" {
            Some(
                ms_to_mjai(payload["tile"].as_str().context("record draw is missing")?)?.to_owned(),
            )
        } else {
            None
        };
        let message = ParsedMessage {
            msg_type: MessageType::Notify,
            msg_id: None,
            method_name: ".lq.ActionPrototype".into(),
            payload: json!({"name":action, "data":payload}),
        };
        let mut mapped = bridge.handle_action_prototype(&message);
        // The live bridge logs malformed actions and returns no action. Do not
        // treat a lone queued reach_accepted as successful record conversion.
        ensure!(
            mapped
                .iter()
                .any(|event| !matches!(event, MjaiEvent::ReachAccepted { .. })),
            "failed to convert {name}"
        );
        for event in &mut mapped {
            match event {
                MjaiEvent::StartKyoku { tehais, .. } => {
                    *tehais = hands
                        .as_ref()
                        .context("unexpected round mapping")?
                        .0
                        .clone()
                }
                MjaiEvent::Tsumo { pai, .. } => {
                    *pai = draw
                        .as_ref()
                        .or_else(|| hands.as_ref().map(|hands| &hands.1))
                        .context("complete draw is missing")?
                        .clone()
                }
                MjaiEvent::EndKyoku => in_round = false,
                _ => {}
            }
        }
        events.extend(mapped);
        ensure!(events.len() <= 50_000, "record event count is too large");
    }
    ensure!(
        !in_round && matches!(events.last(), Some(MjaiEvent::EndKyoku)),
        "record ends before the final round settlement"
    );
    Ok(events)
}

fn enrich(recorded: &MjaiEvent, complete: &MjaiEvent, seat: u8) -> Result<MjaiEvent> {
    let mut result = recorded.clone();
    match (&mut result, complete) {
        (
            MjaiEvent::StartKyoku {
                tehais,
                num_players,
                ..
            },
            MjaiEvent::StartKyoku { tehais: full, .. },
        ) => {
            ensure!(
                *num_players == 4 && tehais.len() == 4,
                "recording is not four-player"
            );
            for (index, (hand, full_hand)) in tehais.iter_mut().zip(full).enumerate() {
                ensure!(hand.len() == 13, "recorded starting hand is incomplete");
                ensure!(
                    index != seat as usize || !hand.iter().any(|tile| tile == "?"),
                    "recording hides the recorded player's hand"
                );
                let mut remaining = full_hand.clone();
                for known in hand.iter().filter(|tile| *tile != "?") {
                    let position = remaining
                        .iter()
                        .position(|tile| tile == known)
                        .context("record starting hand differs")?;
                    remaining.remove(position);
                }
                let mut missing = remaining.into_iter();
                for tile in hand.iter_mut().filter(|tile| *tile == "?") {
                    *tile = missing.next().context("record starting hand differs")?;
                }
            }
            // Compare the round's public fields independently of hand order.
            let mut comparison = complete.clone();
            if let MjaiEvent::StartKyoku {
                tehais: expected, ..
            } = &mut comparison
            {
                *expected = tehais.clone();
            }
            ensure!(result == comparison, "record round metadata differs");
        }
        (
            MjaiEvent::Tsumo { actor, pai },
            MjaiEvent::Tsumo {
                actor: expected_actor,
                pai: expected,
            },
        ) => {
            ensure!(actor == expected_actor, "record draw actor differs");
            if pai == "?" {
                ensure!(*actor != seat, "recording hides the recorded player's draw");
                *pai = expected.clone();
            } else {
                ensure!(pai == expected, "record draw differs");
            }
        }
        (
            MjaiEvent::Hora {
                actor,
                target,
                deltas,
                ura_markers,
            },
            MjaiEvent::Hora {
                actor: expected_actor,
                target: expected_target,
                deltas: expected_deltas,
                ura_markers: expected_ura,
            },
        ) => {
            ensure!(
                actor == expected_actor
                    && target == expected_target
                    && (deltas.is_none() || deltas == expected_deltas)
                    && (ura_markers.is_none() || ura_markers == expected_ura),
                "record win differs"
            );
        }
        (MjaiEvent::Ryukyoku { deltas }, MjaiEvent::Ryukyoku { deltas: expected }) => {
            ensure!(
                deltas.is_none() || deltas == expected,
                "record draw settlement differs"
            );
        }
        _ => ensure!(recorded == complete, "record public action differs"),
    }
    Ok(result)
}

/// Verify a completed official record against an existing perspective and
/// fill only unknown initial hands and draws, retaining every original index.
pub fn complete_source(data: &[u8], original: &[MjaiEvent], seat: u8) -> Result<Vec<MjaiEvent>> {
    ensure!(
        seat < 4 && original.len() <= 50_000,
        "invalid recording seat or size"
    );
    ensure!(
        matches!(original.first(), Some(MjaiEvent::StartGame { num_players: 4, id, .. }) if id.is_none_or(|id| id == seat))
            && matches!(original.last(), Some(MjaiEvent::EndGame { .. })),
        "recording must be a complete four-player game"
    );
    let complete = convert(data, seat)?;
    let mut source = Vec::with_capacity(original.len());
    let mut next = complete.iter();
    for (index, event) in original.iter().enumerate() {
        match event {
            MjaiEvent::StartGame { .. } if index == 0 => source.push(event.clone()),
            MjaiEvent::EndGame { .. } if index + 1 == original.len() => source.push(event.clone()),
            MjaiEvent::None => source.push(event.clone()),
            _ => source.push(
                enrich(
                    event,
                    next.next()
                        .context("record has fewer events than the recording")?,
                    seat,
                )
                .with_context(|| format!("record mismatch at event {index}"))?,
            ),
        }
    }
    ensure!(
        next.next().is_none(),
        "record has more events than the recording"
    );
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn encode(name: &str, value: Value) -> Vec<u8> {
        DynamicMessage::deserialize(POOL.get_message_by_name(name).unwrap(), value)
            .unwrap()
            .encode_to_vec()
    }

    fn wrapper(name: &str, data: &[u8]) -> Vec<u8> {
        encode(
            "lq.Wrapper",
            json!({"name":format!(".{name}"), "data":BASE64.encode(data)}),
        )
    }

    fn archive(records: &[(String, Value)], version: u32, wrapped: bool) -> Vec<u8> {
        let results: Vec<String> = records
            .iter()
            .map(|(name, payload)| {
                let name = format!("lq.{name}");
                BASE64.encode(wrapper(&name, &encode(&name, payload.clone())))
            })
            .collect();
        let body = if version == 0 {
            json!({"records":results})
        } else {
            let mut actions = vec![json!({"type":2})];
            actions.extend(
                results
                    .into_iter()
                    .map(|result| json!({"type":1,"result":result})),
            );
            json!({"version":version,"actions":actions})
        };
        let data = encode("lq.GameDetailRecords", body);
        if wrapped {
            wrapper("lq.GameDetailRecords", &data)
        } else {
            data
        }
    }

    fn tiles(text: &str) -> Vec<&str> {
        text.split_whitespace().collect()
    }

    fn fixture() -> Vec<(String, Value)> {
        vec![
            (
                "RecordNewRound".into(),
                json!({"chang":0,"ju":0,"ben":0,"liqibang":0,
                "doras":["1z"],"scores":[25000,25000,25000,25000],
                "tiles0":tiles("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 4p 0m"),
                "tiles1":tiles("1s 2s 3s 4s 5s 6s 7s 8s 9s 1z 2z 3z 4z"),
                "tiles2":tiles("1m 2m 3m 4m 5m 6m 7m 8m 9m 5p 6p 7p 8p"),
                "tiles3":tiles("1s 2s 3s 4s 5s 6s 7s 8s 9s 5z 6z 7z 9p")}),
            ),
            (
                "RecordDiscardTile".into(),
                json!({"seat":0,"tile":"0m","moqie":true,"is_liqi":true,"doras":["1z"]}),
            ),
            (
                "RecordDealTile".into(),
                json!({"seat":1,"tile":"7z","doras":["1z"]}),
            ),
            (
                "RecordDiscardTile".into(),
                json!({"seat":1,"tile":"7z","moqie":true,"doras":["1z"]}),
            ),
            (
                "RecordHule".into(),
                json!({"hules":[{"seat":2,"zimo":false}],"delta_scores":[0,-3900,3900,0]}),
            ),
        ]
    }

    fn original() -> Vec<MjaiEvent> {
        serde_json::from_value(json!([
            {"type":"start_game","id":0,"names":["A","B","C","D"]},
            {"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"dora_marker":"E",
                "scores":[25000,25000,25000,25000],"tehais":[tiles("1m 2m 3m 4m 5m 6m 7m 8m 9m 1p 2p 3p 4p"),vec!["?";13],vec!["?";13],vec!["?";13]]},
            {"type":"tsumo","actor":0,"pai":"5mr"},
            {"type":"reach","actor":0},
            {"type":"dahai","actor":0,"pai":"5mr","tsumogiri":true},
            {"type":"reach_accepted","actor":0},
            {"type":"tsumo","actor":1,"pai":"?"},
            {"type":"dahai","actor":1,"pai":"C","tsumogiri":true},
            {"type":"hora","actor":2,"target":1,"deltas":[0,-3900,3900,0]},
            {"type":"end_kyoku"},{"type":"end_game"}
        ])).unwrap()
    }

    #[test]
    fn both_record_layouts_fill_hidden_tiles_without_changing_public_events() {
        let original = original();
        for version in [0, 210715] {
            for wrapped in [false, true] {
                let source =
                    complete_source(&archive(&fixture(), version, wrapped), &original, 0).unwrap();
                assert_eq!(source.len(), original.len());
                assert_eq!(
                    source[6],
                    MjaiEvent::Tsumo {
                        actor: 1,
                        pai: "C".into()
                    }
                );
                let MjaiEvent::StartKyoku { tehais, .. } = &source[1] else {
                    panic!()
                };
                assert_eq!(tehais[1][9..], ["E", "S", "W", "N"]);
                assert!(tehais.iter().flatten().all(|tile| tile != "?"));
                for index in [0, 2, 3, 4, 5, 7, 8, 9, 10] {
                    assert_eq!(source[index], original[index]);
                }
                let mut visible = source;
                if let MjaiEvent::StartKyoku { tehais, .. } = &mut visible[1] {
                    for hand in &mut tehais[1..] {
                        hand.fill("?".into());
                    }
                }
                visible[6] = original[6].clone();
                assert_eq!(visible, original);
            }
        }
    }

    #[test]
    fn own_hand_draw_and_public_action_mismatches_are_rejected() {
        let data = archive(&fixture(), 210715, true);
        for index in [1, 2, 7] {
            let mut events = original();
            match &mut events[index] {
                MjaiEvent::StartKyoku { tehais, .. } => tehais[0][0] = "9s".into(),
                MjaiEvent::Tsumo { pai, .. } | MjaiEvent::Dahai { pai, .. } => *pai = "9s".into(),
                _ => unreachable!(),
            }
            assert!(complete_source(&data, &events, 0).is_err());
        }
        assert!(complete_source(&data, &original(), 1).is_err());
    }

    #[test]
    fn incomplete_hidden_draw_and_three_player_records_are_rejected() {
        let mut missing_draw = fixture();
        missing_draw[2].1["tile"] = json!("");
        assert!(complete_source(&archive(&missing_draw, 0, true), &original(), 0).is_err());
        let mut three_player = fixture();
        three_player[0].1["tiles3"] = json!([]);
        assert!(complete_source(&archive(&three_player, 0, true), &original(), 0).is_err());
        let mut incomplete = fixture();
        incomplete.pop();
        assert!(complete_source(&archive(&incomplete, 0, true), &original(), 0).is_err());
        let mut unsupported = fixture();
        unsupported.insert(1, ("RecordBaBei".into(), json!({"seat":0})));
        assert!(complete_source(&archive(&unsupported, 0, true), &original(), 0).is_err());
        assert!(complete_source(&archive(&fixture(), 42, true), &original(), 0).is_err());
    }

    #[test]
    fn legacy_dora_and_non_dealer_opening_are_completed() {
        let mut records = fixture();
        records[0].1["doras"] = json!([]);
        records[0].1["dora"] = json!("1z");
        let mapped = convert(&archive(&records, 0, true), 1).unwrap();
        assert_eq!(
            mapped[1],
            MjaiEvent::Tsumo {
                actor: 0,
                pai: "5mr".into()
            }
        );
        let MjaiEvent::StartKyoku {
            tehais,
            dora_marker,
            ..
        } = &mapped[0]
        else {
            panic!()
        };
        assert_eq!(dora_marker, "E");
        assert!(tehais.iter().flatten().all(|tile| tile != "?"));
    }

    #[test]
    fn kan_dora_timing_uses_the_live_bridge_rules() {
        for kind in [2, 3] {
            let records = vec![
                fixture().remove(0),
                (
                    "RecordAnGangAddGang".into(),
                    json!({"seat":0,"type":kind,"tiles":"1m","doras":["1z"]}),
                ),
                (
                    "RecordDealTile".into(),
                    json!({"seat":0,"tile":"9p","doras":["1z","2z"]}),
                ),
                (
                    "RecordDiscardTile".into(),
                    json!({"seat":0,"tile":"9p","moqie":true,"doras":["1z","2z"]}),
                ),
                ("RecordLiuJu".into(), json!({"type":1})),
            ];
            let mapped = convert(&archive(&records, 210715, true), 0).unwrap();
            let dora = mapped
                .iter()
                .position(|event| matches!(event, MjaiEvent::Dora { .. }))
                .unwrap();
            let draw = mapped
                .iter()
                .position(|event| matches!(event, MjaiEvent::Tsumo { pai, .. } if pai == "9p"))
                .unwrap();
            if kind == 3 {
                assert_eq!(dora + 1, draw);
            } else {
                assert_eq!(draw + 1, dora);
            }
        }
    }
}
