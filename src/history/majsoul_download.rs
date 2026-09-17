//! Fetch completed records through the browser's existing authenticated lobby.
//! Complete hands are private review labels; they never enter the live MJAI bus.

use std::{sync::Arc, time::Duration};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chromiumoxide::page::Page;
use futures_util::StreamExt;
use prost::Message;
use prost_reflect::DynamicMessage;
use serde::Serialize;
use serde_json::Value;
use tokio::sync::{broadcast, Semaphore};

use super::HistoryStore;
use crate::{
    autoplay::AutoplayContext,
    schema::{HistoryEvent, MatchInfo, Platform},
};

const TRANSPORT: &str = include_str!("majsoul_record_transport.js");
const MAX_BYTES: usize = 8 * 1024 * 1024;
static DOWNLOAD_SLOT: Semaphore = Semaphore::const_new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TruthFetchError {
    BrowserUnavailable,
    LoginRequired,
    MissingGameUuid,
    Unsupported,
    RecordNotReady,
    DownloadFailed,
    SourceMismatch,
}

/// Installed before login, including on future navigations. No page reload.
pub async fn install_transport(page: &Page) -> anyhow::Result<()> {
    page.evaluate_on_new_document(TRANSPORT).await?;
    page.evaluate(TRANSPORT).await?;
    Ok(())
}

pub async fn ensure_source(
    store: Arc<HistoryStore>,
    context: &AutoplayContext,
    id: &str,
) -> Result<(), TruthFetchError> {
    use TruthFetchError::*;
    let _permit = DOWNLOAD_SLOT.acquire().await.map_err(|_| DownloadFailed)?;
    if store
        .get_local_review_source(id)
        .map_err(|_| DownloadFailed)?
        .is_some()
    {
        return Ok(());
    }
    let record = store
        .get(id)
        .map_err(|_| DownloadFailed)?
        .ok_or(DownloadFailed)?;
    let seat = record
        .our_seat
        .filter(|seat| *seat < 4)
        .ok_or(Unsupported)?;
    if record.num_players != 4 || record.platform != Platform::Majsoul {
        return Err(Unsupported);
    }
    let uuid = match record.match_info {
        Some(MatchInfo::Majsoul {
            game_uuid: Some(uuid),
            ..
        }) if valid_uuid(&uuid) => uuid,
        Some(MatchInfo::Majsoul { .. }) | None => return Err(MissingGameUuid),
        _ => return Err(Unsupported),
    };
    let page = context
        .page
        .read()
        .await
        .clone()
        .ok_or(BrowserUnavailable)?;
    let expression = format!(
        "globalThis.__akagiRecordTransport ? globalThis.__akagiRecordTransport.fetch({}) : ({{error:'login_required'}})",
        serde_json::to_string(&uuid).map_err(|_| DownloadFailed)?
    );
    let value: Value = tokio::time::timeout(Duration::from_secs(20), page.evaluate(expression))
        .await
        .map_err(|_| RecordNotReady)?
        .map_err(|_| BrowserUnavailable)?
        .into_value()
        .map_err(|_| DownloadFailed)?;
    if let Some(error) = value.get("error").and_then(Value::as_str) {
        return Err(match error {
            "login_required" => LoginRequired,
            "record_not_ready" => RecordNotReady,
            _ => DownloadFailed,
        });
    }
    let encoded = value
        .get("data")
        .and_then(Value::as_str)
        .ok_or(DownloadFailed)?;
    if encoded.len() > MAX_BYTES * 4 / 3 + 4 {
        return Err(DownloadFailed);
    }
    let wire = BASE64.decode(encoded).map_err(|_| DownloadFailed)?;
    let response = decode_response(&wire, &uuid)?;
    let data = match response {
        RecordData::Inline(data) => data,
        RecordData::Url(url) => download_data(&url).await?,
    };
    let id = id.to_string();
    tokio::task::spawn_blocking(move || {
        let original = store
            .get_events(&id)
            .map_err(|_| DownloadFailed)?
            .ok_or(DownloadFailed)?;
        let source = crate::bridge::majsoul::record::complete_source(&data, &original, seat)
            .map_err(|_| SourceMismatch)?;
        store
            .save_local_review_source(&id, &source)
            .map_err(|_| SourceMismatch)
    })
    .await
    .map_err(|_| DownloadFailed)?
}

fn valid_uuid(uuid: &str) -> bool {
    (10..=160).contains(&uuid.len())
        && uuid
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[derive(prost::Message)]
struct Wrapper {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(bytes = "vec", tag = "2")]
    data: Vec<u8>,
}

#[derive(Debug)]
enum RecordData {
    Inline(Vec<u8>),
    Url(String),
}

fn decode_response(wire: &[u8], uuid: &str) -> Result<RecordData, TruthFetchError> {
    use TruthFetchError::*;
    if wire.len() < 3 || wire.len() > MAX_BYTES || wire[0] != 3 {
        return Err(DownloadFailed);
    }
    let wrapper = Wrapper::decode(&wire[3..]).map_err(|_| DownloadFailed)?;
    if !wrapper.name.is_empty() {
        return Err(DownloadFailed);
    }
    let desc = crate::bridge::majsoul::parser::POOL
        .get_message_by_name("lq.ResGameRecord")
        .ok_or(DownloadFailed)?;
    let message =
        DynamicMessage::decode(desc, wrapper.data.as_slice()).map_err(|_| DownloadFailed)?;
    let value = serde_json::to_value(message).map_err(|_| DownloadFailed)?;
    let error = value
        .pointer("/error/code")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if error != 0 {
        return Err(if error == 1004 {
            LoginRequired
        } else {
            RecordNotReady
        });
    }
    if value.pointer("/head/uuid").and_then(Value::as_str) != Some(uuid) {
        return Err(SourceMismatch);
    }
    if let Some(data) = value
        .get("data")
        .and_then(Value::as_str)
        .filter(|data| !data.is_empty())
    {
        let data = BASE64.decode(data).map_err(|_| DownloadFailed)?;
        if data.len() > MAX_BYTES {
            return Err(DownloadFailed);
        }
        return Ok(RecordData::Inline(data));
    }
    value
        .get("dataUrl")
        .or_else(|| value.get("data_url"))
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
        .map(|url| RecordData::Url(url.to_string()))
        .ok_or(RecordNotReady)
}

fn record_url(url: &str) -> Result<reqwest::Url, TruthFetchError> {
    let parsed = reqwest::Url::parse(url).map_err(|_| TruthFetchError::DownloadFailed)?;
    let allowed = parsed.host_str().is_some_and(|host| {
        ["maj-soul.com", "mahjongsoul.com"]
            .iter()
            .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
    });
    if parsed.scheme() != "https"
        || !allowed
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(TruthFetchError::DownloadFailed);
    }
    Ok(parsed)
}

async fn download_data(url: &str) -> Result<Vec<u8>, TruthFetchError> {
    use TruthFetchError::DownloadFailed;
    let url = record_url(url)?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| DownloadFailed)?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|_| DownloadFailed)?
        .error_for_status()
        .map_err(|_| DownloadFailed)?;
    if response
        .content_length()
        .is_some_and(|size| size > MAX_BYTES as u64)
    {
        return Err(DownloadFailed);
    }
    let mut stream = response.bytes_stream();
    let mut data = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| DownloadFailed)?;
        if data.len() + chunk.len() > MAX_BYTES {
            return Err(DownloadFailed);
        }
        data.extend_from_slice(&chunk);
    }
    if data.is_empty() {
        return Err(DownloadFailed);
    }
    Ok(data)
}

pub async fn drive_loop(
    store: Arc<HistoryStore>,
    context: Arc<AutoplayContext>,
    mut rx: broadcast::Receiver<HistoryEvent>,
) {
    loop {
        match rx.recv().await {
            Ok(HistoryEvent::Recorded { record }) => {
                if record.num_players != 4
                    || record.our_seat.is_none()
                    || !matches!(
                        record.match_info,
                        Some(MatchInfo::Majsoul {
                            game_uuid: Some(_),
                            ..
                        })
                    )
                {
                    continue;
                }
                for delay in [1, 5, 20, 60] {
                    tokio::time::sleep(Duration::from_secs(delay)).await;
                    match ensure_source(store.clone(), &context, &record.id).await {
                        Ok(()) => {
                            tracing::info!(target: "akagi::history", "complete review source saved for {}", record.id);
                            break;
                        }
                        Err(
                            TruthFetchError::SourceMismatch
                            | TruthFetchError::MissingGameUuid
                            | TruthFetchError::Unsupported,
                        ) => break,
                        Err(reason) => {
                            tracing::debug!(target: "akagi::history", ?reason, "complete review source pending")
                        }
                    }
                }
            }
            Ok(HistoryEvent::Deleted { .. }) | Err(broadcast::error::RecvError::Lagged(_)) => {}
            Err(broadcast::error::RecvError::Closed) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost_reflect::DeserializeOptions;

    fn response(value: Value) -> Vec<u8> {
        let desc = crate::bridge::majsoul::parser::POOL
            .get_message_by_name("lq.ResGameRecord")
            .unwrap();
        let message =
            DynamicMessage::deserialize_with_options(desc, value, &DeserializeOptions::new())
                .unwrap();
        let mut wire = vec![3, 255, 255];
        wire.extend(
            Wrapper {
                name: String::new(),
                data: message.encode_to_vec(),
            }
            .encode_to_vec(),
        );
        wire
    }

    #[test]
    fn validates_record_identity_and_server_errors_before_reading_private_data() {
        let wire = response(
            serde_json::json!({"head":{"uuid":"test-game-uuid"},"data":BASE64.encode(b"record")}),
        );
        assert!(
            matches!(decode_response(&wire, "test-game-uuid"), Ok(RecordData::Inline(data)) if data == b"record")
        );
        assert!(matches!(
            decode_response(&wire, "other-game-uuid"),
            Err(TruthFetchError::SourceMismatch)
        ));
        assert!(matches!(
            decode_response(
                &response(serde_json::json!({"error":{"code":1004}})),
                "test-game-uuid"
            ),
            Err(TruthFetchError::LoginRequired)
        ));
        assert!(matches!(
            decode_response(
                &response(serde_json::json!({"error":{"code":1201}})),
                "test-game-uuid"
            ),
            Err(TruthFetchError::RecordNotReady)
        ));
    }

    #[test]
    fn accepts_record_urls_only_on_https_platform_hosts() {
        assert!(
            record_url("https://record-v2.maj-soul.com:5333/majsoul/game_record/example").is_ok()
        );
        for url in [
            "http://record.maj-soul.com/a",
            "https://maj-soul.com.evil.test/a",
            "https://127.0.0.1/a",
            "https://user:pass@record.maj-soul.com/a",
        ] {
            assert!(record_url(url).is_err());
        }
        assert!(!valid_uuid("../../private-file"));
    }
}
