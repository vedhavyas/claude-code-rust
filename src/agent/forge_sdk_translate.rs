//! Thin shim around `bridge::message_handlers::handle_sdk_message`.
//!
//! The Node bridge in `agent-sdk/` historically consumed raw SDK
//! messages and emitted `BridgeEvent`s through stdout. The forge-sdk-
//! backed bridge lives in-process so it skips the NDJSON round-trip,
//! but the `BridgeEvent` shape on the consumer side stays identical
//! — `app/connect/event_dispatch.rs` is unaware which backend
//! produced the events.
//!
//! As of Stage 3 most translation lives in
//! `crate::agent::bridge::message_handlers::handle_sdk_message` —
//! that path takes `&mut BridgeSession` so it can pair `tool_use` ↔
//! `tool_result` and track per-session state across messages.
//!
//! This module retains only the `elicitation_request` system-subtype
//! handler. Elicitation is a top-level `BridgeEvent`, not a
//! `SessionUpdate`, and its lifetime doesn't depend on the bridge
//! session state, so it stays here for now.

use forge_sdk::Message as SdkMessage;

use crate::agent::wire::BridgeEvent;

/// Translate one SDK message into zero or more `BridgeEvent`s.
/// Today only `system/elicitation_request` is handled here; everything
/// else flows through `bridge::message_handlers::handle_sdk_message`.
#[must_use]
pub fn translate_message(msg: SdkMessage) -> Vec<BridgeEvent> {
    match msg {
        SdkMessage::System { subtype, session_id, data } if subtype == "elicitation_request" => {
            elicitation_request_to_event(session_id.unwrap_or_default(), &data)
                .into_iter()
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Translate a `system/elicitation_request` SDK message into the
/// upstream `BridgeEvent::ElicitationRequest` shape so the MCP overlay
/// in the TUI can prompt the user for the form payload.
fn elicitation_request_to_event(
    session_id: String,
    data: &serde_json::Value,
) -> Option<BridgeEvent> {
    use crate::agent::types::{ElicitationMode, ElicitationRequest};
    let request_id = data.get("request_id").and_then(|v| v.as_str())?.to_owned();
    let server_name = data
        .get("server_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();
    let message = data
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();
    let mode = match data.get("mode").and_then(|v| v.as_str()) {
        Some("url") => ElicitationMode::Url,
        _ => ElicitationMode::Form,
    };
    let url = data.get("url").and_then(|v| v.as_str()).map(str::to_owned);
    let elicitation_id = data
        .get("elicitation_id")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let requested_schema = data.get("requested_schema").cloned();
    Some(BridgeEvent::ElicitationRequest {
        session_id,
        request: ElicitationRequest {
            request_id,
            server_name,
            message,
            mode,
            url,
            elicitation_id,
            requested_schema,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn elicitation_request_translates_to_top_level_bridge_event() {
        let data = json!({
            "request_id": "req-1",
            "server_name": "ms-server",
            "message": "approve?",
            "mode": "form",
        });
        let msg = SdkMessage::System {
            subtype: "elicitation_request".to_owned(),
            session_id: Some("sess".to_owned()),
            data,
        };
        let events = translate_message(msg);
        assert_eq!(events.len(), 1);
        let BridgeEvent::ElicitationRequest { session_id, request } = &events[0] else {
            panic!("expected ElicitationRequest");
        };
        assert_eq!(session_id, "sess");
        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.server_name, "ms-server");
    }

    #[test]
    fn assistant_messages_are_now_handled_by_bridge_module() {
        // translate_message is a thin shim; assistant routing lives in
        // bridge::message_handlers. Verify the shim returns an empty
        // Vec for non-elicitation variants so the worker doesn't
        // double-emit.
        let env: forge_sdk::AssistantEnvelope = serde_json::from_value(json!({
            "id": "msg_1",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{ "type": "text", "text": "hi" }],
        }))
        .unwrap();
        let msg = SdkMessage::Assistant {
            message: env,
            session_id: "sess".to_owned(),
            parent_tool_use_id: None,
            error: None,
            uuid: None,
        };
        assert!(translate_message(msg).is_empty());
    }
}
