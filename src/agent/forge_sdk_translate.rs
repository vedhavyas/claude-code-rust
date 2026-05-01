//! Translate `forge_sdk::Message` into upstream's `BridgeEvent`.
//!
//! The Node bridge in `agent-sdk/` consumes raw SDK messages and emits
//! `BridgeEvent`s through stdout. The forge-sdk-backed bridge lives
//! in-process so it skips the NDJSON round-trip, but the `BridgeEvent`
//! shape on the consumer side stays identical -- `app/connect/event_dispatch.rs`
//! is unaware which backend produced the events.
//!
//! TODO(cleanup-phase): once the TUI talks directly to the forge daemon
//! (or to `forge_sdk::Client` natively) this translation layer can be
//! deleted. The `BridgeEvent` shape was inherited from the upstream
//! Node-bridge wire format; the daemon will prefer the SDK's own
//! `Message` enum once the consumer side migrates. Drop together with
//! the `BridgeEvent` enum and the `forge_sdk_event_loop` adapter in
//! `connect/bridge_lifecycle.rs`.
//!
//! Variants currently translated:
//! - `Message::Assistant` -> one `SessionUpdate` per content block
//!   (`Text` / `Thinking` -> `AgentMessageChunk` / `AgentThoughtChunk`,
//!   `ToolUse` -> `ToolCall`).
//! - `Message::Result` -> `TurnComplete` (success) or `TurnError`
//!   (when `is_error`), carrying the SDK subtype as `error_kind` +
//!   `sdk_result_subtype` so upstream's classification logic in
//!   `agent/error_handling.rs` can branch on it.
//!
//! Variants currently dropped (added as needed):
//! `User`, `System`, `TaskStarted/Progress/Notification`,
//! `RateLimitEvent`, `StreamEvent`, `Error`, `Unknown`.

use forge_sdk::{AssistantEnvelope, ContentBlock as SdkContentBlock, Message as SdkMessage};

use crate::agent::bridge::tooling::create_tool_call;
use crate::agent::types::{ContentBlock as TuiContentBlock, SessionUpdate};
use crate::agent::wire::BridgeEvent;

/// Translate one SDK message into zero, one, or many `BridgeEvent`s.
/// One `Message::Assistant` commonly fans out into multiple
/// `SessionUpdate` envelopes (one per content block), hence the Vec.
#[must_use]
pub fn translate_message(msg: SdkMessage) -> Vec<BridgeEvent> {
    match msg {
        SdkMessage::Assistant {
            message: envelope,
            session_id,
            ..
        } => assistant_to_events(&session_id, &envelope),
        SdkMessage::Result {
            subtype,
            is_error,
            session_id,
            ..
        } => {
            let event = if is_error {
                BridgeEvent::TurnError {
                    session_id,
                    message: subtype.clone(),
                    error_kind: Some(subtype.clone()),
                    sdk_result_subtype: Some(subtype),
                    assistant_error: None,
                    terminal_reason: None,
                }
            } else {
                BridgeEvent::TurnComplete { session_id, terminal_reason: None }
            };
            vec![event]
        }
        SdkMessage::System {
            subtype,
            session_id,
            data,
        } if subtype == "elicitation_request" => {
            elicitation_request_to_event(session_id.unwrap_or_default(), &data)
                .into_iter()
                .collect()
        }
        _ => {
            tracing::debug!(
                target: crate::logging::targets::BRIDGE_PROTOCOL,
                "forge_sdk_translate: variant not yet translated",
            );
            Vec::new()
        }
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
    let url = data
        .get("url")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
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

fn assistant_to_events(session_id: &str, envelope: &AssistantEnvelope) -> Vec<BridgeEvent> {
    envelope
        .content
        .iter()
        .filter_map(|block| {
            content_block_to_update(block).map(|update| BridgeEvent::SessionUpdate {
                session_id: session_id.to_owned(),
                update,
            })
        })
        .collect()
}

fn content_block_to_update(block: &SdkContentBlock) -> Option<SessionUpdate> {
    match block {
        SdkContentBlock::Text { text } => Some(SessionUpdate::AgentMessageChunk {
            content: TuiContentBlock::Text { text: text.clone() },
        }),
        SdkContentBlock::Thinking { thinking, .. } => Some(SessionUpdate::AgentThoughtChunk {
            content: TuiContentBlock::Text { text: thinking.clone() },
        }),
        SdkContentBlock::ToolUse { id, name, input } => {
            // Status starts at "in_progress" mirroring upstream's
            // emitToolCall — `create_tool_call` defaults to "pending"
            // so we patch it here to match.
            let mut tool_call = create_tool_call(id, name, input, None);
            "in_progress".clone_into(&mut tool_call.status);
            Some(SessionUpdate::ToolCall { tool_call })
        }
        // ToolResult, ServerToolUse / ServerToolResult, Document,
        // Image, Unknown -- not surfaced to the UI yet. The lifted UI
        // renderers learn these shapes incrementally.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assistant_text(session: &str, text: &str) -> SdkMessage {
        let envelope: AssistantEnvelope = serde_json::from_value(json!({
            "id": "msg_1",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{ "type": "text", "text": text }],
        }))
        .unwrap();
        SdkMessage::Assistant {
            message: envelope,
            session_id: session.to_owned(),
            parent_tool_use_id: None,
            error: None,
            uuid: None,
        }
    }

    #[test]
    fn assistant_text_emits_agent_message_chunk() {
        let events = translate_message(assistant_text("sess_1", "hello"));
        assert_eq!(events.len(), 1);
        let BridgeEvent::SessionUpdate { session_id, update } = &events[0] else {
            panic!("expected SessionUpdate, got {:?}", events[0]);
        };
        assert_eq!(session_id, "sess_1");
        let SessionUpdate::AgentMessageChunk { content } = update else {
            panic!("expected AgentMessageChunk, got {update:?}");
        };
        let TuiContentBlock::Text { text } = content else {
            panic!("expected Text content, got {content:?}");
        };
        assert_eq!(text, "hello");
    }

    #[test]
    fn assistant_text_and_tool_use_fans_out_two_events() {
        let envelope: AssistantEnvelope = serde_json::from_value(json!({
            "id": "msg_1",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                { "type": "text", "text": "running ls" },
                {
                    "type": "tool_use",
                    "id": "tu_x",
                    "name": "Bash",
                    "input": { "command": "ls" },
                },
            ],
        }))
        .unwrap();
        let msg = SdkMessage::Assistant {
            message: envelope,
            session_id: "sess_1".to_owned(),
            parent_tool_use_id: None,
            error: None,
            uuid: None,
        };
        let events = translate_message(msg);
        assert_eq!(events.len(), 2);
        // First event: text chunk.
        let BridgeEvent::SessionUpdate { update: u0, .. } = &events[0] else {
            panic!("first not SessionUpdate");
        };
        let SessionUpdate::AgentMessageChunk { .. } = u0 else {
            panic!("first not AgentMessageChunk");
        };
        // Second event: tool call.
        let BridgeEvent::SessionUpdate { update: u1, .. } = &events[1] else {
            panic!("second not SessionUpdate");
        };
        let SessionUpdate::ToolCall { tool_call } = u1 else {
            panic!("second not ToolCall");
        };
        assert_eq!(tool_call.tool_call_id, "tu_x");
        // create_tool_call extracts command for Bash-tool title.
        assert_eq!(tool_call.title, "ls");
        assert_eq!(tool_call.kind, "execute");
        assert_eq!(tool_call.raw_input, Some(json!({ "command": "ls" })));
    }

    #[test]
    fn assistant_thinking_emits_thought_chunk() {
        let envelope: AssistantEnvelope = serde_json::from_value(json!({
            "id": "msg_1",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{ "type": "thinking", "thinking": "Hmm.", "signature": "sig" }],
        }))
        .unwrap();
        let msg = SdkMessage::Assistant {
            message: envelope,
            session_id: "sess_1".to_owned(),
            parent_tool_use_id: None,
            error: None,
            uuid: None,
        };
        let events = translate_message(msg);
        assert_eq!(events.len(), 1);
        let BridgeEvent::SessionUpdate { update, .. } = &events[0] else {
            panic!();
        };
        let SessionUpdate::AgentThoughtChunk { content } = update else {
            panic!();
        };
        let TuiContentBlock::Text { text } = content else {
            panic!();
        };
        assert_eq!(text, "Hmm.");
    }

    #[test]
    fn result_success_emits_turn_complete() {
        let msg: SdkMessage = serde_json::from_value(json!({
            "type": "result",
            "subtype": "success",
            "session_id": "sess_1",
            "is_error": false,
            "num_turns": 1,
            "duration_ms": 100,
            "duration_api_ms": 80,
        }))
        .unwrap();
        let events = translate_message(msg);
        assert_eq!(events.len(), 1);
        let BridgeEvent::TurnComplete { session_id, .. } = &events[0] else {
            panic!();
        };
        assert_eq!(session_id, "sess_1");
    }

    #[test]
    fn result_error_emits_turn_error_with_subtype() {
        let msg: SdkMessage = serde_json::from_value(json!({
            "type": "result",
            "subtype": "error_during_execution",
            "session_id": "sess_1",
            "is_error": true,
            "num_turns": 2,
            "duration_ms": 50,
            "duration_api_ms": 30,
        }))
        .unwrap();
        let events = translate_message(msg);
        assert_eq!(events.len(), 1);
        let BridgeEvent::TurnError {
            session_id,
            error_kind,
            sdk_result_subtype,
            ..
        } = &events[0]
        else {
            panic!();
        };
        assert_eq!(session_id, "sess_1");
        assert_eq!(error_kind.as_deref(), Some("error_during_execution"));
        assert_eq!(sdk_result_subtype.as_deref(), Some("error_during_execution"));
    }
}
