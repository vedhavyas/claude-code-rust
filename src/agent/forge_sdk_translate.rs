//! Translate `forge_sdk::Message` into upstream's `BridgeEvent`.
//!
//! The Node bridge in `agent-sdk/` consumes raw SDK messages and emits
//! `BridgeEvent`s through stdout. The forge-sdk-backed bridge lives
//! in-process so it skips the NDJSON round-trip, but the `BridgeEvent`
//! shape on the consumer side stays identical -- `app/connect/event_dispatch.rs`
//! is unaware which backend produced the events.
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

use crate::agent::types::{ContentBlock as TuiContentBlock, SessionUpdate, ToolCall};
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
        _ => {
            tracing::debug!(
                target: crate::logging::targets::BRIDGE_PROTOCOL,
                "forge_sdk_translate: variant not yet translated",
            );
            Vec::new()
        }
    }
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
        SdkContentBlock::ToolUse { id, name, input } => Some(SessionUpdate::ToolCall {
            tool_call: synth_tool_call(id.clone(), name, Some(input)),
        }),
        // ToolResult, ServerToolUse / ServerToolResult, Document,
        // Image, Unknown -- not surfaced to the UI yet. The lifted UI
        // renderers learn these shapes incrementally.
        _ => None,
    }
}

fn synth_tool_call(
    tool_call_id: String,
    tool_name: &str,
    raw_input: Option<&serde_json::Value>,
) -> ToolCall {
    ToolCall {
        tool_call_id,
        title: tool_name.to_owned(),
        kind: "execute".to_owned(),
        status: "pending".to_owned(),
        content: Vec::new(),
        raw_input: raw_input.cloned(),
        raw_output: None,
        output_metadata: None,
        task_metadata: None,
        locations: Vec::new(),
        meta: None,
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
        assert_eq!(tool_call.title, "Bash");
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
