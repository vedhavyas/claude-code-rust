//! Live SDK-message dispatcher. Mirrors upstream's
//! `agent-sdk/src/bridge/message_handlers.ts`. The entry point
//! `handle_sdk_message(&mut BridgeSession, &Message, &mut Vec<BridgeEvent>)`
//! routes each `forge_sdk::Message` variant to the right per-subtype
//! handler.
//!
//! Coverage today (incremental — see plan stages):
//! - Assistant content blocks (text, thinking, tool_use, tool_result)
//!   via `handle_assistant_message`.
//! - User tool_result blocks via `handle_user_tool_result_blocks` plus
//!   `tool_use_result` envelope on parent_tool_use_id.
//! - Result subtype "success" / non-"success" with `terminal_reason`
//!   extraction + `finalize_open_tool_calls` via `handle_result_message`.
//! - System subtypes: api_retry, session_state_changed, compact_boundary,
//!   local_command_output, status, init (partial — slash_commands +
//!   agents fan-out + mode updates).
//! - prompt_suggestion / settings_parse_error / rate_limit_event /
//!   tool_progress / tool_use_summary top-level types.
//!
//! Variants still deferred (later stages): auth_status, stream_event
//! partial-messages, full task_started/progress/notification routing,
//! mcp_auth_redirect.

use forge_sdk::Message as SdkMessage;
use serde_json::{Map, Value};

use crate::agent::types::{ContentBlock, SessionUpdate, TerminalReason};
use crate::agent::wire::BridgeEvent;

use super::agents::{emit_available_agents_if_changed, map_available_agents_from_names};
use super::commands::{build_mode_state, refresh_supported_modes_for_session};
use super::session_lifecycle::refresh_current_model;
use super::state::{BridgeSession, PermissionMode};
use super::state_parsing::{
    build_api_retry_update, build_rate_limit_update, normalize_settings_parse_errors,
    parse_fast_mode_state, parse_runtime_session_state,
};
use super::tool_calls::{
    emit_plan_if_todo_write, emit_tool_call, emit_tool_progress_update, emit_tool_result_update,
    emit_tool_summary_update, finalize_open_tool_calls,
};
use super::tooling::{TOOL_RESULT_TYPES, is_tool_use_block_type, unwrap_tool_use_result};

fn push_session_update(out: &mut Vec<BridgeEvent>, session_id: &str, update: SessionUpdate) {
    out.push(BridgeEvent::SessionUpdate { session_id: session_id.to_owned(), update });
}

fn emit_fast_mode_update_if_changed(
    session: &mut BridgeSession,
    next: Option<&Value>,
    out: &mut Vec<BridgeEvent>,
) {
    let Some(state) = parse_fast_mode_state(next) else { return };
    if session.fast_mode_state == state {
        return;
    }
    session.fast_mode_state = state;
    push_session_update(
        out,
        &session.session_id,
        SessionUpdate::FastModeUpdate { fast_mode_state: state },
    );
}

fn handle_content_block(
    session: &mut BridgeSession,
    block: &Value,
    parent_tool_use_id: Option<&str>,
    out: &mut Vec<BridgeEvent>,
) {
    let Some(block_record) = block.as_object() else { return };
    let block_type = block_record.get("type").and_then(Value::as_str).unwrap_or("");

    if block_type == "text" {
        let text = block_record.get("text").and_then(Value::as_str).unwrap_or("");
        if !text.is_empty() {
            push_session_update(
                out,
                &session.session_id,
                SessionUpdate::AgentMessageChunk {
                    content: ContentBlock::Text { text: text.to_owned() },
                },
            );
        }
        return;
    }
    if block_type == "thinking" {
        let text = block_record.get("thinking").and_then(Value::as_str).unwrap_or("");
        if !text.is_empty() {
            push_session_update(
                out,
                &session.session_id,
                SessionUpdate::AgentThoughtChunk {
                    content: ContentBlock::Text { text: text.to_owned() },
                },
            );
        }
        return;
    }
    if is_tool_use_block_type(block_type) {
        let Some(tool_use_id) = block_record.get("id").and_then(Value::as_str) else { return };
        if tool_use_id.is_empty() {
            return;
        }
        let name = block_record.get("name").and_then(Value::as_str).unwrap_or("Tool");
        let empty_input = Value::Object(Map::new());
        let input = block_record.get("input").unwrap_or(&empty_input);
        emit_plan_if_todo_write(session, name, input, out);
        emit_tool_call(session, tool_use_id, name, input, parent_tool_use_id, out);
        return;
    }
    if TOOL_RESULT_TYPES.contains(&block_type) {
        let Some(tool_use_id) = block_record.get("tool_use_id").and_then(Value::as_str) else { return };
        if tool_use_id.is_empty() {
            return;
        }
        let is_error = block_record.get("is_error").and_then(Value::as_bool).unwrap_or(false);
        let raw_content = block_record.get("content");
        emit_tool_result_update(session, tool_use_id, is_error, raw_content, Some(block), out);
    }
}

fn handle_assistant_message(
    session: &mut BridgeSession,
    parent_tool_use_id: Option<&str>,
    error: Option<&str>,
    message: &Value,
    out: &mut Vec<BridgeEvent>,
) {
    if let Some(err) = error
        && !err.is_empty()
    {
        session.last_assistant_error = Some(err.to_owned());
    }
    let Some(message_record) = message.as_object() else { return };

    // Text + thinking blocks come through here too via handle_content_block.
    let Some(content) = message_record.get("content").and_then(Value::as_array) else { return };
    for block in content {
        let Some(block_record) = block.as_object() else { continue };
        let block_type = block_record.get("type").and_then(Value::as_str).unwrap_or("");
        if matches!(block_type, "text" | "thinking")
            || is_tool_use_block_type(block_type)
            || TOOL_RESULT_TYPES.contains(&block_type)
        {
            handle_content_block(session, block, parent_tool_use_id, out);
        }
    }
}

fn handle_user_tool_result_blocks(
    session: &mut BridgeSession,
    parent_tool_use_id: Option<&str>,
    message: &Value,
    out: &mut Vec<BridgeEvent>,
) {
    let Some(message_record) = message.as_object() else { return };
    let Some(content) = message_record.get("content").and_then(Value::as_array) else { return };
    for block in content {
        let Some(block_record) = block.as_object() else { continue };
        let block_type = block_record.get("type").and_then(Value::as_str).unwrap_or("");
        if TOOL_RESULT_TYPES.contains(&block_type) {
            handle_content_block(session, block, parent_tool_use_id, out);
        }
    }
}

fn terminal_reason_from_value(value: Option<&Value>) -> Option<TerminalReason> {
    serde_json::from_value(value?.clone()).ok()
}

fn handle_result_message(
    session: &mut BridgeSession,
    is_error: bool,
    subtype: &str,
    fast_mode_state: Option<&Value>,
    terminal_reason: Option<&Value>,
    errors: Option<&Value>,
    out: &mut Vec<BridgeEvent>,
) {
    emit_fast_mode_update_if_changed(session, fast_mode_state, out);
    let terminal_reason_typed = terminal_reason_from_value(terminal_reason);

    if !is_error && subtype == "success" {
        session.last_assistant_error = None;
        finalize_open_tool_calls(session, "completed", out);
        out.push(BridgeEvent::TurnComplete {
            session_id: session.session_id.clone(),
            terminal_reason: terminal_reason_typed,
        });
        return;
    }

    let error_strings: Vec<String> = errors
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default();
    let assistant_error = session.last_assistant_error.clone();
    finalize_open_tool_calls(session, "failed", out);
    let message = if error_strings.is_empty() {
        if subtype.is_empty() { "turn failed".to_owned() } else { format!("turn failed: {subtype}") }
    } else {
        error_strings.join("\n")
    };
    let error_kind = classify_turn_error_kind(subtype, &error_strings, assistant_error.as_deref());
    out.push(BridgeEvent::TurnError {
        session_id: session.session_id.clone(),
        message,
        error_kind: Some(error_kind.to_owned()),
        sdk_result_subtype: if subtype.is_empty() { None } else { Some(subtype.to_owned()) },
        assistant_error,
        terminal_reason: terminal_reason_typed,
    });
    session.last_assistant_error = None;
}

fn looks_like_auth_required(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("authentication_failed")
        || lower.contains("not authenticated")
        || lower.contains("authentication required")
        || lower.contains("unauthenticated")
        || (lower.contains("401") && lower.contains("auth"))
}

fn classify_turn_error_kind(
    subtype: &str,
    errors: &[String],
    assistant_error: Option<&str>,
) -> &'static str {
    let plan_limit_signals = [
        "error_max_turns",
        "error_max_budget_usd",
        "billing_error",
        "rate_limit",
    ];
    if plan_limit_signals.iter().any(|s| subtype.contains(s)) {
        return "plan_limit";
    }
    if errors.iter().any(|e| plan_limit_signals.iter().any(|s| e.contains(s))) {
        return "plan_limit";
    }
    if assistant_error == Some("authentication_failed") {
        return "auth_required";
    }
    if errors.iter().any(|e| looks_like_auth_required(e)) {
        return "auth_required";
    }
    if assistant_error == Some("server_error") {
        return "internal";
    }
    "other"
}

fn handle_system_init(
    session: &mut BridgeSession,
    incoming_session_id: Option<&str>,
    msg_record: &Map<String, Value>,
    out: &mut Vec<BridgeEvent>,
) {
    if let Some(sid) = incoming_session_id
        && !sid.is_empty()
        && sid != session.session_id
    {
        sid.clone_into(&mut session.session_id);
    }
    if let Some(model) = msg_record.get("model").and_then(Value::as_str) {
        model.clone_into(&mut session.model_id);
    }
    let current_model_changed = refresh_current_model(session, false, &mut Vec::new());
    if let Some(mode_str) = msg_record.get("permissionMode").and_then(Value::as_str)
        && let Some(mode) = PermissionMode::from_wire(mode_str)
    {
        session.mode = Some(mode);
    }
    refresh_supported_modes_for_session(session);
    emit_fast_mode_update_if_changed(session, msg_record.get("fast_mode_state"), out);

    // The initial Connected event was already emitted at spawn (the
    // worker sets session.connected = true there). When system/init
    // lands later — which happens once the user sends their first
    // message — fire the model/mode follow-ups so the footer chip
    // refreshes from "Opus [1M]" (resolved off the alias) to
    // "Claude Opus 4.7" (resolved off the full id the CLI just sent).
    // Mirrors upstream's `if (session.connected) { emitCurrentModelUpdate(...) }`
    // branch in bridge.ts's handleSdkMessage.
    if session.connected {
        if current_model_changed
            && let Some(cm) = session.current_model.clone()
        {
            push_session_update(
                out,
                &session.session_id,
                SessionUpdate::CurrentModelUpdate { current_model: cm },
            );
        }
        if let Some(mode) = session.mode {
            let mode_state = build_mode_state(session, mode);
            push_session_update(
                out,
                &session.session_id,
                SessionUpdate::ModeStateUpdate { mode: mode_state },
            );
        }
    }

    if let Some(slash_commands) = msg_record.get("slash_commands").and_then(Value::as_array) {
        let commands: Vec<crate::agent::types::AvailableCommand> = slash_commands
            .iter()
            .filter_map(|v| v.as_str())
            .map(|name| crate::agent::types::AvailableCommand {
                name: name.to_owned(),
                description: String::new(),
                input_hint: None,
            })
            .collect();
        if !commands.is_empty() {
            push_session_update(
                out,
                &session.session_id,
                SessionUpdate::AvailableCommandsUpdate { commands },
            );
        }
    }

    if session.last_agents_signature.is_none()
        && let Some(agents) = msg_record.get("agents")
    {
        let mapped = map_available_agents_from_names(Some(agents));
        emit_available_agents_if_changed(session, mapped, out);
    }

    let settings_errors = msg_record
        .get("settings_errors")
        .or_else(|| msg_record.get("settingsErrors"));
    if let Some(v) = settings_errors {
        for err in normalize_settings_parse_errors(v) {
            push_session_update(
                out,
                &session.session_id,
                SessionUpdate::SettingsParseError {
                    file: err.file,
                    path: err.path,
                    message: err.message,
                },
            );
        }
    }
}

fn handle_system_status(
    session: &mut BridgeSession,
    msg_record: &Map<String, Value>,
    out: &mut Vec<BridgeEvent>,
) {
    if let Some(mode_str) = msg_record.get("permissionMode").and_then(Value::as_str)
        && let Some(mode) = PermissionMode::from_wire(mode_str)
    {
        session.mode = Some(mode);
        refresh_supported_modes_for_session(session);
        push_session_update(
            out,
            &session.session_id,
            SessionUpdate::CurrentModeUpdate { current_mode_id: mode.as_wire().to_owned() },
        );
    }
    let status = msg_record.get("status");
    if status.and_then(Value::as_str) == Some("compacting") {
        push_session_update(
            out,
            &session.session_id,
            SessionUpdate::SessionStatusUpdate {
                status: crate::agent::types::SessionStatus::Compacting,
            },
        );
    } else if matches!(status, Some(v) if v.is_null()) {
        push_session_update(
            out,
            &session.session_id,
            SessionUpdate::SessionStatusUpdate { status: crate::agent::types::SessionStatus::Idle },
        );
    }
    emit_fast_mode_update_if_changed(session, msg_record.get("fast_mode_state"), out);
}

fn handle_system_compact_boundary(
    session: &BridgeSession,
    msg_record: &Map<String, Value>,
    out: &mut Vec<BridgeEvent>,
) {
    let Some(meta) = msg_record.get("compact_metadata").and_then(Value::as_object) else { return };
    let trigger = meta.get("trigger").and_then(Value::as_str).unwrap_or("");
    let pre_tokens = meta
        .get("pre_tokens")
        .or_else(|| meta.get("preTokens"))
        .and_then(Value::as_u64);
    let Some(pre_tokens) = pre_tokens else { return };
    let trigger_typed = match trigger {
        "manual" => crate::agent::types::CompactionTrigger::Manual,
        "auto" => crate::agent::types::CompactionTrigger::Auto,
        _ => return,
    };
    push_session_update(
        out,
        &session.session_id,
        SessionUpdate::CompactionBoundary { trigger: trigger_typed, pre_tokens },
    );
}

/// Top-level dispatcher. Routes one `forge_sdk::Message` to its
/// per-subtype handler. The bridge session captures continuous state
/// (open tool_calls, last assistant error, fast mode, supported modes)
/// across messages.
#[allow(clippy::too_many_lines)]
pub fn handle_sdk_message(
    session: &mut BridgeSession,
    msg: &SdkMessage,
    out: &mut Vec<BridgeEvent>,
) {
    match msg {
        SdkMessage::Assistant { message, parent_tool_use_id, error, .. } => {
            // Wrap the typed envelope back into a JSON record so the
            // generic content_block walker can handle every block type.
            // The `error` field travels with the outer envelope, not the
            // message body.
            let value = serde_json::to_value(message).unwrap_or(Value::Null);
            let error_str = error.as_ref().map(|e| serde_json::to_value(e).ok()
                .and_then(|v| v.as_str().map(str::to_owned)))
                .and_then(|opt| opt);
            handle_assistant_message(
                session,
                parent_tool_use_id.as_deref(),
                error_str.as_deref(),
                &value,
                out,
            );
        }
        SdkMessage::User { message, parent_tool_use_id, tool_use_result, .. } => {
            let msg_value = serde_json::to_value(message).unwrap_or(Value::Null);
            handle_user_tool_result_blocks(
                session,
                parent_tool_use_id.as_deref(),
                &msg_value,
                out,
            );
            // `tool_use_result` is a CLI-emitted side payload for
            // sub-agent results (parent_tool_use_id present).
            if let Some(result) = tool_use_result
                && let Some(tool_use_id) = parent_tool_use_id.as_deref()
                && !tool_use_id.is_empty()
            {
                let parsed = unwrap_tool_use_result(result);
                emit_tool_result_update(
                    session,
                    tool_use_id,
                    parsed.is_error,
                    Some(&parsed.content),
                    Some(result),
                    out,
                );
            }
        }
        SdkMessage::Result {
            subtype,
            is_error,
            ..
        } => {
            // Walk the raw JSON for terminal_reason / errors /
            // fast_mode_state since those aren't typed fields on
            // forge-sdk's Result variant today.
            let raw = serde_json::to_value(msg).unwrap_or(Value::Null);
            let raw_record = raw.as_object();
            handle_result_message(
                session,
                *is_error,
                subtype,
                raw_record.and_then(|r| r.get("fast_mode_state")),
                raw_record.and_then(|r| r.get("terminal_reason")),
                raw_record.and_then(|r| r.get("errors")),
                out,
            );
        }
        SdkMessage::System { subtype, session_id: msg_session_id, data } => {
            let msg_record = data.as_object();
            let Some(msg_record) = msg_record else { return };
            match subtype.as_str() {
                "api_retry" => {
                    if let Some(update) = build_api_retry_update(msg_record) {
                        push_session_update(out, &session.session_id, update);
                    }
                }
                "session_state_changed" => {
                    if let Some(state) = parse_runtime_session_state(msg_record.get("state")) {
                        push_session_update(
                            out,
                            &session.session_id,
                            SessionUpdate::RuntimeSessionStateUpdate { state },
                        );
                    }
                }
                "init" => {
                    handle_system_init(session, msg_session_id.as_deref(), msg_record, out);
                }
                "status" => {
                    handle_system_status(session, msg_record, out);
                }
                "compact_boundary" => {
                    handle_system_compact_boundary(session, msg_record, out);
                }
                "local_command_output" => {
                    let content = msg_record.get("content").and_then(Value::as_str).unwrap_or("");
                    if !content.trim().is_empty() {
                        push_session_update(
                            out,
                            &session.session_id,
                            SessionUpdate::AgentMessageChunk {
                                content: ContentBlock::Text { text: content.to_owned() },
                            },
                        );
                    }
                }
                "elicitation_complete" => {
                    let elicitation_id =
                        msg_record.get("elicitation_id").and_then(Value::as_str).unwrap_or("");
                    if !elicitation_id.is_empty() {
                        out.push(BridgeEvent::ElicitationComplete {
                            session_id: session.session_id.clone(),
                            elicitation_id: elicitation_id.to_owned(),
                            server_name: msg_record
                                .get("mcp_server_name")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                        });
                    }
                }
                _ => {
                    // task_started/progress/notification fall here in
                    // the upstream port; they need state coordination
                    // (taskToolUseIds map, tool_call lifecycle) wired
                    // in a follow-up. Hook is a no-op for now.
                }
            }
        }
        SdkMessage::RateLimitEvent { rate_limit_info, .. } => {
            let value = serde_json::to_value(rate_limit_info).unwrap_or(Value::Null);
            if let Some(update) = build_rate_limit_update(Some(&value)) {
                push_session_update(out, &session.session_id, update);
            }
        }
        SdkMessage::TaskStarted { tool_use_id, task_id, .. } => {
            let id = tool_use_id.as_deref().unwrap_or("");
            if !id.is_empty() {
                emit_tool_progress_update(session, id, "Task", out);
                if !task_id.is_empty() {
                    session.task_tool_use_ids.insert(task_id.clone(), id.to_owned());
                }
            }
        }
        SdkMessage::TaskProgress { tool_use_id, .. } => {
            let id = tool_use_id.as_deref().unwrap_or("");
            if !id.is_empty() {
                emit_tool_progress_update(session, id, "Task", out);
            }
        }
        SdkMessage::TaskNotification { tool_use_id, summary, .. } => {
            let id = tool_use_id.as_deref().unwrap_or("");
            if !id.is_empty() {
                emit_tool_summary_update(session, id, summary, out);
            }
        }
        // StreamEvent + Error + Unknown are no-ops for now; future
        // stages can layer in.
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::wire::BridgeEvent;
    use forge_sdk::{AssistantEnvelope, Message as SdkMessage};
    use serde_json::json;

    fn fresh_session() -> BridgeSession {
        BridgeSession::new("sess".to_owned(), "/tmp".to_owned())
    }

    fn assistant_msg(blocks: &serde_json::Value) -> SdkMessage {
        let env: AssistantEnvelope = serde_json::from_value(json!({
            "id": "msg_x",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": blocks,
        }))
        .unwrap();
        SdkMessage::Assistant {
            message: env,
            session_id: "sess".to_owned(),
            parent_tool_use_id: None,
            error: None,
            uuid: None,
        }
    }

    #[test]
    fn assistant_text_emits_chunk() {
        let mut s = fresh_session();
        let mut out = Vec::new();
        let msg = assistant_msg(&json!([{"type":"text","text":"hi"}]));
        handle_sdk_message(&mut s, &msg, &mut out);
        // Text blocks fan-through handle_assistant_message → handle_content_block
        // which emits AgentMessageChunk
        assert_eq!(out.len(), 1);
        match &out[0] {
            BridgeEvent::SessionUpdate { update: SessionUpdate::AgentMessageChunk { .. }, .. } => {}
            _ => panic!("expected AgentMessageChunk"),
        }
    }

    #[test]
    fn assistant_tool_use_then_user_tool_result_pairs() {
        let mut s = fresh_session();
        let mut out = Vec::new();
        // First, assistant emits a tool_use.
        let msg = assistant_msg(&json!([{
            "type":"tool_use","id":"tu1","name":"Bash","input":{"command":"ls"}
        }]));
        handle_sdk_message(&mut s, &msg, &mut out);
        assert!(s.tool_calls.contains_key("tu1"));
        let tu_calls_before = out.iter().filter(|e| matches!(e, BridgeEvent::SessionUpdate { update: SessionUpdate::ToolCall { .. }, .. })).count();
        assert_eq!(tu_calls_before, 1);
        out.clear();

        // Then user message with a tool_result block referring to tu1.
        let user_envelope: forge_sdk::UserEnvelope = serde_json::from_value(json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "tu1",
                "is_error": false,
                "content": "stdout text",
            }]
        }))
        .unwrap();
        let user_msg = SdkMessage::User {
            message: user_envelope,
            session_id: "sess".to_owned(),
            parent_tool_use_id: None,
            uuid: None,
            tool_use_result: None,
        };
        handle_sdk_message(&mut s, &user_msg, &mut out);
        let updates: Vec<_> = out
            .iter()
            .filter_map(|e| match e {
                BridgeEvent::SessionUpdate { update: SessionUpdate::ToolCallUpdate { tool_call_update }, .. } => {
                    Some(tool_call_update)
                }
                _ => None,
            })
            .collect();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].tool_call_id, "tu1");
        assert_eq!(updates[0].fields.status.as_deref(), Some("completed"));
    }

    #[test]
    fn result_success_emits_turn_complete_and_finalizes_open_tools() {
        let mut s = fresh_session();
        let mut out = Vec::new();
        // Open a tool.
        let msg = assistant_msg(&json!([{
            "type":"tool_use","id":"tu1","name":"Bash","input":{"command":"sleep 1"}
        }]));
        handle_sdk_message(&mut s, &msg, &mut out);
        out.clear();

        let result: SdkMessage = serde_json::from_value(json!({
            "type": "result",
            "subtype": "success",
            "session_id": "sess",
            "is_error": false,
            "num_turns": 1,
            "duration_ms": 100,
            "duration_api_ms": 80,
        }))
        .unwrap();
        handle_sdk_message(&mut s, &result, &mut out);
        let has_finalize = out.iter().any(|e| matches!(e,
            BridgeEvent::SessionUpdate { update: SessionUpdate::ToolCallUpdate { tool_call_update }, .. }
                if tool_call_update.fields.status.as_deref() == Some("completed") && tool_call_update.tool_call_id == "tu1"));
        assert!(has_finalize, "expected tu1 finalize");
        let has_turn = out.iter().any(|e| matches!(e, BridgeEvent::TurnComplete { .. }));
        assert!(has_turn, "expected TurnComplete");
    }

    #[test]
    fn result_error_emits_turn_error_with_classify() {
        let mut s = fresh_session();
        let mut out = Vec::new();
        let result: SdkMessage = serde_json::from_value(json!({
            "type": "result",
            "subtype": "error_max_turns",
            "session_id": "sess",
            "is_error": true,
            "num_turns": 5,
            "duration_ms": 1,
            "duration_api_ms": 1,
        }))
        .unwrap();
        handle_sdk_message(&mut s, &result, &mut out);
        let BridgeEvent::TurnError { error_kind, .. } = out.last().unwrap() else {
            panic!("expected TurnError");
        };
        assert_eq!(error_kind.as_deref(), Some("plan_limit"));
    }

    #[test]
    fn classify_turn_error_kind_table() {
        assert_eq!(classify_turn_error_kind("error_max_turns", &[], None), "plan_limit");
        assert_eq!(classify_turn_error_kind("error_max_budget_usd", &[], None), "plan_limit");
        assert_eq!(classify_turn_error_kind("billing_error", &[], None), "plan_limit");
        assert_eq!(
            classify_turn_error_kind("internal", &[], Some("authentication_failed")),
            "auth_required"
        );
        assert_eq!(
            classify_turn_error_kind("internal", &[], Some("server_error")),
            "internal"
        );
        assert_eq!(classify_turn_error_kind("anything_else", &[], None), "other");
        assert_eq!(
            classify_turn_error_kind("internal", &["401: authentication required".to_owned()], None),
            "auth_required"
        );
    }
}
