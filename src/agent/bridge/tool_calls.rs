//! Live tool-call lifecycle: 1:1 port of upstream's
//! `agent-sdk/src/bridge/tool_calls.ts` (483 LoC).
//!
//! Public entry points:
//! - `emit_tool_call(session, id, name, input, parent)` — assistant
//!   announces a tool invocation; emits `ToolCall` SessionUpdate or
//!   `ToolCallUpdate` if already known.
//! - `emit_tool_result_update(session, id, is_error, content, result)`
//!   — paired result block; runs the buildToolResultFields pipeline.
//! - `finalize_open_tool_calls(session, status)` — turn-end cleanup.
//! - `emit_tool_progress_update`, `emit_tool_summary_update`,
//!   `set_tool_call_status`, `ensure_tool_call_visible`,
//!   `emit_plan_if_todo_write`, `resolve_task_tool_use_id`,
//!   `task_progress_text`, `task_updated_fields`.

use serde_json::{Map, Value};

use crate::agent::types::{
    PlanEntry, SessionUpdate, TaskMetadata, ToolCall, ToolCallContent, ToolCallUpdate,
    ToolCallUpdateFields,
};
use crate::agent::wire::BridgeEvent;

use super::state::BridgeSession;
use super::tooling::{build_tool_result_fields, create_tool_call};

fn parent_tool_use_id_from_meta(meta: Option<&Value>) -> Option<String> {
    let claude_code = meta?.get("claudeCode")?.as_object()?;
    let id = claude_code.get("parentToolUseId")?.as_str()?;
    if id.is_empty() { None } else { Some(id.to_owned()) }
}

fn merge_task_metadata(
    current: Option<TaskMetadata>,
    update: Option<TaskMetadata>,
) -> Option<TaskMetadata> {
    match (current, update) {
        (None, None) => None,
        (Some(c), None) => Some(c),
        (None, Some(u)) => Some(u),
        (Some(mut c), Some(u)) => {
            if u.end_time.is_some() {
                c.end_time = u.end_time;
            }
            if u.total_paused_ms.is_some() {
                c.total_paused_ms = u.total_paused_ms;
            }
            if u.error.is_some() {
                c.error = u.error;
            }
            if u.is_backgrounded.is_some() {
                c.is_backgrounded = u.is_backgrounded;
            }
            Some(c)
        }
    }
}

fn apply_fields_to_base(base: &mut ToolCall, fields: &ToolCallUpdateFields) {
    if let Some(t) = &fields.title {
        base.title.clone_from(t);
    }
    if let Some(k) = &fields.kind {
        base.kind.clone_from(k);
    }
    if let Some(s) = &fields.status {
        base.status.clone_from(s);
    }
    if let Some(input) = &fields.raw_input {
        base.raw_input = Some(input.clone());
    }
    if let Some(out) = &fields.raw_output {
        base.raw_output = Some(out.clone());
    }
    if let Some(locs) = &fields.locations {
        base.locations.clone_from(locs);
    }
    if let Some(meta) = &fields.output_metadata {
        base.output_metadata = Some(meta.clone());
    }
    if let Some(tm) = fields.task_metadata.clone() {
        base.task_metadata = merge_task_metadata(base.task_metadata.clone(), Some(tm));
    }
    if let Some(meta) = &fields.meta {
        base.meta = Some(meta.clone());
    }
    if let Some(content) = &fields.content {
        base.content.clone_from(content);
    }
}

fn push_session_update(out: &mut Vec<BridgeEvent>, session_id: &str, update: SessionUpdate) {
    out.push(BridgeEvent::SessionUpdate { session_id: session_id.to_owned(), update });
}

fn emit_initial_tool_call(
    session: &mut BridgeSession,
    tool_call: ToolCall,
    out: &mut Vec<BridgeEvent>,
) {
    session.tool_calls.insert(tool_call.tool_call_id.clone(), tool_call.clone());
    push_session_update(
        out,
        &session.session_id,
        SessionUpdate::ToolCall { tool_call },
    );
}

/// Mirrors `emitToolCallUpdate(session, toolUseId, fields, updateKind)`.
pub fn emit_tool_call_update(
    session: &mut BridgeSession,
    tool_use_id: &str,
    fields: ToolCallUpdateFields,
    out: &mut Vec<BridgeEvent>,
) {
    if let Some(base) = session.tool_calls.get_mut(tool_use_id) {
        apply_fields_to_base(base, &fields);
    }
    push_session_update(
        out,
        &session.session_id,
        SessionUpdate::ToolCallUpdate {
            tool_call_update: ToolCallUpdate { tool_call_id: tool_use_id.to_owned(), fields },
        },
    );
}

/// Mirrors `emitToolCall(session, toolUseId, name, input, parentToolUseId)`.
pub fn emit_tool_call(
    session: &mut BridgeSession,
    tool_use_id: &str,
    name: &str,
    input: &Value,
    parent_tool_use_id: Option<&str>,
    out: &mut Vec<BridgeEvent>,
) {
    let existing = session.tool_calls.get(tool_use_id).cloned();
    let resolved_parent = parent_tool_use_id
        .map(str::to_owned)
        .or_else(|| parent_tool_use_id_from_meta(existing.as_ref().and_then(|e| e.meta.as_ref())));
    let mut tool_call = create_tool_call(tool_use_id, name, input, resolved_parent.as_deref());
    "in_progress".clone_into(&mut tool_call.status);

    if existing.is_none() {
        emit_initial_tool_call(session, tool_call, out);
        return;
    }

    let mut fields = ToolCallUpdateFields {
        title: Some(tool_call.title.clone()),
        kind: Some(tool_call.kind.clone()),
        status: Some("in_progress".to_owned()),
        raw_input: tool_call.raw_input.clone(),
        locations: Some(tool_call.locations.clone()),
        meta: tool_call.meta.clone(),
        ..Default::default()
    };
    if !tool_call.content.is_empty() {
        fields.content = Some(tool_call.content.clone());
    }
    emit_tool_call_update(session, tool_use_id, fields, out);
}

/// Mirrors `ensureToolCallVisible`.
pub fn ensure_tool_call_visible(
    session: &mut BridgeSession,
    tool_use_id: &str,
    name: &str,
    input: &Value,
    parent_tool_use_id: Option<&str>,
    out: &mut Vec<BridgeEvent>,
) {
    if let Some(existing) = session.tool_calls.get(tool_use_id).cloned() {
        let existing_parent = parent_tool_use_id_from_meta(existing.meta.as_ref());
        if let Some(p) = parent_tool_use_id
            && existing_parent.as_deref() != Some(p)
        {
            let refreshed = create_tool_call(tool_use_id, name, input, Some(p));
            emit_tool_call_update(
                session,
                tool_use_id,
                ToolCallUpdateFields { meta: refreshed.meta, ..Default::default() },
                out,
            );
        }
        return;
    }
    let tool_call = create_tool_call(tool_use_id, name, input, parent_tool_use_id);
    emit_initial_tool_call(session, tool_call, out);
}

/// Mirrors `emitPlanIfTodoWrite`. Fires when the assistant invokes
/// the TodoWrite tool with a `todos` array.
pub fn emit_plan_if_todo_write(
    session: &BridgeSession,
    name: &str,
    input: &Value,
    out: &mut Vec<BridgeEvent>,
) {
    if name != "TodoWrite" {
        return;
    }
    let Some(todos) = input.as_object().and_then(|r| r.get("todos")).and_then(Value::as_array) else {
        return;
    };
    let entries: Vec<PlanEntry> = todos
        .iter()
        .filter_map(|todo| {
            let r = todo.as_object()?;
            let content = r.get("content").and_then(Value::as_str)?.to_owned();
            if content.is_empty() {
                return None;
            }
            let status = r.get("status").and_then(Value::as_str).unwrap_or("pending").to_owned();
            let active_form = status.clone();
            Some(PlanEntry { content, status, active_form })
        })
        .collect();
    if !entries.is_empty() {
        push_session_update(out, &session.session_id, SessionUpdate::Plan { entries });
    }
}

/// Mirrors `emitToolResultUpdate`.
pub fn emit_tool_result_update(
    session: &mut BridgeSession,
    tool_use_id: &str,
    is_error: bool,
    raw_content: Option<&Value>,
    raw_result: Option<&Value>,
    out: &mut Vec<BridgeEvent>,
) {
    let base = session.tool_calls.get(tool_use_id).cloned();
    let fields = build_tool_result_fields(is_error, raw_content, base.as_ref(), raw_result);
    emit_tool_call_update(session, tool_use_id, fields, out);
}

/// Mirrors `finalizeOpenToolCalls(session, status)`. Walks the
/// in-flight tool_calls map and finalises every still-pending entry
/// to the given terminal status.
pub fn finalize_open_tool_calls(
    session: &mut BridgeSession,
    status: &str,
    out: &mut Vec<BridgeEvent>,
) {
    let pending: Vec<String> = session
        .tool_calls
        .iter()
        .filter(|(_, t)| matches!(t.status.as_str(), "pending" | "in_progress"))
        .map(|(id, _)| id.clone())
        .collect();
    for id in pending {
        emit_tool_call_update(
            session,
            &id,
            ToolCallUpdateFields { status: Some(status.to_owned()), ..Default::default() },
            out,
        );
    }
}

/// Mirrors `emitToolProgressUpdate`.
pub fn emit_tool_progress_update(
    session: &mut BridgeSession,
    tool_use_id: &str,
    name: &str,
    out: &mut Vec<BridgeEvent>,
) {
    let existing = session.tool_calls.get(tool_use_id).cloned();
    let Some(existing) = existing else {
        emit_tool_call(session, tool_use_id, name, &Value::Object(Map::new()), None, out);
        return;
    };
    if matches!(existing.status.as_str(), "in_progress" | "completed" | "failed" | "killed") {
        return;
    }
    emit_tool_call_update(
        session,
        tool_use_id,
        ToolCallUpdateFields { status: Some("in_progress".to_owned()), ..Default::default() },
        out,
    );
}

/// Mirrors `emitToolSummaryUpdate`.
pub fn emit_tool_summary_update(
    session: &mut BridgeSession,
    tool_use_id: &str,
    summary: &str,
    out: &mut Vec<BridgeEvent>,
) {
    let Some(base) = session.tool_calls.get(tool_use_id).cloned() else { return };
    let status = if matches!(base.status.as_str(), "failed" | "killed") {
        base.status
    } else {
        "completed".to_owned()
    };
    let fields = ToolCallUpdateFields {
        status: Some(status),
        raw_output: Some(summary.to_owned()),
        content: Some(vec![ToolCallContent::Content {
            content: crate::agent::types::ContentBlock::Text { text: summary.to_owned() },
        }]),
        ..Default::default()
    };
    emit_tool_call_update(session, tool_use_id, fields, out);
}

/// Mirrors `setToolCallStatus`.
pub fn set_tool_call_status(
    session: &mut BridgeSession,
    tool_use_id: &str,
    status: &str,
    message: Option<&str>,
    out: &mut Vec<BridgeEvent>,
) {
    if !session.tool_calls.contains_key(tool_use_id) {
        return;
    }
    let mut fields = ToolCallUpdateFields { status: Some(status.to_owned()), ..Default::default() };
    if let Some(msg) = message
        && !msg.is_empty()
    {
        fields.raw_output = Some(msg.to_owned());
        fields.content = Some(vec![ToolCallContent::Content {
            content: crate::agent::types::ContentBlock::Text { text: msg.to_owned() },
        }]);
    }
    emit_tool_call_update(session, tool_use_id, fields, out);
}

/// Mirrors `resolveTaskToolUseId(session, msg)`.
#[must_use]
pub fn resolve_task_tool_use_id(session: &BridgeSession, msg: &Map<String, Value>) -> String {
    if let Some(direct) = msg.get("tool_use_id").and_then(Value::as_str)
        && !direct.is_empty()
    {
        return direct.to_owned();
    }
    let Some(task_id) = msg.get("task_id").and_then(Value::as_str) else { return String::new() };
    if task_id.is_empty() {
        return String::new();
    }
    session.task_tool_use_ids.get(task_id).cloned().unwrap_or_default()
}

/// Mirrors `taskProgressText(msg)`.
#[must_use]
pub fn task_progress_text(msg: &Map<String, Value>) -> String {
    if let Some(s) = msg.get("summary").and_then(Value::as_str) {
        let trimmed = s.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    let description = msg.get("description").and_then(Value::as_str).unwrap_or("");
    let last_tool = msg.get("last_tool_name").and_then(Value::as_str).unwrap_or("");
    if !description.is_empty() && !last_tool.is_empty() {
        return format!("{description} (last tool: {last_tool})");
    }
    if description.is_empty() { last_tool.to_owned() } else { description.to_owned() }
}

fn task_patch_status(value: Option<&Value>) -> Option<String> {
    let v = value.and_then(Value::as_str)?;
    Some(match v {
        "pending" => "pending",
        "running" => "in_progress",
        "completed" => "completed",
        "failed" => "failed",
        "killed" => "killed",
        _ => return None,
    }
    .to_owned())
}

fn build_task_metadata(patch: &Map<String, Value>) -> Option<TaskMetadata> {
    let mut tm = TaskMetadata::default();
    let mut any = false;
    if let Some(s) = patch.get("error").and_then(Value::as_str)
        && !s.is_empty()
    {
        tm.error = Some(s.to_owned());
        any = true;
    }
    if let Some(b) = patch.get("is_backgrounded").and_then(Value::as_bool) {
        tm.is_backgrounded = Some(b);
        any = true;
    }
    if let Some(n) = patch.get("end_time").and_then(Value::as_f64)
        && n.is_finite()
        && n >= 0.0
    {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v = n as u64;
        tm.end_time = Some(v);
        any = true;
    }
    if let Some(n) = patch.get("total_paused_ms").and_then(Value::as_f64)
        && n.is_finite()
        && n >= 0.0
    {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v = n as u64;
        tm.total_paused_ms = Some(v);
        any = true;
    }
    if any { Some(tm) } else { None }
}

/// Mirrors `taskUpdatedFields(msg)`.
#[must_use]
pub fn task_updated_fields(msg: &Map<String, Value>) -> ToolCallUpdateFields {
    let patch_owned: Map<String, Value> = msg
        .get("patch")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let patch = &patch_owned;
    let mut fields = ToolCallUpdateFields::default();
    let status = task_patch_status(patch.get("status"));
    let description = patch.get("description").and_then(Value::as_str).unwrap_or("");
    let error = patch.get("error").and_then(Value::as_str).unwrap_or("");

    if let Some(s) = status.as_ref() {
        fields.status = Some(s.clone());
    }
    if !description.is_empty() {
        fields.raw_output = Some(description.to_owned());
        fields.content = Some(vec![ToolCallContent::Content {
            content: crate::agent::types::ContentBlock::Text { text: description.to_owned() },
        }]);
    } else if matches!(status.as_deref(), Some("failed" | "killed")) && !error.is_empty() {
        fields.raw_output = Some(error.to_owned());
        fields.content = Some(vec![ToolCallContent::Content {
            content: crate::agent::types::ContentBlock::Text { text: error.to_owned() },
        }]);
    }
    if let Some(tm) = build_task_metadata(patch) {
        fields.task_metadata = Some(tm);
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fresh() -> BridgeSession {
        BridgeSession::new("s".to_owned(), "/tmp".to_owned())
    }

    #[test]
    fn emit_tool_call_inserts_into_map_and_emits() {
        let mut s = fresh();
        let mut out = Vec::new();
        emit_tool_call(&mut s, "tu1", "Bash", &json!({"command":"ls"}), None, &mut out);
        assert_eq!(out.len(), 1);
        assert!(s.tool_calls.contains_key("tu1"));
        let tc = s.tool_calls.get("tu1").unwrap();
        assert_eq!(tc.title, "ls");
        assert_eq!(tc.status, "in_progress");
    }

    #[test]
    fn emit_tool_call_existing_emits_update_only() {
        let mut s = fresh();
        let mut out = Vec::new();
        emit_tool_call(&mut s, "tu1", "Bash", &json!({"command":"ls"}), None, &mut out);
        out.clear();
        emit_tool_call(&mut s, "tu1", "Bash", &json!({"command":"pwd"}), None, &mut out);
        assert_eq!(out.len(), 1);
        let BridgeEvent::SessionUpdate {
            update: SessionUpdate::ToolCallUpdate { tool_call_update }, ..
        } = &out[0]
        else {
            panic!("expected ToolCallUpdate");
        };
        assert_eq!(tool_call_update.tool_call_id, "tu1");
        assert_eq!(tool_call_update.fields.title.as_deref(), Some("pwd"));
    }

    #[test]
    fn emit_tool_result_update_pairs_with_base() {
        let mut s = fresh();
        let mut out = Vec::new();
        emit_tool_call(&mut s, "tu1", "Bash", &json!({"command":"ls"}), None, &mut out);
        out.clear();
        let raw_result = json!({"stdout":"file1\nfile2\n","stderr":""});
        emit_tool_result_update(&mut s, "tu1", false, Some(&json!("ignored")), Some(&raw_result), &mut out);
        assert_eq!(out.len(), 1);
        let BridgeEvent::SessionUpdate {
            update: SessionUpdate::ToolCallUpdate { tool_call_update }, ..
        } = &out[0]
        else {
            panic!("expected ToolCallUpdate");
        };
        assert_eq!(tool_call_update.fields.status.as_deref(), Some("completed"));
        assert!(tool_call_update.fields.raw_output.as_deref().unwrap().contains("file1"));
        // Base should be patched.
        let base = s.tool_calls.get("tu1").unwrap();
        assert_eq!(base.status, "completed");
    }

    #[test]
    fn finalize_open_tool_calls_only_terminates_pending() {
        let mut s = fresh();
        let mut out = Vec::new();
        emit_tool_call(&mut s, "tu1", "Bash", &json!({"command":"a"}), None, &mut out);
        emit_tool_call(&mut s, "tu2", "Bash", &json!({"command":"b"}), None, &mut out);
        // Mark tu1 completed already.
        s.tool_calls.get_mut("tu1").unwrap().status = "completed".to_owned();
        out.clear();
        finalize_open_tool_calls(&mut s, "failed", &mut out);
        assert_eq!(out.len(), 1);
        let BridgeEvent::SessionUpdate {
            update: SessionUpdate::ToolCallUpdate { tool_call_update }, ..
        } = &out[0]
        else {
            panic!("expected ToolCallUpdate");
        };
        assert_eq!(tool_call_update.tool_call_id, "tu2");
        assert_eq!(tool_call_update.fields.status.as_deref(), Some("failed"));
    }

    #[test]
    fn emit_plan_if_todo_write_emits_plan_entries() {
        let s = fresh();
        let mut out = Vec::new();
        let input = json!({"todos": [
            {"content":"a","status":"pending"},
            {"content":"","status":"pending"},
            {"content":"b","status":"in_progress"},
        ]});
        emit_plan_if_todo_write(&s, "TodoWrite", &input, &mut out);
        assert_eq!(out.len(), 1);
        let BridgeEvent::SessionUpdate { update: SessionUpdate::Plan { entries }, .. } = &out[0]
        else {
            panic!("expected Plan");
        };
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].content, "a");
        assert_eq!(entries[1].content, "b");
    }

    #[test]
    fn task_progress_text_summary_takes_priority() {
        let m: Map<String, Value> = serde_json::from_value(json!({
            "summary": "  good  ",
            "description": "ignored",
            "last_tool_name": "Bash",
        }))
        .unwrap();
        assert_eq!(task_progress_text(&m), "good");
    }

    #[test]
    fn task_updated_fields_extracts_status_and_metadata() {
        let m: Map<String, Value> = serde_json::from_value(json!({
            "patch": {
                "status": "running",
                "description": "doing things",
                "is_backgrounded": false,
                "end_time": 1000.0,
            }
        }))
        .unwrap();
        let f = task_updated_fields(&m);
        assert_eq!(f.status.as_deref(), Some("in_progress"));
        assert_eq!(f.raw_output.as_deref(), Some("doing things"));
        let tm = f.task_metadata.unwrap();
        assert_eq!(tm.is_backgrounded, Some(false));
        assert_eq!(tm.end_time, Some(1000));
    }
}
