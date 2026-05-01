//! Tool-call construction + tool-result extraction. 1:1 port of
//! upstream's `agent-sdk/src/bridge/tooling.ts` (714 LoC).
//!
//! Three groups of helpers:
//! 1. Front-of-tool: `create_tool_call`, `tool_title`, `normalize_tool_kind`,
//!    `is_tool_use_block_type`, `TOOL_RESULT_TYPES`.
//! 2. Result extraction: `build_tool_result_fields` per-tool branches
//!    (Bash / Read / Write / Edit / Agent / `ReadMcpResource`),
//!    `normalize_tool_result_text`, `extract_text`,
//!    `persisted_output_first_line`, `sanitize_sdk_rejection_text`,
//!    `extract_tool_output_metadata`.
//! 3. Helpers: `unwrap_tool_use_result`.

use serde_json::{Map, Value, json};

use crate::agent::types::{
    BashOutputMetadata, ContentBlock, TodoWriteOutputMetadata, ToolCall, ToolCallContent,
    ToolCallUpdateFields, ToolLocation, ToolOutputMetadata,
};

use super::cache_policy::{CACHE_SPLIT_POLICY, preview_kilobyte_label};

/// Block types the CLI uses for tool results. Mirrors upstream's
/// `TOOL_RESULT_TYPES` set in tooling.ts:5.
pub const TOOL_RESULT_TYPES: &[&str] = &[
    "tool_result",
    "tool_search_tool_result",
    "web_fetch_tool_result",
    "web_search_tool_result",
    "code_execution_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "mcp_tool_result",
];

#[must_use]
pub fn is_tool_result_block_type(block_type: &str) -> bool {
    TOOL_RESULT_TYPES.contains(&block_type)
}

#[must_use]
pub fn is_tool_use_block_type(block_type: &str) -> bool {
    matches!(block_type, "tool_use" | "server_tool_use" | "mcp_tool_use")
}

/// Mirrors `normalizeToolKind`. Maps tool name → kind string consumed
/// by the TUI's tool-card renderer.
#[must_use]
pub fn normalize_tool_kind(name: &str) -> &'static str {
    match name {
        "Bash" => "execute",
        "Read" | "ReadMcpResource" => "read",
        "Write" | "Edit" => "edit",
        "Delete" => "delete",
        "Move" => "move",
        "Glob" | "Grep" => "search",
        "WebFetch" => "fetch",
        "TodoWrite" => "other",
        "ExitPlanMode" => "switch_mode",
        // "Task" / "Agent" fall through to the "think" default.
        _ => "think",
    }
}

/// Mirrors `toolTitle(name, input)`. Produces the human-friendly
/// title shown on the tool card header (e.g. Bash → command, Glob →
/// pattern + path, etc.).
#[must_use]
pub fn tool_title(name: &str, input: &Value) -> String {
    let record = input.as_object();
    let s = |k: &str| -> &str {
        record.and_then(|r| r.get(k)).and_then(Value::as_str).unwrap_or("")
    };

    match name {
        "Bash" => {
            let command = s("command");
            if command.is_empty() { "Terminal".to_owned() } else { command.to_owned() }
        }
        "Glob" => {
            let pattern = s("pattern");
            let path = s("path");
            match (pattern.is_empty(), path.is_empty()) {
                (false, false) => format!("Glob {pattern} in {path}"),
                (false, true) => format!("Glob {pattern}"),
                (true, false) => format!("Glob {path}"),
                (true, true) => name.to_owned(),
            }
        }
        "WebFetch" => {
            let url = s("url");
            if url.is_empty() { name.to_owned() } else { format!("WebFetch {url}") }
        }
        "WebSearch" => {
            let query = s("query");
            if query.is_empty() { name.to_owned() } else { format!("WebSearch {query}") }
        }
        "Read" | "Write" | "Edit" => {
            let file_path = s("file_path");
            if file_path.is_empty() { name.to_owned() } else { format!("{name} {file_path}") }
        }
        "ReadMcpResource" => {
            let uri = s("uri");
            let server = s("server");
            match (server.is_empty(), uri.is_empty()) {
                (false, false) => format!("ReadMcpResource {server} {uri}"),
                (true, false) => format!("ReadMcpResource {uri}"),
                _ => name.to_owned(),
            }
        }
        _ => name.to_owned(),
    }
}

/// Mirrors `editDiffContent(name, input)` — initial diff content for
/// Edit / Write tool cards (before the result lands). Empty Vec for
/// other tools.
#[must_use]
fn edit_diff_content(name: &str, input: &Value) -> Vec<ToolCallContent> {
    let Some(record) = input.as_object() else { return Vec::new() };
    let s = |k: &str| record.get(k).and_then(Value::as_str).unwrap_or("");
    let file_path = s("file_path");
    if file_path.is_empty() {
        return Vec::new();
    }

    if name == "Edit" {
        let old_text = s("old_string");
        let new_text = s("new_string");
        if old_text.is_empty() && new_text.is_empty() {
            return Vec::new();
        }
        return vec![ToolCallContent::Diff {
            old_path: file_path.to_owned(),
            new_path: file_path.to_owned(),
            old: old_text.to_owned(),
            new: new_text.to_owned(),
            repository: None,
        }];
    }
    if name == "Write" {
        let new_text = s("content");
        if new_text.is_empty() {
            return Vec::new();
        }
        return vec![ToolCallContent::Diff {
            old_path: file_path.to_owned(),
            new_path: file_path.to_owned(),
            old: String::new(),
            new: new_text.to_owned(),
            repository: None,
        }];
    }
    Vec::new()
}

/// Mirrors `createToolCall(toolUseId, name, input, parentToolUseId)`.
/// Builds the rich `ToolCall` envelope upstream emits when the
/// assistant first invokes a tool.
#[must_use]
pub fn create_tool_call(
    tool_use_id: &str,
    name: &str,
    input: &Value,
    parent_tool_use_id: Option<&str>,
) -> ToolCall {
    let file_path = input
        .as_object()
        .and_then(|r| r.get("file_path"))
        .and_then(Value::as_str);
    let locations = file_path
        .map(|p| vec![ToolLocation { path: p.to_owned(), line: None }])
        .unwrap_or_default();
    let meta = json!({
        "claudeCode": {
            "toolName": name,
            "parentToolUseId": parent_tool_use_id,
        }
    });

    ToolCall {
        tool_call_id: tool_use_id.to_owned(),
        title: tool_title(name, input),
        kind: normalize_tool_kind(name).to_owned(),
        status: "pending".to_owned(),
        content: edit_diff_content(name, input),
        raw_input: Some(input.clone()),
        raw_output: None,
        output_metadata: None,
        task_metadata: None,
        locations,
        meta: Some(meta),
    }
}

// ----- text extraction + persisted-output preview -----

/// Mirrors `extractText(value)` — flattens a tool_result content
/// payload into a single `String`. Accepts string, array of
/// `{ type: "text", text }` blocks, or single `{ text }` object.
#[must_use]
pub fn extract_text(value: &Value) -> String {
    if let Some(s) = value.as_str() {
        return s.to_owned();
    }
    if let Some(arr) = value.as_array() {
        return arr
            .iter()
            .filter_map(|entry| {
                if let Some(s) = entry.as_str() {
                    return Some(s.to_owned());
                }
                entry.as_object().and_then(|r| r.get("text")).and_then(Value::as_str).map(str::to_owned)
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
    }
    if let Some(s) = value.as_object().and_then(|r| r.get("text")).and_then(Value::as_str) {
        return s.to_owned();
    }
    String::new()
}

const PERSISTED_OUTPUT_OPEN_TAG: &str = "<persisted-output>";
const PERSISTED_OUTPUT_CLOSE_TAG: &str = "</persisted-output>";

fn persisted_output_inner_text(text: &str) -> Option<&str> {
    let lower = text.to_ascii_lowercase();
    let open = lower.find(PERSISTED_OUTPUT_OPEN_TAG)?;
    let body_start = open + PERSISTED_OUTPUT_OPEN_TAG.len();
    let close = lower[body_start..].find(PERSISTED_OUTPUT_CLOSE_TAG)? + body_start;
    Some(&text[body_start..close])
}

fn persisted_output_first_line(text: &str) -> Option<String> {
    let inner = persisted_output_inner_text(text)?;
    let expected_preview = format!("preview (first {}):", preview_kilobyte_label(CACHE_SPLIT_POLICY).to_lowercase());
    for line in inner.split(['\n', '\r']) {
        let cleaned: String = line
            .chars()
            .skip_while(|c| matches!(*c, ' ' | '\t' | '|' | '\u{2502}' | '\u{2503}' | '\u{2551}'))
            .collect();
        let cleaned = cleaned.trim();
        if cleaned.is_empty() {
            continue;
        }
        if cleaned.eq_ignore_ascii_case(&expected_preview) {
            continue;
        }
        return Some(cleaned.to_owned());
    }
    None
}

// ----- SDK rejection sanitization -----

const USER_REJECTED_TOOL_USE_EXACT: &str =
    "The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and wait for the user to tell you how to proceed.";
const USER_REJECTED_TOOL_USE_PREFIX: &str =
    "The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). To tell you how to proceed, the user said:";
const PERMISSION_DENIED_TOOL_USE_EXACT: &str =
    "Permission for this tool use was denied. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). Try a different approach or report the limitation to complete your task.";
const PERMISSION_DENIED_TOOL_USE_PREFIX: &str =
    "Permission for this tool use was denied. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). The user said:";

fn sanitize_sdk_rejection_text(text: &str) -> String {
    let normalized = text.trim();
    if normalized == USER_REJECTED_TOOL_USE_EXACT
        || normalized.starts_with(USER_REJECTED_TOOL_USE_PREFIX)
    {
        return "Cancelled by user.".to_owned();
    }
    if normalized == PERMISSION_DENIED_TOOL_USE_EXACT
        || normalized.starts_with(PERMISSION_DENIED_TOOL_USE_PREFIX)
    {
        return "Permission denied.".to_owned();
    }
    text.to_owned()
}

#[must_use]
pub fn normalize_tool_result_text(value: &Value, is_error: bool) -> String {
    let text = extract_text(value);
    if text.is_empty() {
        return String::new();
    }
    let normalized = persisted_output_first_line(&text).unwrap_or(text);
    if is_error { sanitize_sdk_rejection_text(&normalized) } else { normalized }
}

// ----- Result-record candidate walking -----

fn push_record(candidates: &mut Vec<Map<String, Value>>, value: &Value) {
    if let Some(r) = value.as_object() {
        candidates.push(r.clone());
    }
}

fn push_records(candidates: &mut Vec<Map<String, Value>>, value: &Value) {
    if let Some(arr) = value.as_array() {
        for entry in arr {
            push_record(candidates, entry);
        }
        return;
    }
    push_record(candidates, value);
}

fn push_nested_records(candidates: &mut Vec<Map<String, Value>>, value: &Value) {
    if let Some(arr) = value.as_array() {
        for entry in arr {
            push_nested_records(candidates, entry);
        }
        return;
    }
    let Some(record) = value.as_object() else { return };
    if let Some(v) = record.get("result") {
        push_records(candidates, v);
    }
    if let Some(v) = record.get("data") {
        push_records(candidates, v);
    }
    if let Some(v) = record.get("content") {
        push_records(candidates, v);
    }
}

fn result_record_candidates(raw_result: Option<&Value>, raw_content: Option<&Value>) -> Vec<Map<String, Value>> {
    let mut out = Vec::new();
    if let Some(v) = raw_result {
        push_records(&mut out, v);
        push_nested_records(&mut out, v);
    }
    if let Some(v) = raw_content {
        push_records(&mut out, v);
        push_nested_records(&mut out, v);
    }
    out
}

// ----- per-tool extraction -----

fn find_bash_result_record(
    raw_result: Option<&Value>,
    raw_content: Option<&Value>,
) -> Option<Map<String, Value>> {
    result_record_candidates(raw_result, raw_content).into_iter().find(|c| {
        c.contains_key("stdout")
            || c.contains_key("stderr")
            || c.contains_key("backgroundTaskId")
            || c.contains_key("backgroundedByUser")
            || c.contains_key("assistantAutoBackgrounded")
    })
}

fn bash_background_message(record: &Map<String, Value>) -> String {
    let Some(task_id) = record.get("backgroundTaskId").and_then(Value::as_str) else { return String::new() };
    if task_id.is_empty() {
        return String::new();
    }
    if record.get("assistantAutoBackgrounded").and_then(Value::as_bool) == Some(true) {
        return format!("Command was auto-backgrounded by assistant mode with ID: {task_id}.");
    }
    if record.get("backgroundedByUser").and_then(Value::as_bool) == Some(true) {
        return format!("Command was backgrounded by user with ID: {task_id}.");
    }
    format!("Command is running in background with ID: {task_id}.")
}

fn build_bash_display_output(record: &Map<String, Value>) -> String {
    let mut segments = Vec::new();
    if let Some(s) = record.get("stdout").and_then(Value::as_str)
        && !s.is_empty()
    {
        segments.push(s.to_owned());
    }
    if let Some(s) = record.get("stderr").and_then(Value::as_str)
        && !s.is_empty()
    {
        segments.push(s.to_owned());
    }
    if record.get("interrupted").and_then(Value::as_bool) == Some(true) {
        segments.push("Command was aborted before completion.".to_owned());
    }
    let bg = bash_background_message(record);
    if !bg.is_empty() {
        segments.push(bg);
    }
    segments.join("\n")
}

fn file_unchanged_result_text(
    raw_result: Option<&Value>,
    raw_content: Option<&Value>,
) -> String {
    for candidate in result_record_candidates(raw_result, raw_content) {
        if candidate.get("type").and_then(Value::as_str) != Some("file_unchanged") {
            continue;
        }
        if let Some(file_record) = candidate.get("file").and_then(Value::as_object)
            && let Some(file_path) = file_record.get("filePath").and_then(Value::as_str)
        {
            let trimmed = file_path.trim();
            if !trimmed.is_empty() {
                return format!("File unchanged: {trimmed}");
            }
        }
    }
    String::new()
}

fn agent_title_from_agent_output(
    raw_result: Option<&Value>,
    raw_content: Option<&Value>,
) -> String {
    for candidate in result_record_candidates(raw_result, raw_content) {
        if let Some(agent_type) = candidate.get("agentType").and_then(Value::as_str) {
            let trimmed = agent_type.trim();
            if !trimmed.is_empty() {
                return trimmed.to_owned();
            }
        }
    }
    String::new()
}

fn write_diff_from_input(raw_input: Option<&Value>) -> Vec<ToolCallContent> {
    let Some(record) = raw_input.and_then(Value::as_object) else { return Vec::new() };
    let file_path = record.get("file_path").and_then(Value::as_str).unwrap_or("");
    let content = record.get("content").and_then(Value::as_str).unwrap_or("");
    if file_path.is_empty() || content.is_empty() {
        return Vec::new();
    }
    vec![ToolCallContent::Diff {
        old_path: file_path.to_owned(),
        new_path: file_path.to_owned(),
        old: String::new(),
        new: content.to_owned(),
        repository: None,
    }]
}

fn edit_diff_from_input(raw_input: Option<&Value>) -> Vec<ToolCallContent> {
    let Some(record) = raw_input.and_then(Value::as_object) else { return Vec::new() };
    let file_path = record.get("file_path").and_then(Value::as_str).unwrap_or("");
    let old_text = record
        .get("old_string")
        .or_else(|| record.get("oldString"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let new_text = record
        .get("new_string")
        .or_else(|| record.get("newString"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if file_path.is_empty() || (old_text.is_empty() && new_text.is_empty()) {
        return Vec::new();
    }
    vec![ToolCallContent::Diff {
        old_path: file_path.to_owned(),
        new_path: file_path.to_owned(),
        old: old_text.to_owned(),
        new: new_text.to_owned(),
        repository: None,
    }]
}

fn write_diff_from_result(raw_content: Option<&Value>) -> Vec<ToolCallContent> {
    let candidates: Vec<&Value> = match raw_content {
        Some(v) => v.as_array().map_or_else(|| vec![v], |a| a.iter().collect()),
        None => Vec::new(),
    };
    for candidate in candidates {
        let Some(record) = candidate.as_object() else { continue };
        let file_path = record
            .get("filePath")
            .or_else(|| record.get("file_path"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let content = record.get("content").and_then(Value::as_str).unwrap_or("");
        let original_raw = record.get("originalFile").or_else(|| record.get("original_file"));
        let git_diff = record.get("gitDiff").and_then(Value::as_object);
        let repository = git_diff
            .and_then(|g| g.get("repository"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        if file_path.is_empty() || content.is_empty() || original_raw.is_none() {
            continue;
        }
        let original = original_raw
            .and_then(Value::as_str)
            .map_or_else(String::new, str::to_owned);
        return vec![ToolCallContent::Diff {
            old_path: file_path.to_owned(),
            new_path: file_path.to_owned(),
            old: original,
            new: content.to_owned(),
            repository,
        }];
    }
    Vec::new()
}

fn edit_diff_from_result(
    raw_result: Option<&Value>,
    raw_input: Option<&Value>,
) -> Vec<ToolCallContent> {
    let Some(input_record) = raw_input.and_then(Value::as_object) else {
        return edit_diff_from_input(raw_input);
    };
    let file_path = input_record.get("file_path").and_then(Value::as_str).unwrap_or("");
    let old_text = input_record
        .get("old_string")
        .or_else(|| input_record.get("oldString"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let new_text = input_record
        .get("new_string")
        .or_else(|| input_record.get("newString"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if file_path.is_empty() || (old_text.is_empty() && new_text.is_empty()) {
        return Vec::new();
    }
    for candidate in result_record_candidates(raw_result, None) {
        let candidate_path = candidate
            .get("filePath")
            .or_else(|| candidate.get("file_path"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let git_diff = candidate.get("gitDiff").and_then(Value::as_object);
        if candidate_path.is_empty() && git_diff.is_none() {
            continue;
        }
        if !candidate_path.is_empty() && candidate_path != file_path {
            continue;
        }
        let repository = git_diff
            .and_then(|g| g.get("repository"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        return vec![ToolCallContent::Diff {
            old_path: file_path.to_owned(),
            new_path: file_path.to_owned(),
            old: old_text.to_owned(),
            new: new_text.to_owned(),
            repository,
        }];
    }
    edit_diff_from_input(raw_input)
}

fn parse_json_candidate(value: Option<&Value>) -> Option<Value> {
    let text = value.map_or_else(String::new, |v| {
        if let Some(s) = v.as_str() { s.to_owned() } else { extract_text(v) }
    });
    let trimmed = text.trim();
    if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}

fn push_structured_record_candidates(candidates: &mut Vec<Map<String, Value>>, value: &Value) {
    let Some(record) = value.as_object() else { return };
    candidates.push(record.clone());
    for key in ["result", "data", "content"] {
        if let Some(nested) = record.get(key).and_then(Value::as_object) {
            candidates.push(nested.clone());
        }
    }
}

fn mcp_resource_content_from_result(
    raw_result: Option<&Value>,
    raw_content: Option<&Value>,
) -> Vec<ToolCallContent> {
    let mut candidates: Vec<Map<String, Value>> = Vec::new();
    if let Some(v) = raw_result {
        push_structured_record_candidates(&mut candidates, v);
    }
    if let Some(v) = raw_content {
        push_structured_record_candidates(&mut candidates, v);
    }
    if let Some(v) = parse_json_candidate(raw_result) {
        push_structured_record_candidates(&mut candidates, &v);
    }
    if let Some(v) = parse_json_candidate(raw_content) {
        push_structured_record_candidates(&mut candidates, &v);
    }

    for candidate in candidates {
        let Some(contents) = candidate.get("contents").and_then(Value::as_array) else { continue };
        if contents.is_empty() {
            continue;
        }
        let mut mapped: Vec<ToolCallContent> = Vec::new();
        for entry in contents {
            let Some(record) = entry.as_object() else { continue };
            let Some(uri) = record.get("uri").and_then(Value::as_str) else { continue };
            if uri.is_empty() {
                continue;
            }
            let text = record
                .get("text")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            let mime_type = record
                .get("mimeType")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            let blob_saved_to = record
                .get("blobSavedTo")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            if text.is_none() && blob_saved_to.is_none() {
                continue;
            }
            mapped.push(ToolCallContent::McpResource {
                uri: uri.to_owned(),
                mime_type,
                text,
                blob_saved_to,
            });
        }
        if !mapped.is_empty() {
            return mapped;
        }
    }
    Vec::new()
}

// ----- output_metadata extractor -----

fn extract_tool_output_metadata(
    tool_name: &str,
    raw_result: Option<&Value>,
    raw_content: Option<&Value>,
) -> Option<ToolOutputMetadata> {
    let candidates = result_record_candidates(raw_result, raw_content);

    if tool_name == "Bash" {
        for candidate in &candidates {
            if let Some(b) = candidate.get("assistantAutoBackgrounded").and_then(Value::as_bool) {
                return Some(ToolOutputMetadata {
                    bash: Some(BashOutputMetadata { assistant_auto_backgrounded: Some(b) }),
                    todo_write: None,
                });
            }
        }
        return None;
    }

    if tool_name == "TodoWrite" {
        for candidate in &candidates {
            if let Some(b) = candidate.get("verificationNudgeNeeded").and_then(Value::as_bool) {
                return Some(ToolOutputMetadata {
                    bash: None,
                    todo_write: Some(TodoWriteOutputMetadata { verification_nudge_needed: Some(b) }),
                });
            }
        }
        return None;
    }

    None
}

fn resolve_tool_name(base: Option<&ToolCall>) -> String {
    let Some(meta) = base.and_then(|b| b.meta.as_ref()) else { return String::new() };
    let Some(claude_code) = meta.get("claudeCode").and_then(Value::as_object) else { return String::new() };
    claude_code.get("toolName").and_then(Value::as_str).unwrap_or("").to_owned()
}

// ----- main entry: build_tool_result_fields -----

/// Mirrors `buildToolResultFields(isError, rawContent, base?, rawResult?)`.
/// The high-level entry that turns an arbitrary tool_result block into
/// the structured `ToolCallUpdateFields` upstream's TUI consumes.
#[must_use]
pub fn build_tool_result_fields(
    is_error: bool,
    raw_content: Option<&Value>,
    base: Option<&ToolCall>,
    raw_result: Option<&Value>,
) -> ToolCallUpdateFields {
    let tool_name = resolve_tool_name(base);
    let mut fields = ToolCallUpdateFields {
        status: Some(if is_error { "failed".to_owned() } else { "completed".to_owned() }),
        ..Default::default()
    };

    // Read: file_unchanged shortcut. Early return.
    let file_unchanged_text =
        if !is_error && tool_name == "Read" { file_unchanged_result_text(raw_result, raw_content) } else { String::new() };
    if !file_unchanged_text.is_empty() {
        fields.raw_output = Some(file_unchanged_text.clone());
        fields.content = Some(vec![ToolCallContent::Content {
            content: ContentBlock::Text { text: file_unchanged_text },
        }]);
        return fields;
    }

    // Agent: title from agentType.
    if !is_error && tool_name == "Agent" {
        let agent_title = agent_title_from_agent_output(raw_result, raw_content);
        if !agent_title.is_empty() {
            fields.title = Some(agent_title);
        }
    }

    // Bash: extract structured output, otherwise normalise the text.
    let bash_record = if tool_name == "Bash" {
        find_bash_result_record(raw_result, raw_content)
    } else {
        None
    };
    let normalized_raw_output = raw_content
        .map(|v| normalize_tool_result_text(v, is_error))
        .unwrap_or_default();
    let raw_output = if let Some(record) = bash_record.as_ref() {
        build_bash_display_output(record)
    } else if !normalized_raw_output.is_empty() {
        normalized_raw_output
    } else {
        raw_content
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .unwrap_or_default()
    };
    if !raw_output.is_empty() {
        fields.raw_output = Some(raw_output.clone());
    }

    // output_metadata: per-tool.
    if let Some(meta) = extract_tool_output_metadata(&tool_name, raw_result, raw_content) {
        fields.output_metadata = Some(meta);
    }

    // Write: structured diff.
    if !is_error && tool_name == "Write" {
        let structured = write_diff_from_result(raw_content);
        if !structured.is_empty() {
            fields.content = Some(structured);
            return fields;
        }
        let from_input = write_diff_from_input(base.and_then(|b| b.raw_input.as_ref()));
        if !from_input.is_empty() {
            fields.content = Some(from_input);
            return fields;
        }
    }

    // Edit: structured diff (fall through to base.content if already set).
    if !is_error && tool_name == "Edit" {
        let structured = edit_diff_from_result(raw_result, base.and_then(|b| b.raw_input.as_ref()));
        if !structured.is_empty() {
            fields.content = Some(structured);
            return fields;
        }
        if base.is_some_and(|b| {
            b.content
                .iter()
                .any(|c| matches!(c, ToolCallContent::Diff { .. }))
        }) {
            return fields;
        }
    }

    // ReadMcpResource: structured resource content.
    if !is_error && tool_name == "ReadMcpResource" {
        let structured = mcp_resource_content_from_result(raw_result, raw_content);
        if !structured.is_empty() {
            fields.content = Some(structured);
            return fields;
        }
    }

    // Generic fallback: wrap raw_output as content.
    if !raw_output.is_empty() {
        fields.content = Some(vec![ToolCallContent::Content {
            content: ContentBlock::Text { text: raw_output },
        }]);
    }
    fields
}

/// Mirrors `unwrapToolUseResult(rawResult)`. Walks the tool_result
/// envelope: if the value is an object, surfaces the inner `content`
/// / `result` / `text` field and the `is_error` / `error` flag.
pub struct UnwrappedToolResult {
    pub is_error: bool,
    pub content: Value,
}

#[must_use]
pub fn unwrap_tool_use_result(raw_result: &Value) -> UnwrappedToolResult {
    let Some(record) = raw_result.as_object() else {
        return UnwrappedToolResult { is_error: false, content: raw_result.clone() };
    };
    let is_error = record.get("is_error").and_then(Value::as_bool).unwrap_or(false)
        || record.get("error").and_then(Value::as_bool).unwrap_or(false);
    if let Some(c) = record.get("content") {
        return UnwrappedToolResult { is_error, content: c.clone() };
    }
    if let Some(c) = record.get("result") {
        return UnwrappedToolResult { is_error, content: c.clone() };
    }
    if let Some(c) = record.get("text") {
        return UnwrappedToolResult { is_error, content: c.clone() };
    }
    UnwrappedToolResult { is_error, content: raw_result.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_base(name: &str, input: &serde_json::Value) -> ToolCall {
        create_tool_call("tu_1", name, input, None)
    }

    #[test]
    fn create_tool_call_titles_per_tool() {
        let bash = create_tool_call("tu1", "Bash", &json!({"command": "ls"}), None);
        assert_eq!(bash.title, "ls");
        assert_eq!(bash.kind, "execute");

        let read = create_tool_call("tu2", "Read", &json!({"file_path": "/x"}), None);
        assert_eq!(read.title, "Read /x");
        assert_eq!(read.kind, "read");

        let glob = create_tool_call("tu3", "Glob", &json!({"pattern": "*.rs", "path": "src"}), None);
        assert_eq!(glob.title, "Glob *.rs in src");
        assert_eq!(glob.kind, "search");

        let task = create_tool_call("tu4", "Task", &json!({}), None);
        assert_eq!(task.kind, "think");
    }

    #[test]
    fn create_tool_call_locations_from_file_path() {
        let r = create_tool_call("tu", "Edit", &json!({"file_path": "/foo"}), None);
        assert_eq!(r.locations.len(), 1);
        assert_eq!(r.locations[0].path, "/foo");
    }

    #[test]
    fn edit_initial_content_is_diff() {
        let r = create_tool_call("tu", "Edit", &json!({"file_path": "/f", "old_string": "a", "new_string": "b"}), None);
        assert_eq!(r.content.len(), 1);
        let ToolCallContent::Diff { old, new, .. } = &r.content[0] else { panic!("expected diff") };
        assert_eq!(old, "a");
        assert_eq!(new, "b");
    }

    #[test]
    fn write_initial_content_is_full_diff() {
        let r = create_tool_call("tu", "Write", &json!({"file_path": "/f", "content": "hello"}), None);
        assert_eq!(r.content.len(), 1);
        let ToolCallContent::Diff { old, new, .. } = &r.content[0] else { panic!() };
        assert_eq!(old, "");
        assert_eq!(new, "hello");
    }

    #[test]
    fn extract_text_handles_string_array_object() {
        assert_eq!(extract_text(&json!("hi")), "hi");
        assert_eq!(extract_text(&json!([{"type":"text","text":"a"},{"type":"text","text":"b"}])), "a\nb");
        assert_eq!(extract_text(&json!({"text":"o"})), "o");
        assert_eq!(extract_text(&json!(123)), "");
    }

    #[test]
    fn normalize_text_skips_persisted_preview_line() {
        let v = json!("<persisted-output>\npreview (first 2kb):\n│ hello world\n│ second line\n</persisted-output>");
        let out = normalize_tool_result_text(&v, false);
        assert_eq!(out, "hello world");
    }

    #[test]
    fn sanitize_user_rejected_text_collapses() {
        let v = json!(USER_REJECTED_TOOL_USE_EXACT);
        assert_eq!(normalize_tool_result_text(&v, true), "Cancelled by user.");

        let custom = format!("{USER_REJECTED_TOOL_USE_PREFIX} extra reason");
        let v2 = json!(custom);
        assert_eq!(normalize_tool_result_text(&v2, true), "Cancelled by user.");

        let pd = json!(PERMISSION_DENIED_TOOL_USE_EXACT);
        assert_eq!(normalize_tool_result_text(&pd, true), "Permission denied.");
    }

    #[test]
    fn build_fields_bash_extracts_stdout_stderr() {
        let base = make_base("Bash", &json!({"command":"echo"}));
        let raw_content = json!([{"type":"text","text":"<persisted-output>\n│ hello\n</persisted-output>"}]);
        let raw_result = json!({"stdout":"hello\n","stderr":"","interrupted":false});
        let f = build_tool_result_fields(false, Some(&raw_content), Some(&base), Some(&raw_result));
        assert_eq!(f.status.as_deref(), Some("completed"));
        assert_eq!(f.raw_output.as_deref(), Some("hello\n"));
        // generic content fallback wraps the bash output as text.
        let Some(ToolCallContent::Content { content: ContentBlock::Text { text } }) =
            f.content.as_ref().and_then(|c| c.first())
        else {
            panic!("expected text content");
        };
        assert!(text.contains("hello"));
    }

    #[test]
    fn build_fields_read_file_unchanged_shortcut() {
        let base = make_base("Read", &json!({"file_path": "/x"}));
        let raw_result = json!([{"type":"file_unchanged","file":{"filePath":"/x"}}]);
        let f = build_tool_result_fields(false, Some(&json!("ignored")), Some(&base), Some(&raw_result));
        assert_eq!(f.raw_output.as_deref(), Some("File unchanged: /x"));
    }

    #[test]
    fn build_fields_write_emits_structured_diff() {
        let base = make_base("Write", &json!({"file_path":"/x","content":"new"}));
        let raw_content = json!([{"filePath":"/x","content":"new","originalFile":"old"}]);
        let f = build_tool_result_fields(false, Some(&raw_content), Some(&base), None);
        assert!(matches!(f.content.as_ref().and_then(|c| c.first()), Some(ToolCallContent::Diff { .. })));
    }

    #[test]
    fn build_fields_agent_title_from_agent_type() {
        let base = make_base("Agent", &json!({}));
        let raw_result = json!([{"agentType":"researcher"}]);
        let f = build_tool_result_fields(false, Some(&json!("ok")), Some(&base), Some(&raw_result));
        assert_eq!(f.title.as_deref(), Some("researcher"));
    }

    #[test]
    fn build_fields_mcp_resource_structured() {
        let base = make_base("ReadMcpResource", &json!({"uri":"file:///x"}));
        let raw_content = json!({"contents":[{"uri":"file:///x","text":"hi","mimeType":"text/plain"}]});
        let f = build_tool_result_fields(false, Some(&raw_content), Some(&base), None);
        assert!(matches!(
            f.content.as_ref().and_then(|c| c.first()),
            Some(ToolCallContent::McpResource { .. })
        ));
    }

    #[test]
    fn build_fields_bash_output_metadata() {
        let base = make_base("Bash", &json!({"command":"sleep 30 &"}));
        let raw_result = json!({"backgroundTaskId":"bg1","assistantAutoBackgrounded":true});
        let f = build_tool_result_fields(false, Some(&json!("started")), Some(&base), Some(&raw_result));
        let meta = f.output_metadata.expect("meta");
        assert_eq!(meta.bash.unwrap().assistant_auto_backgrounded, Some(true));
    }

    #[test]
    fn unwrap_tool_use_result_envelopes() {
        let v = json!({"is_error": true, "content": "boom"});
        let u = unwrap_tool_use_result(&v);
        assert!(u.is_error);
        assert_eq!(u.content, json!("boom"));

        let v2 = json!({"result": "ok"});
        let u2 = unwrap_tool_use_result(&v2);
        assert!(!u2.is_error);
        assert_eq!(u2.content, json!("ok"));

        let v3 = json!("plain");
        let u3 = unwrap_tool_use_result(&v3);
        assert_eq!(u3.content, json!("plain"));
    }
}
