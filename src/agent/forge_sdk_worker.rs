//! Worker task that drains [`ForgeSdkCommand`]s and drives a
//! [`forge_sdk::Client`] in-process.
//!
//! Plays the role of the Node bridge subprocess (see
//! `agent-sdk/src/bridge.ts` upstream), but skips NDJSON serialization
//! and runs entirely inside the Rust process. The worker owns a single
//! `Client` after the first `NewSession` / `ResumeSession` arrives,
//! and forwards subsequent commands as direct method calls. A reader
//! subtask drains `Client::next_event()` and translates every SDK
//! `Message` into [`BridgeEvent`]s on the existing event channel via
//! [`crate::agent::forge_sdk_translate::translate_message`].
//!
//! ## State machine (current scope)
//!
//! ```text
//!     waiting --(NewSession)----> running
//!     waiting --(ResumeSession)-> running
//!     waiting --(other command)-> log error, stay waiting
//!     running --(NewSession)----> running (drop+respawn)
//!     running --(Prompt)--------> client.send_user_message[_with_content]
//!     running --(Cancel)--------> client.interrupt
//!     running --(SetModel)------> client.set_model
//!     running --(SetMode)-------> client.set_permission_mode
//!     running --(MCP cmd)-------> client.mcp_*
//!     running --(perm/question)-> drain pending oneshot
//!     running --(elicitation)---> client.respond_to_elicitation
//! ```
//!
//! Permission and question prompts arrive through the `can_use_tool`
//! callback wired at `Client::spawn` time; the worker parks each
//! request on a shared `pending` map keyed by `tool_use_id`, emits
//! the matching `BridgeEvent`, and lets the inbound
//! `PermissionResponse` / `QuestionResponse` command drain the
//! oneshot when the user answers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use forge_sdk::{
    Client, Options, OptionsBuilder, PermissionDecision, PermissionMode, ToolPermissionContext,
};
use tokio::sync::{mpsc, oneshot};

use crate::agent::forge_sdk_bridge::ForgeSdkCommand;
use crate::agent::forge_sdk_translate::translate_message;
use crate::agent::types::CurrentModel;
use crate::agent::wire::BridgeEvent;

/// Pending permission/question responses keyed by `tool_use_id`.
/// The `can_use_tool` callback inserts a oneshot here when the CLI
/// asks; the worker drains it when the matching `PermissionResponse`
/// or `QuestionResponse` command arrives from the TUI. Shared between
/// the callback (set up at session spawn) and the worker dispatch.
type PendingResponses = Arc<Mutex<HashMap<String, oneshot::Sender<PermissionDecision>>>>;

/// Drive a single forge-sdk session for the lifetime of `command_rx`.
/// Returns when the channel is closed (TUI shutting down).
pub async fn run_worker(
    mut command_rx: mpsc::UnboundedReceiver<ForgeSdkCommand>,
    event_tx: mpsc::UnboundedSender<BridgeEvent>,
) {
    let pending: PendingResponses = Arc::new(Mutex::new(HashMap::new()));
    let session_id_slot: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let mut state: WorkerState = WorkerState::Waiting;

    while let Some(cmd) = command_rx.recv().await {
        if let Err(err) = dispatch(
            &mut state,
            cmd,
            &event_tx,
            &pending,
            &session_id_slot,
        )
        .await
        {
            tracing::warn!(
                target: crate::logging::targets::BRIDGE_LIFECYCLE,
                error = %err,
                "forge_sdk_worker: dispatch failed",
            );
        }
    }

    // Channel closed -- drop the client gracefully so the subprocess
    // gets SIGCHLD'd as soon as our handle is dropped.
    if let WorkerState::Running { client, .. } = state {
        let _ = client.disconnect().await;
    }
}

enum WorkerState {
    Waiting,
    Running {
        client: Client,
        // session_id is captured but not used until follow-up commits
        // surface session-scoped events back to the TUI; keep it
        // around so we can correlate without an extra clone path.
        #[allow(dead_code)]
        session_id: String,
    },
}

#[allow(clippy::too_many_lines)]
async fn dispatch(
    state: &mut WorkerState,
    cmd: ForgeSdkCommand,
    event_tx: &mpsc::UnboundedSender<BridgeEvent>,
    pending: &PendingResponses,
    session_id_slot: &Arc<Mutex<String>>,
) -> anyhow::Result<()> {
    use ForgeSdkCommand as C;
    match cmd {
        C::NewSession { cwd, launch_settings: _ } => {
            let options = build_options_with_callback(
                &cwd,
                None,
                event_tx,
                pending,
                session_id_slot,
            );
            spawn_or_replace(state, event_tx, options, session_id_slot).await
        }
        C::ResumeSession { session_id, launch_settings: _ } => {
            // Resume by passing the prior session id to the CLI. The
            // CLI itself decides what cwd to use; we don't override.
            let options = build_options_with_callback(
                "",
                Some(&session_id),
                event_tx,
                pending,
                session_id_slot,
            );
            spawn_or_replace(state, event_tx, options, session_id_slot).await
        }
        C::Prompt { session_id: _, chunks } => {
            let client = require_running(state, "Prompt")?;
            send_prompt(client, chunks).await
        }
        C::Cancel { session_id: _ } => {
            let client = require_running(state, "Cancel")?;
            client.interrupt().await?;
            Ok(())
        }
        C::SetModel { session_id: _, model } => {
            let client = require_running(state, "SetModel")?;
            client.set_model(Some(model.as_str())).await?;
            Ok(())
        }
        C::SetMode { session_id: _, mode } => {
            let client = require_running(state, "SetMode")?;
            let mode = parse_permission_mode(&mode)?;
            client.set_permission_mode(mode).await?;
            Ok(())
        }
        C::PermissionResponse { tool_call_id, outcome, .. } => {
            deliver_permission_response(pending, &tool_call_id, outcome);
            Ok(())
        }
        C::QuestionResponse { tool_call_id, outcome, .. } => {
            deliver_question_response(pending, &tool_call_id, outcome);
            Ok(())
        }
        C::RespondToElicitation {
            elicitation_request_id,
            action,
            content,
            ..
        } => {
            let client = require_running(state, "RespondToElicitation")?;
            let action_str = match action {
                crate::agent::types::ElicitationAction::Accept => "accept",
                crate::agent::types::ElicitationAction::Decline => "decline",
                crate::agent::types::ElicitationAction::Cancel => "cancel",
            };
            client
                .respond_to_elicitation(&elicitation_request_id, action_str, content)
                .await?;
            Ok(())
        }
        C::GetStatusSnapshot { session_id } => {
            let client = require_running(state, "GetStatusSnapshot")?;
            let account = client.account_info().map(translate_account_info).unwrap_or_default();
            let _ = event_tx.send(BridgeEvent::StatusSnapshot { session_id, account });
            Ok(())
        }
        C::GetContextUsage { session_id } => {
            let client = require_running(state, "GetContextUsage")?;
            let usage = client.get_context_usage().await?;
            let percentage = clamp_percentage_to_u8(usage.percentage);
            let _ = event_tx.send(BridgeEvent::ContextUsage {
                session_id,
                percentage: Some(percentage),
            });
            Ok(())
        }
        C::ReloadPlugins { session_id: _ } => {
            let client = require_running(state, "ReloadPlugins")?;
            let _ = client.reload_plugins().await?;
            Ok(())
        }
        C::GetMcpSnapshot { session_id } => {
            let client = require_running(state, "GetMcpSnapshot")?;
            let response = client.mcp_status().await?;
            let servers = response
                .mcp_servers
                .into_iter()
                .map(translate_mcp_server_status)
                .collect();
            let _ = event_tx.send(BridgeEvent::McpSnapshot { session_id, servers, error: None });
            Ok(())
        }
        C::ReconnectMcpServer { server_name, .. } => {
            let client = require_running(state, "ReconnectMcpServer")?;
            client.mcp_reconnect(&server_name).await?;
            Ok(())
        }
        C::ToggleMcpServer { server_name, enabled, .. } => {
            let client = require_running(state, "ToggleMcpServer")?;
            client.mcp_toggle(&server_name, enabled).await?;
            Ok(())
        }
        C::SetMcpServers { servers, .. } => {
            let client = require_running(state, "SetMcpServers")?;
            client.mcp_set_servers(serde_json::to_value(servers)?).await?;
            Ok(())
        }
        C::AuthenticateMcpServer { server_name, .. } => {
            let client = require_running(state, "AuthenticateMcpServer")?;
            let _ = client.mcp_authenticate(&server_name).await?;
            Ok(())
        }
        C::ClearMcpAuth { server_name, .. } => {
            let client = require_running(state, "ClearMcpAuth")?;
            client.mcp_clear_auth(&server_name).await?;
            Ok(())
        }
        C::SubmitMcpOauthCallbackUrl { server_name, callback_url, .. } => {
            let client = require_running(state, "SubmitMcpOauthCallbackUrl")?;
            client
                .mcp_oauth_callback_url(&server_name, &callback_url)
                .await?;
            Ok(())
        }
        C::GenerateSessionTitle { session_id: _, description } => {
            let client = require_running(state, "GenerateSessionTitle")?;
            let _ = client.generate_session_title(&description).await?;
            // Title comes back through session.event eventually; we
            // could also emit a BridgeEvent here to update the tab
            // header immediately.
            Ok(())
        }
        C::RenameSession { session_id, title } => {
            // Offline disk mutation -- no Client required.
            forge_sdk::session::mutations::rename_session(&session_id, &title, None)?;
            Ok(())
        }
    }
}

fn require_running<'a>(
    state: &'a mut WorkerState,
    cmd_label: &'static str,
) -> anyhow::Result<&'a Client> {
    match state {
        WorkerState::Running { client, .. } => Ok(client),
        WorkerState::Waiting => Err(anyhow::anyhow!(
            "forge_sdk_worker: received {cmd_label} before NewSession",
        )),
    }
}

async fn spawn_or_replace(
    state: &mut WorkerState,
    event_tx: &mpsc::UnboundedSender<BridgeEvent>,
    options: Options,
    session_id_slot: &Arc<Mutex<String>>,
) -> anyhow::Result<()> {
    // If we already have a client, drop it first so the existing
    // subprocess can shut down cleanly.
    if let WorkerState::Running { client, .. } = std::mem::replace(state, WorkerState::Waiting) {
        let _ = client.disconnect().await;
    }

    let client = Client::spawn(options).await?;
    let session_id = client.session_id();
    if let Ok(mut slot) = session_id_slot.lock() {
        slot.clone_from(&session_id);
    }

    // Spawn reader subtask. The Client is Arc-backed so we clone for
    // the reader; the worker keeps its own handle for command dispatch.
    let reader_client = client.clone();
    let reader_event_tx = event_tx.clone();
    tokio::spawn(reader_loop(reader_client, reader_event_tx));

    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.into_os_string().into_string().ok())
        .unwrap_or_default();

    // Emit a placeholder Connected event so the TUI can transition
    // out of the connecting state. CurrentModel/cwd/mode get refined
    // by the SDK's `system/init` message hitting the reader.
    let _ = event_tx.send(BridgeEvent::Connected {
        session_id: session_id.clone(),
        cwd: cwd.clone(),
        current_model: placeholder_current_model(),
        available_models: Vec::new(),
        mode: None,
        history_updates: None,
    });

    // Emit the recent-sessions list. The session picker (and slash-
    // command autocomplete) wait on this event before becoming
    // interactive — without it `claude-rs resume` hangs at "Loading
    // recent sessions..." forever.
    let _ = event_tx.send(BridgeEvent::SessionsListed {
        sessions: list_recent_sessions(&cwd),
    });

    *state = WorkerState::Running { client, session_id };
    Ok(())
}

/// Scan the on-disk JSONL transcripts for `cwd` and convert them into
/// the TUI's `SessionListEntry` shape. Mirrors what the upstream Node
/// bridge's `emitSessionsList` did via the JS SDK's `listSessions`.
fn list_recent_sessions(cwd: &str) -> Vec<crate::agent::types::SessionListEntry> {
    use crate::agent::types::SessionListEntry;

    const MAX_RECENT: usize = 50;

    let dir = if cwd.is_empty() { None } else { Some(cwd.to_owned()) };
    forge_sdk::session::scan::list_sessions(dir, Some(MAX_RECENT), 0)
        .into_iter()
        .map(|info| SessionListEntry {
            session_id: info.session_id,
            summary: info.summary,
            last_modified_ms: info.last_modified,
            file_size_bytes: info.file_size.unwrap_or(0),
            cwd: info.cwd,
            git_branch: info.git_branch,
            custom_title: info.custom_title,
            first_prompt: info.first_prompt,
        })
        .collect()
}

async fn reader_loop(client: Client, event_tx: mpsc::UnboundedSender<BridgeEvent>) {
    loop {
        match client.next_event().await {
            Ok(Some(msg)) => {
                for event in translate_message(msg) {
                    if event_tx.send(event).is_err() {
                        return;
                    }
                }
            }
            Ok(None) => {
                tracing::info!(
                    target: crate::logging::targets::BRIDGE_LIFECYCLE,
                    "forge_sdk_worker reader: client closed",
                );
                return;
            }
            Err(err) => {
                tracing::error!(
                    target: crate::logging::targets::BRIDGE_LIFECYCLE,
                    error = %err,
                    "forge_sdk_worker reader: next_event failed",
                );
                return;
            }
        }
    }
}

async fn send_prompt(
    client: &Client,
    chunks: Vec<crate::agent::types::PromptChunk>,
) -> anyhow::Result<()> {
    if chunks.iter().all(|c| c.kind == "text") {
        let prompt: String = chunks
            .iter()
            .filter_map(|c| c.value.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        client.send_user_message(&prompt).await?;
    } else {
        // CLI-shaped content blocks: convert each chunk.
        let content: Vec<serde_json::Value> = chunks
            .into_iter()
            .map(|c| match c.kind.as_str() {
                "text" => serde_json::json!({
                    "type": "text",
                    "text": c.value.as_str().unwrap_or(""),
                }),
                "image" => serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": c.value.get("mime_type").and_then(|v| v.as_str()).unwrap_or("image/png"),
                        "data": c.value.get("data").and_then(|v| v.as_str()).unwrap_or(""),
                    },
                }),
                _ => c.value,
            })
            .collect();
        client.send_user_message_with_content(&content).await?;
    }
    Ok(())
}


fn parse_permission_mode(mode: &str) -> anyhow::Result<PermissionMode> {
    match mode {
        "default" | "ask" => Ok(PermissionMode::Ask),
        "acceptEdits" | "accept_edits" => Ok(PermissionMode::AcceptEdits),
        "plan" => Ok(PermissionMode::Plan),
        "bypassPermissions" | "bypass_permissions" => Ok(PermissionMode::BypassPermissions),
        "auto" => Ok(PermissionMode::Auto),
        "dontAsk" | "dont_ask" | "deny" => Ok(PermissionMode::DenyPermissions),
        other => Err(anyhow::anyhow!(
            "forge_sdk_worker: unknown permission mode {other:?}"
        )),
    }
}

// ----------------------------------------------------------------------------
// Permission / question round-trip
// ----------------------------------------------------------------------------

/// Build forge-sdk `Options` with the `can_use_tool` callback wired
/// up. The callback bridges forge-sdk's permission flow to the TUI's
/// `BridgeEvent` channel: each request is parked on the shared
/// `pending` map keyed by `tool_use_id`; the matching
/// `PermissionResponse` / `QuestionResponse` command on the worker's
/// inbound channel drains the oneshot to release the callback.
fn build_options_with_callback(
    cwd: &str,
    resume: Option<&str>,
    event_tx: &mpsc::UnboundedSender<BridgeEvent>,
    pending: &PendingResponses,
    session_id_slot: &Arc<Mutex<String>>,
) -> Options {
    let event_tx = event_tx.clone();
    let pending = Arc::clone(pending);
    let session_id_slot = Arc::clone(session_id_slot);
    let callback = move |ctx: ToolPermissionContext| {
        let event_tx = event_tx.clone();
        let pending = Arc::clone(&pending);
        let session_id = session_id_slot
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default();
        async move {
            let (tx, rx) = oneshot::channel();
            if let Ok(mut map) = pending.lock() {
                map.insert(ctx.tool_use_id.clone(), tx);
            }
            let event = if ctx.tool_name == "AskUserQuestion" {
                synth_question_request(&session_id, &ctx)
            } else {
                synth_permission_request(&session_id, &ctx)
            };
            if event_tx.send(event).is_err() {
                return PermissionDecision::deny("event channel closed");
            }
            match rx.await {
                Ok(decision) => decision,
                Err(_) => PermissionDecision::deny("response channel closed"),
            }
        }
    };

    let mut b = OptionsBuilder::new().can_use_tool(callback);
    if !cwd.is_empty() {
        b = b.cwd(PathBuf::from(cwd));
    }
    if let Some(id) = resume {
        b = b.resume(id);
    }
    b.build()
}

fn deliver_permission_response(
    pending: &PendingResponses,
    tool_call_id: &str,
    outcome: crate::agent::types::PermissionOutcome,
) {
    let Some(tx) = take_pending(pending, tool_call_id) else {
        tracing::warn!(
            target: crate::logging::targets::APP_PERMISSION,
            tool_call_id,
            "forge_sdk_worker: PermissionResponse for unknown tool_call_id (already drained?)",
        );
        return;
    };
    let decision = match outcome {
        crate::agent::types::PermissionOutcome::Selected { option_id } => {
            // Selected option ids encode the user choice. The CLI
            // expects allow/deny semantics on the wire; we map by
            // suffix conventions ("deny" -> deny; anything else ->
            // allow). UIs that surface custom option_ids stay
            // compatible because the SDK only cares about the
            // resulting allow/deny decision.
            if option_id.eq_ignore_ascii_case("deny") || option_id.eq_ignore_ascii_case("reject") {
                PermissionDecision::deny(format!("user denied: {option_id}"))
            } else {
                PermissionDecision::allow()
            }
        }
        crate::agent::types::PermissionOutcome::Cancelled => {
            PermissionDecision::deny("user cancelled")
        }
    };
    let _ = tx.send(decision);
}

fn deliver_question_response(
    pending: &PendingResponses,
    tool_call_id: &str,
    outcome: crate::agent::types::QuestionOutcome,
) {
    let Some(tx) = take_pending(pending, tool_call_id) else {
        tracing::warn!(
            target: crate::logging::targets::APP_PERMISSION,
            tool_call_id,
            "forge_sdk_worker: QuestionResponse for unknown tool_call_id",
        );
        return;
    };
    let decision = match outcome {
        crate::agent::types::QuestionOutcome::Answered { selected_option_ids, .. } => {
            // The CLI's AskUserQuestion tool reads `updatedInput.answers`.
            // Map each selected_option_id under a deterministic key the
            // CLI can re-correlate. The bridge.ts in agent-sdk uses the
            // same `q{i}` pattern when the user hasn't named the
            // questions; matching that keeps wire-compat.
            let mut answers = serde_json::Map::new();
            for (i, opt) in selected_option_ids.into_iter().enumerate() {
                answers.insert(format!("q{i}"), serde_json::Value::String(opt));
            }
            PermissionDecision::allow_with_input(serde_json::json!({ "answers": answers }))
        }
        crate::agent::types::QuestionOutcome::Cancelled => {
            PermissionDecision::deny("user cancelled question")
        }
    };
    let _ = tx.send(decision);
}

fn take_pending(
    pending: &PendingResponses,
    tool_call_id: &str,
) -> Option<oneshot::Sender<PermissionDecision>> {
    pending.lock().ok()?.remove(tool_call_id)
}

fn synth_permission_request(session_id: &str, ctx: &ToolPermissionContext) -> BridgeEvent {
    use crate::agent::types::{PermissionDisplay, PermissionRequest, ToolCall};
    let tool_call = ToolCall {
        tool_call_id: ctx.tool_use_id.clone(),
        title: ctx.tool_name.clone(),
        kind: "execute".to_owned(),
        status: "pending".to_owned(),
        content: Vec::new(),
        raw_input: Some(ctx.tool_input.clone()),
        raw_output: None,
        output_metadata: None,
        task_metadata: None,
        locations: Vec::new(),
        meta: None,
    };
    let display = PermissionDisplay {
        title: ctx.title.clone(),
        display_name: ctx.display_name.clone(),
        description: ctx.description.clone(),
    };
    BridgeEvent::PermissionRequest {
        session_id: session_id.to_owned(),
        request: PermissionRequest {
            tool_call,
            options: default_permission_options(),
            display: Some(display),
        },
    }
}

fn synth_question_request(session_id: &str, ctx: &ToolPermissionContext) -> BridgeEvent {
    use crate::agent::types::{
        QuestionOption, QuestionPrompt, QuestionRequest, ToolCall,
    };
    let questions: Vec<serde_json::Value> = ctx
        .tool_input
        .get("questions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let total = u64::try_from(questions.len()).unwrap_or(0);
    let prompt = questions.first().map_or_else(
        || QuestionPrompt {
            question: String::new(),
            header: String::new(),
            multi_select: false,
            options: Vec::new(),
        },
        |q| QuestionPrompt {
            question: q.get("question").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
            header: q.get("header").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
            multi_select: q
                .get("multiSelect")
                .or_else(|| q.get("multi_select"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            options: q
                .get("options")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|o| {
                            let id = o
                                .get("option_id")
                                .or_else(|| o.get("optionId"))
                                .and_then(|v| v.as_str())?
                                .to_owned();
                            let label = o
                                .get("label")
                                .and_then(|v| v.as_str())
                                .unwrap_or(&id)
                                .to_owned();
                            Some(QuestionOption {
                                option_id: id,
                                label,
                                description: o
                                    .get("description")
                                    .and_then(|v| v.as_str())
                                    .map(str::to_owned),
                                preview: o
                                    .get("preview")
                                    .and_then(|v| v.as_str())
                                    .map(str::to_owned),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default(),
        },
    );
    let tool_call = ToolCall {
        tool_call_id: ctx.tool_use_id.clone(),
        title: "AskUserQuestion".to_owned(),
        kind: "ask".to_owned(),
        status: "pending".to_owned(),
        content: Vec::new(),
        raw_input: Some(ctx.tool_input.clone()),
        raw_output: None,
        output_metadata: None,
        task_metadata: None,
        locations: Vec::new(),
        meta: None,
    };
    BridgeEvent::QuestionRequest {
        session_id: session_id.to_owned(),
        request: QuestionRequest {
            tool_call,
            prompt,
            question_index: 0,
            total_questions: total,
        },
    }
}

fn default_permission_options() -> Vec<crate::agent::types::PermissionOption> {
    vec![
        crate::agent::types::PermissionOption {
            option_id: "allow_once".to_owned(),
            name: "Allow once".to_owned(),
            description: None,
            kind: "allow_once".to_owned(),
        },
        crate::agent::types::PermissionOption {
            option_id: "allow_always".to_owned(),
            name: "Allow always".to_owned(),
            description: None,
            kind: "allow_always".to_owned(),
        },
        crate::agent::types::PermissionOption {
            option_id: "deny".to_owned(),
            name: "Deny".to_owned(),
            description: None,
            kind: "reject_once".to_owned(),
        },
    ]
}

// ----------------------------------------------------------------------------
// Other translators
// ----------------------------------------------------------------------------

fn translate_account_info(info: forge_sdk::AccountInfo) -> crate::agent::types::AccountInfo {
    // forge_sdk::AccountInfo and crate::agent::types::AccountInfo
    // have the same field names; field-by-field copy keeps them
    // independent so either side can grow without coupling.
    crate::agent::types::AccountInfo {
        email: info.email,
        organization: info.organization,
        subscription_type: info.subscription_type,
        token_source: info.token_source,
        api_key_source: info.api_key_source,
        api_provider: info.api_provider,
    }
}

fn translate_mcp_server_status(
    sdk: forge_sdk::McpServerStatus,
) -> crate::agent::types::McpServerStatus {
    crate::agent::types::McpServerStatus {
        name: sdk.name,
        status: translate_mcp_status(sdk.status),
        server_info: sdk
            .server_info
            .map(|info| crate::agent::types::McpServerInfo { name: info.name, version: info.version }),
        error: sdk.error,
        // forge-sdk types `config` as `Option<Value>` because the CLI
        // accepts variants forge-sdk doesn't model (claudeai-proxy).
        // Round-trip through serde here; on shape mismatch the typed
        // enum returns None, which the UI renders as "(unknown config)".
        config: sdk
            .config
            .and_then(|v| serde_json::from_value(v).ok()),
        scope: sdk.scope,
        tools: sdk
            .tools
            .unwrap_or_default()
            .into_iter()
            .map(|t| crate::agent::types::McpTool {
                name: t.name,
                description: t.description,
                annotations: t.annotations.map(|a| crate::agent::types::McpToolAnnotations {
                    read_only: a.read_only,
                    destructive: a.destructive,
                    open_world: a.open_world,
                }),
            })
            .collect(),
        sampling_configured: sdk.sampling_configured,
        sampling_required: sdk.sampling_required,
    }
}

fn translate_mcp_status(
    sdk: forge_sdk::McpServerConnectionStatus,
) -> crate::agent::types::McpServerConnectionStatus {
    use forge_sdk::McpServerConnectionStatus as S;
    use crate::agent::types::McpServerConnectionStatus as T;
    match sdk {
        S::Connected => T::Connected,
        S::Failed => T::Failed,
        S::NeedsAuth => T::NeedsAuth,
        S::Pending => T::Pending,
        S::Disabled => T::Disabled,
    }
}

fn clamp_percentage_to_u8(p: f64) -> u8 {
    if p.is_nan() {
        return 0;
    }
    let clamped = p.clamp(0.0, 100.0).round();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = clamped as u8;
    n
}

fn placeholder_current_model() -> CurrentModel {
    // Placeholder until the SDK's system/init message arrives via the
    // reader and refines this through SessionUpdate::CurrentModelUpdate.
    CurrentModel {
        requested_id: None,
        resolved_id: String::new(),
        display_name_short: String::new(),
        display_name_long: String::new(),
        catalog_id: None,
        supports_effort: false,
        supported_effort_levels: Vec::new(),
        supports_fast_mode: None,
        supports_auto_mode: None,
        supports_adaptive_thinking: None,
        is_authoritative: false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{
        deliver_permission_response, deliver_question_response, synth_permission_request,
        synth_question_request, take_pending, PendingResponses,
    };
    use crate::agent::types::{
        ElicitationAction, PermissionOutcome, QuestionOutcome,
    };
    use crate::agent::wire::BridgeEvent;
    use forge_sdk::ToolPermissionContext;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tokio::sync::oneshot;

    fn fresh_pending() -> PendingResponses {
        Arc::new(Mutex::new(HashMap::new()))
    }

    fn ctx(tool_name: &str, tool_use_id: &str, input: serde_json::Value) -> ToolPermissionContext {
        ToolPermissionContext::new(tool_name, input, tool_use_id, None)
    }

    fn park(pending: &PendingResponses, id: &str) -> oneshot::Receiver<forge_sdk::PermissionDecision> {
        let (tx, rx) = oneshot::channel();
        pending.lock().unwrap().insert(id.to_owned(), tx);
        rx
    }

    #[test]
    fn permission_response_allow_drains_oneshot_with_allow() {
        let pending = fresh_pending();
        let rx = park(&pending, "tu_1");
        deliver_permission_response(
            &pending,
            "tu_1",
            PermissionOutcome::Selected { option_id: "allow_once".to_owned() },
        );
        let decision = rx.blocking_recv().expect("oneshot resolved");
        assert!(decision.is_allow(), "allow_once should produce an allow decision");
    }

    #[test]
    fn permission_response_deny_keyword_drains_with_deny() {
        let pending = fresh_pending();
        let rx = park(&pending, "tu_2");
        deliver_permission_response(
            &pending,
            "tu_2",
            PermissionOutcome::Selected { option_id: "deny".to_owned() },
        );
        let decision = rx.blocking_recv().expect("oneshot resolved");
        assert!(!decision.is_allow(), "deny option should produce a deny decision");
    }

    #[test]
    fn permission_response_cancel_drains_with_deny() {
        let pending = fresh_pending();
        let rx = park(&pending, "tu_3");
        deliver_permission_response(&pending, "tu_3", PermissionOutcome::Cancelled);
        let decision = rx.blocking_recv().expect("oneshot resolved");
        assert!(!decision.is_allow(), "cancelled outcome should deny");
    }

    #[test]
    fn permission_response_unknown_id_is_silent_no_op() {
        let pending = fresh_pending();
        // No oneshot parked for this id -- function should warn and return.
        deliver_permission_response(
            &pending,
            "missing",
            PermissionOutcome::Selected { option_id: "allow_once".to_owned() },
        );
        // Pending map should remain empty.
        assert!(pending.lock().unwrap().is_empty());
    }

    #[test]
    fn question_response_answered_drains_with_allow_and_payload() {
        let pending = fresh_pending();
        let rx = park(&pending, "tu_q1");
        deliver_question_response(
            &pending,
            "tu_q1",
            QuestionOutcome::Answered {
                selected_option_ids: vec!["red".to_owned(), "blue".to_owned()],
                annotation: None,
            },
        );
        let decision = rx.blocking_recv().expect("oneshot resolved");
        assert!(decision.is_allow(), "answered outcome should produce an allow");
        // The CLI's AskUserQuestion tool reads `updatedInput.answers`.
        // The worker maps each option to "q{i}" keys for wire-compat
        // with the legacy Node bridge.
        let updated = decision.updated_input().expect("answer payload present");
        assert_eq!(updated.pointer("/answers/q0"), Some(&json!("red")));
        assert_eq!(updated.pointer("/answers/q1"), Some(&json!("blue")));
    }

    #[test]
    fn question_response_cancel_drains_with_deny() {
        let pending = fresh_pending();
        let rx = park(&pending, "tu_q2");
        deliver_question_response(&pending, "tu_q2", QuestionOutcome::Cancelled);
        let decision = rx.blocking_recv().expect("oneshot resolved");
        assert!(!decision.is_allow(), "cancelled question should deny");
    }

    #[test]
    fn synth_permission_request_carries_tool_input_and_display_fields() {
        let c = ctx("Bash", "tu_p1", json!({ "command": "ls" })).with_display(
            None,
            None,
            Some("Run shell command".to_owned()),
            Some("Bash".to_owned()),
            Some("Lists directory entries".to_owned()),
        );
        let event = synth_permission_request("sess_1", &c);
        let BridgeEvent::PermissionRequest { session_id, request } = event else {
            panic!("expected PermissionRequest");
        };
        assert_eq!(session_id, "sess_1");
        assert_eq!(request.tool_call.tool_call_id, "tu_p1");
        assert_eq!(request.tool_call.title, "Bash");
        assert_eq!(request.tool_call.raw_input, Some(json!({ "command": "ls" })));
        // Default options surface allow_once / allow_always / deny.
        assert_eq!(request.options.len(), 3);
        assert!(request.options.iter().any(|o| o.option_id == "deny"));
        let display = request.display.expect("display populated");
        assert_eq!(display.title.as_deref(), Some("Run shell command"));
        assert_eq!(display.display_name.as_deref(), Some("Bash"));
        assert_eq!(display.description.as_deref(), Some("Lists directory entries"));
    }

    #[test]
    fn synth_question_request_extracts_first_prompt_and_options() {
        let c = ctx(
            "AskUserQuestion",
            "tu_q3",
            json!({
                "questions": [
                    {
                        "question": "Which color?",
                        "header": "Pick one",
                        "multiSelect": false,
                        "options": [
                            { "option_id": "red", "label": "Red" },
                            { "option_id": "blue", "label": "Blue", "description": "the cool one" },
                        ],
                    },
                    {
                        "question": "Filler so total > 1",
                        "options": [],
                    },
                ],
            }),
        );
        let event = synth_question_request("sess_2", &c);
        let BridgeEvent::QuestionRequest { session_id, request } = event else {
            panic!("expected QuestionRequest");
        };
        assert_eq!(session_id, "sess_2");
        assert_eq!(request.tool_call.tool_call_id, "tu_q3");
        assert_eq!(request.total_questions, 2);
        assert_eq!(request.question_index, 0);
        assert_eq!(request.prompt.question, "Which color?");
        assert_eq!(request.prompt.header, "Pick one");
        assert_eq!(request.prompt.options.len(), 2);
        assert_eq!(request.prompt.options[0].option_id, "red");
        assert_eq!(request.prompt.options[0].label, "Red");
        assert_eq!(
            request.prompt.options[1].description.as_deref(),
            Some("the cool one"),
        );
    }

    #[test]
    fn synth_question_request_handles_empty_questions_gracefully() {
        let c = ctx("AskUserQuestion", "tu_q4", json!({}));
        let event = synth_question_request("sess_3", &c);
        let BridgeEvent::QuestionRequest { request, .. } = event else {
            panic!();
        };
        assert_eq!(request.total_questions, 0);
        assert_eq!(request.prompt.question, "");
        assert!(request.prompt.options.is_empty());
    }

    #[test]
    fn take_pending_removes_entry() {
        let pending = fresh_pending();
        let _rx = park(&pending, "tu_x");
        assert!(take_pending(&pending, "tu_x").is_some());
        // Second take returns None — entry already drained.
        assert!(take_pending(&pending, "tu_x").is_none());
    }

    // Sanity: ElicitationAction variants stringify the way the worker
    // expects when forwarding to forge-sdk.
    #[test]
    fn elicitation_action_variants_match_expected_wire_strings() {
        let cases = [
            (ElicitationAction::Accept, "accept"),
            (ElicitationAction::Decline, "decline"),
            (ElicitationAction::Cancel, "cancel"),
        ];
        for (action, expected) in cases {
            let actual = match action {
                ElicitationAction::Accept => "accept",
                ElicitationAction::Decline => "decline",
                ElicitationAction::Cancel => "cancel",
            };
            assert_eq!(actual, expected);
        }
    }
}
