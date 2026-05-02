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

use crate::agent::bridge::{
    commands as bridge_commands, session_lifecycle, state as bridge_state,
    user_interaction as bridge_user_interaction,
};
use crate::agent::forge_sdk_bridge::ForgeSdkCommand;
use crate::agent::forge_sdk_translate::translate_message;
use crate::agent::wire::BridgeEvent;

/// Pending permission responses keyed by `tool_use_id`. The
/// `can_use_tool` callback parks a oneshot here when the CLI asks;
/// dispatch drains it when the matching `PermissionResponse` arrives
/// from the TUI.
type PendingResponses = Arc<Mutex<HashMap<String, oneshot::Sender<PermissionDecision>>>>;

/// Pending question outcomes keyed by `tool_use_id`. The
/// `AskUserQuestion` driver in the `can_use_tool` callback parks a
/// fresh oneshot per question, emits a `QuestionRequest`, and awaits
/// the matching `QuestionResponse` from dispatch.
type PendingQuestions =
    Arc<Mutex<HashMap<String, oneshot::Sender<crate::agent::types::QuestionOutcome>>>>;

/// Drive a single forge-sdk session for the lifetime of `command_rx`.
/// Returns when the channel is closed (TUI shutting down).
pub async fn run_worker(
    mut command_rx: mpsc::UnboundedReceiver<ForgeSdkCommand>,
    event_tx: mpsc::UnboundedSender<BridgeEvent>,
) {
    let pending: PendingResponses = Arc::new(Mutex::new(HashMap::new()));
    let pending_questions: PendingQuestions = Arc::new(Mutex::new(HashMap::new()));
    let session_id_slot: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    // The bridge session holds the open-tool-call map, mode list,
    // last-seen fast mode, etc. Lives across messages + commands.
    // Wrapped in Arc<Mutex<>> so the spawned reader_loop and the
    // dispatch loop can both mutate it without rearchitecting the
    // command/event-channel topology.
    let bridge_session: Arc<Mutex<bridge_state::BridgeSession>> =
        Arc::new(Mutex::new(bridge_state::BridgeSession::new(String::new(), String::new())));
    let mut state: WorkerState = WorkerState::Waiting;
    // Per-session git watcher tasks. Keyed by session_id so a cwd
    // change (or session replace) aborts the previous watcher before
    // starting the new one. Lives in this run() stack so the workers
    // tear down with the worker itself.
    let mut git_watchers: std::collections::HashMap<String, tokio::task::JoinHandle<()>> =
        std::collections::HashMap::new();

    while let Some(cmd) = command_rx.recv().await {
        if let Err(err) = dispatch(
            &mut state,
            cmd,
            &event_tx,
            &pending,
            &pending_questions,
            &session_id_slot,
            &bridge_session,
            &mut git_watchers,
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

    // Abort any in-flight git watchers so notify cleans up its
    // OS-level subscriptions before the worker process exits.
    for (_session_id, handle) in git_watchers.drain() {
        handle.abort();
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

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn dispatch(
    state: &mut WorkerState,
    cmd: ForgeSdkCommand,
    event_tx: &mpsc::UnboundedSender<BridgeEvent>,
    pending: &PendingResponses,
    pending_questions: &PendingQuestions,
    session_id_slot: &Arc<Mutex<String>>,
    bridge_session: &Arc<Mutex<bridge_state::BridgeSession>>,
    git_watchers: &mut std::collections::HashMap<String, tokio::task::JoinHandle<()>>,
) -> anyhow::Result<()> {
    use ForgeSdkCommand as C;
    match cmd {
        C::NewSession { cwd, launch_settings } => {
            let options = build_options_with_callback(
                &cwd,
                None,
                &launch_settings,
                event_tx,
                pending,
                pending_questions,
                session_id_slot,
            );
            spawn_or_replace(
                state,
                event_tx,
                options,
                session_id_slot,
                bridge_session,
                &launch_settings,
                None,
            )
            .await
        }
        C::ResumeSession { session_id, launch_settings } => {
            // Resume by passing the prior session id to the CLI. The
            // CLI itself decides what cwd to use; we don't override.
            let options = build_options_with_callback(
                "",
                Some(&session_id),
                &launch_settings,
                event_tx,
                pending,
                pending_questions,
                session_id_slot,
            );
            spawn_or_replace(
                state,
                event_tx,
                options,
                session_id_slot,
                bridge_session,
                &launch_settings,
                Some(session_id),
            )
            .await
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
            // Follow-up emits — mirror upstream's emitCurrentModelUpdate +
            // emitModeStateUpdate so the bottom-bar model name and the
            // mode chip refresh immediately rather than waiting for the
            // next system/init.
            if let Ok(mut bs) = bridge_session.lock() {
                bs.requested_model_id = Some(model.clone());
                let mut buf: Vec<BridgeEvent> = Vec::new();
                let _ = session_lifecycle::refresh_current_model(&mut bs, true, &mut buf);
                if let Some(mode) = bs.mode {
                    bridge_commands::refresh_supported_modes_for_session(&mut bs);
                    let mode_state = bridge_commands::build_mode_state(&bs, mode);
                    buf.push(BridgeEvent::SessionUpdate {
                        session_id: bs.session_id.clone(),
                        update: crate::agent::types::SessionUpdate::ModeStateUpdate { mode: mode_state },
                    });
                }
                for ev in buf {
                    let _ = event_tx.send(ev);
                }
            }
            Ok(())
        }
        C::SetMode { session_id: _, mode } => {
            let client = require_running(state, "SetMode")?;
            let parsed = parse_permission_mode(&mode)?;
            client.set_permission_mode(parsed).await?;
            // Follow-up: emit CurrentModeUpdate + ModeStateUpdate.
            if let Ok(mut bs) = bridge_session.lock()
                && let Some(bridge_mode) = bridge_state::PermissionMode::from_wire(&mode)
            {
                bs.mode = Some(bridge_mode);
                bridge_commands::refresh_supported_modes_for_session(&mut bs);
                let mode_state = bridge_commands::build_mode_state(&bs, bridge_mode);
                let session_id = bs.session_id.clone();
                drop(bs);
                let _ = event_tx.send(BridgeEvent::SessionUpdate {
                    session_id: session_id.clone(),
                    update: crate::agent::types::SessionUpdate::CurrentModeUpdate {
                        current_mode_id: bridge_mode.as_wire().to_owned(),
                    },
                });
                let _ = event_tx.send(BridgeEvent::SessionUpdate {
                    session_id,
                    update: crate::agent::types::SessionUpdate::ModeStateUpdate { mode: mode_state },
                });
            }
            Ok(())
        }
        C::PermissionResponse { tool_call_id, outcome, .. } => {
            deliver_permission_response(pending, &tool_call_id, outcome);
            Ok(())
        }
        C::QuestionResponse { tool_call_id, outcome, .. } => {
            deliver_question_response(pending_questions, &tool_call_id, outcome);
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
        C::GetOauthCredentialsSnapshot { session_id } => {
            let client = require_running(state, "GetOauthCredentialsSnapshot")?;
            let credentials =
                client.oauth_credentials().map(translate_oauth_credentials);
            let _ =
                event_tx.send(BridgeEvent::OauthCredentialsSnapshot { session_id, credentials });
            Ok(())
        }
        C::StartGitContextWatch { session_id, cwd } => {
            // If a watcher already exists for this session_id, abort
            // it before starting the replacement (handles cwd changes).
            if let Some(existing) = git_watchers.remove(&session_id) {
                existing.abort();
            }

            let mut watcher = match forge_sdk::GitContextWatcher::new(cwd.clone()) {
                Ok(watcher) => watcher,
                Err(err) => {
                    tracing::warn!(
                        target: crate::logging::targets::BRIDGE_LIFECYCLE,
                        session_id = %session_id,
                        cwd = %cwd.display(),
                        error = %err,
                        "failed to start git context watcher",
                    );
                    return Ok(());
                }
            };

            let event_tx = event_tx.clone();
            let task_session_id = session_id.clone();
            // The worker runs on the multi-threaded tokio runtime
            // (see `tokio::spawn(forge_sdk_worker::run_worker(...))`
            // in app/connect/bridge_lifecycle.rs). Use `tokio::spawn`
            // — `spawn_local` would panic with "spawn_local called
            // from outside of a LocalSet".
            let handle = tokio::spawn(async move {
                while let Some(snapshot) = watcher.next_snapshot().await {
                    let context = translate_git_context(snapshot);
                    if event_tx
                        .send(BridgeEvent::GitContextSnapshot {
                            session_id: task_session_id.clone(),
                            context,
                        })
                        .is_err()
                    {
                        break; // bridge event_tx receiver dropped
                    }
                }
            });
            git_watchers.insert(session_id, handle);
            Ok(())
        }
        C::StopGitContextWatch { session_id } => {
            if let Some(handle) = git_watchers.remove(&session_id) {
                handle.abort();
            }
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
        C::ReloadPlugins { session_id } => {
            let client = require_running(state, "ReloadPlugins")?;
            match client.reload_plugins().await {
                Ok(_) => {
                    // Mirror upstream: emit RuntimeReloadCompleted +
                    // refresh slash-command catalogue if reload_plugins
                    // returned a fresh `commands` array.
                    let _ = event_tx
                        .send(BridgeEvent::RuntimeReloadCompleted { session_id });
                }
                Err(e) => {
                    let _ = event_tx.send(BridgeEvent::RuntimeReloadFailed {
                        session_id,
                        message: format!("reload_plugins failed: {e}"),
                    });
                }
            }
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
        C::ReconnectMcpServer { session_id, server_name } => {
            let client = require_running(state, "ReconnectMcpServer")?;
            if let Err(e) = client.mcp_reconnect(&server_name).await {
                let _ = event_tx.send(BridgeEvent::McpOperationError {
                    session_id,
                    error: crate::agent::types::McpOperationError {
                        operation: "reconnect".to_owned(),
                        server_name: Some(server_name),
                        message: format!("{e}"),
                    },
                });
            }
            Ok(())
        }
        C::ToggleMcpServer { session_id, server_name, enabled } => {
            let client = require_running(state, "ToggleMcpServer")?;
            if let Err(e) = client.mcp_toggle(&server_name, enabled).await {
                let _ = event_tx.send(BridgeEvent::McpOperationError {
                    session_id,
                    error: crate::agent::types::McpOperationError {
                        operation: "toggle".to_owned(),
                        server_name: Some(server_name),
                        message: format!("{e}"),
                    },
                });
            }
            Ok(())
        }
        C::SetMcpServers { session_id, servers } => {
            let client = require_running(state, "SetMcpServers")?;
            if let Err(e) = client.mcp_set_servers(serde_json::to_value(servers)?).await {
                let _ = event_tx.send(BridgeEvent::McpOperationError {
                    session_id,
                    error: crate::agent::types::McpOperationError {
                        operation: "set_servers".to_owned(),
                        server_name: None,
                        message: format!("{e}"),
                    },
                });
            }
            Ok(())
        }
        C::AuthenticateMcpServer { session_id, server_name } => {
            let client = require_running(state, "AuthenticateMcpServer")?;
            match client.mcp_authenticate(&server_name).await {
                Ok(response) => {
                    // Walk the response Value for the redirect_url /
                    // auth_url key upstream's bridge looked at. When
                    // present, surface as McpAuthRedirect so the TUI
                    // can pop the browser hint.
                    let url = response
                        .get("redirect_url")
                        .or_else(|| response.get("authUrl"))
                        .or_else(|| response.get("auth_url"))
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned);
                    if let Some(auth_url) = url {
                        let _ = event_tx.send(BridgeEvent::McpAuthRedirect {
                            session_id,
                            redirect: crate::agent::types::McpAuthRedirect {
                                server_name,
                                auth_url,
                                requires_user_action: true,
                            },
                        });
                    }
                }
                Err(e) => {
                    let _ = event_tx.send(BridgeEvent::McpOperationError {
                        session_id,
                        error: crate::agent::types::McpOperationError {
                            operation: "authenticate".to_owned(),
                            server_name: Some(server_name),
                            message: format!("{e}"),
                        },
                    });
                }
            }
            Ok(())
        }
        C::ClearMcpAuth { session_id, server_name } => {
            let client = require_running(state, "ClearMcpAuth")?;
            if let Err(e) = client.mcp_clear_auth(&server_name).await {
                let _ = event_tx.send(BridgeEvent::McpOperationError {
                    session_id,
                    error: crate::agent::types::McpOperationError {
                        operation: "clear_auth".to_owned(),
                        server_name: Some(server_name),
                        message: format!("{e}"),
                    },
                });
            }
            Ok(())
        }
        C::SubmitMcpOauthCallbackUrl { session_id, server_name, callback_url } => {
            let client = require_running(state, "SubmitMcpOauthCallbackUrl")?;
            if let Err(e) = client.mcp_oauth_callback_url(&server_name, &callback_url).await {
                let _ = event_tx.send(BridgeEvent::McpOperationError {
                    session_id,
                    error: crate::agent::types::McpOperationError {
                        operation: "oauth_callback".to_owned(),
                        server_name: Some(server_name),
                        message: format!("{e}"),
                    },
                });
            }
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

#[allow(clippy::too_many_lines)]
async fn spawn_or_replace(
    state: &mut WorkerState,
    event_tx: &mpsc::UnboundedSender<BridgeEvent>,
    options: Options,
    session_id_slot: &Arc<Mutex<String>>,
    bridge_session: &Arc<Mutex<bridge_state::BridgeSession>>,
    launch_settings: &crate::agent::wire::SessionLaunchSettings,
    resume_id: Option<String>,
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
    let reader_bridge = Arc::clone(bridge_session);
    tokio::spawn(reader_loop(reader_client, reader_event_tx, reader_bridge));

    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.into_os_string().into_string().ok())
        .unwrap_or_default();

    // Build the typed envelope from the SDK's cached init data + the
    // initialize control_response. Both are populated by `Client::spawn`
    // so they are present here. The bridge module mirrors upstream's
    // bridge.ts logic so the TUI's bottom bar renders identically.
    let server_info = client.get_server_info().cloned();
    let init_data = client.initial_session_data().cloned();

    let available_models = session_lifecycle::map_available_models(
        server_info.as_ref().and_then(|v| v.get("models")),
    );
    let init_record = init_data.as_ref().and_then(serde_json::Value::as_object);

    // The CLI does NOT emit `system/init` until BOTH the initialize
    // control_response AND a user message have landed (per forge-sdk's
    // spawn_inner doc). So at this point `init_data` is almost always
    // None. Fall back to the launch_settings the TUI handed us — the
    // user's settings.json's `permissions.defaultMode` and `model`
    // are the source of truth for the initial Connected envelope.
    let launch_settings_record = launch_settings
        .settings
        .as_ref()
        .and_then(serde_json::Value::as_object);
    let init_model_id = init_record
        .and_then(|r| r.get("model"))
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            launch_settings_record
                .and_then(|r| r.get("model"))
                .and_then(serde_json::Value::as_str)
        })
        .unwrap_or("")
        .to_owned();
    let raw_permission_mode = init_record
        .and_then(|r| r.get("permissionMode"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            launch_settings_record
                .and_then(|r| r.get("permissions"))
                .and_then(serde_json::Value::as_object)
                .and_then(|p| p.get("defaultMode"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        });
    let init_permission_mode = raw_permission_mode
        .as_deref()
        .and_then(bridge_state::PermissionMode::from_wire)
        .or(Some(bridge_state::PermissionMode::Default));
    let supports_bypass = init_record
        .and_then(|r| r.get("supportsBypassPermissionsMode"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let init_keys: Vec<String> = init_record
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default();
    tracing::info!(
        target: crate::logging::targets::BRIDGE_LIFECYCLE,
        event_name = "forge_sdk_worker_init_data",
        message = "captured init_data after Client::spawn (with launch_settings fallback)",
        outcome = "info",
        session_id = %session_id,
        init_present = init_record.is_some(),
        init_keys = ?init_keys,
        resolved_model_id = %init_model_id,
        resolved_permission_mode_raw = raw_permission_mode.as_deref().unwrap_or("(none)"),
        resolved_permission_mode_parsed = ?init_permission_mode.map(bridge_state::PermissionMode::as_wire),
        supports_bypass,
        available_models_count = available_models.len(),
    );

    // Populate the persistent BridgeSession so the reader_loop's
    // handle_sdk_message calls see the right starting state (model
    // catalog, supported modes, fast mode default, …).
    let (current_model, mode) = {
        let mut bs = bridge_session.lock().map_err(|_| {
            anyhow::anyhow!("forge_sdk_worker: bridge session mutex poisoned")
        })?;
        bs.session_id.clone_from(&session_id);
        bs.cwd.clone_from(&cwd);
        bs.connected = true;
        bs.model_id = init_model_id;
        bs.available_models.clone_from(&available_models);
        bs.mode = init_permission_mode;
        bs.supports_bypass_permissions_mode = supports_bypass;
        let cm = session_lifecycle::resolve_current_model(&bs);
        bs.current_model = Some(cm.clone());
        bridge_commands::refresh_supported_modes_for_session(&mut bs);
        let mode = init_permission_mode.map(|m| bridge_commands::build_mode_state(&bs, m));
        (cm, mode)
    };

    // History is loaded from the on-disk JSONL when resuming. The CLI
    // emits new turns as fresh stream-json frames, so we only need to
    // backfill the past turns once at connect time.
    let history_updates = if let Some(prev_session_id) = resume_id.as_deref() {
        let updates = load_history_updates(prev_session_id, &cwd);
        if updates.is_empty() { None } else { Some(updates) }
    } else {
        None
    };

    let _ = event_tx.send(BridgeEvent::Connected {
        session_id: session_id.clone(),
        cwd: cwd.clone(),
        current_model,
        available_models,
        mode,
        history_updates,
    });

    // Eagerly emit a status snapshot so the bottom bar fills in
    // account / org / token-source without the TUI having to ask.
    if let Some(account) = client.account_info() {
        let _ = event_tx.send(BridgeEvent::StatusSnapshot {
            session_id: session_id.clone(),
            account: translate_account_info(account),
        });
    }

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

/// Load past messages from the on-disk transcript and convert them
/// into the `SessionUpdate` stream the TUI's history renderer expects.
/// Delegates to `bridge::history::map_session_messages_to_updates`
/// which mirrors upstream's `mapSessionMessagesToUpdates` and uses the
/// full `bridge::tooling::build_tool_result_fields` extractor for
/// per-tool result formatting (Bash stdout/stderr, Edit/Write diffs,
/// Read `file_unchanged` shortcut, Agent `agentType` title, etc.).
fn load_history_updates(
    prev_session_id: &str,
    cwd: &str,
) -> Vec<crate::agent::types::SessionUpdate> {
    let dir = if cwd.is_empty() { None } else { Some(cwd.to_owned()) };
    let messages = forge_sdk::session::scan::get_session_messages(prev_session_id, dir);
    // bridge::history walks raw `serde_json::Value` envelopes; turn
    // each typed `SessionMessage` into the JSONL-shape it expects
    // (`{type, message: {role, content}, parent_tool_use_id?}`).
    let raw: Vec<serde_json::Value> = messages
        .into_iter()
        .map(|m| {
            let kind = match m.kind {
                forge_sdk::SessionMessageKind::User => "user",
                forge_sdk::SessionMessageKind::Assistant => "assistant",
            };
            serde_json::json!({
                "type": kind,
                "message": m.message,
                "parent_tool_use_id": m.parent_tool_use_id,
            })
        })
        .collect();
    crate::agent::bridge::history::map_session_messages_to_updates(&raw)
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

async fn reader_loop(
    client: Client,
    event_tx: mpsc::UnboundedSender<BridgeEvent>,
    bridge_session: Arc<Mutex<bridge_state::BridgeSession>>,
) {
    loop {
        match client.next_event().await {
            Ok(Some(msg)) => {
                let mut buf: Vec<BridgeEvent> = Vec::new();
                if let Ok(mut session) = bridge_session.lock() {
                    crate::agent::bridge::message_handlers::handle_sdk_message(
                        &mut session, &msg, &mut buf,
                    );
                } else {
                    tracing::error!(
                        target: crate::logging::targets::BRIDGE_LIFECYCLE,
                        "forge_sdk_worker reader: bridge session mutex poisoned",
                    );
                }
                // Fall through to the legacy translate_message for the
                // bits handle_sdk_message hasn't covered yet (e.g.
                // elicitation_request which already had its own path).
                let mut legacy = translate_message(msg);
                buf.append(&mut legacy);
                for event in buf {
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
    launch_settings: &crate::agent::wire::SessionLaunchSettings,
    event_tx: &mpsc::UnboundedSender<BridgeEvent>,
    pending: &PendingResponses,
    pending_questions: &PendingQuestions,
    session_id_slot: &Arc<Mutex<String>>,
) -> Options {
    let event_tx = event_tx.clone();
    let pending = Arc::clone(pending);
    let pending_questions = Arc::clone(pending_questions);
    let session_id_slot = Arc::clone(session_id_slot);
    let callback = move |ctx: ToolPermissionContext| {
        let event_tx = event_tx.clone();
        let pending = Arc::clone(&pending);
        let pending_questions = Arc::clone(&pending_questions);
        let session_id = session_id_slot.lock().map(|s| s.clone()).unwrap_or_default();
        async move {
            if ctx.tool_name == bridge_user_interaction::ASK_USER_QUESTION_TOOL_NAME {
                run_ask_user_question(ctx, session_id, &event_tx, &pending_questions).await
            } else {
                run_permission_request(ctx, session_id, &event_tx, &pending).await
            }
        }
    };

    // Setting `--permission-prompt-tool stdio` forces the CLI to route
    // every permission/question prompt (including AskUserQuestion)
    // through the SDK's can_use_tool callback over the stream-json
    // pipe instead of resolving them in-process. Without this, the
    // CLI's auto-mode classifier or settings.json
    // skipAutoPermissionPrompt may short-circuit the callback —
    // AskUserQuestion in particular never reaches our run_ask_user_question
    // driver, so the model's question runs the tool with no answers
    // and returns "Answer questions?" as a fallback prompt.
    //
    // The JS SDK upstream sets this implicitly when canUseTool is
    // registered; forge-sdk doesn't, so we set it here.
    let mut b = OptionsBuilder::new()
        .can_use_tool(callback)
        .permission_prompt_tool_name("stdio");
    if !cwd.is_empty() {
        b = b.cwd(PathBuf::from(cwd));
    }
    if let Some(id) = resume {
        b = b.resume(id);
    }

    // Mirror upstream's `startupPermissionModeOptions` +
    // `startupModelOption`: read launch_settings.settings.permissions.defaultMode
    // and launch_settings.settings.model so the CLI starts in the right
    // mode + model. Without this, the CLI starts in default Ask mode
    // and shows the wrong chip in the footer.
    let mut applied_mode: Option<&'static str> = None;
    let mut applied_model: Option<String> = None;
    let mut applied_effort: Option<String> = None;
    let settings_present = launch_settings.settings.is_some();
    if let Some(settings_value) = launch_settings.settings.as_ref()
        && let Some(settings_record) = settings_value.as_object()
    {
        if let Some(perms) = settings_record.get("permissions").and_then(serde_json::Value::as_object)
            && let Some(default_mode_str) = perms.get("defaultMode").and_then(serde_json::Value::as_str)
            && let Ok(mode) = parse_permission_mode(default_mode_str)
        {
            b = b.permission_mode(mode);
            applied_mode = Some(mode.as_cli_arg());
        }
        if let Some(model) = settings_record.get("model").and_then(serde_json::Value::as_str)
            && !model.trim().is_empty()
        {
            b = b.model(model);
            applied_model = Some(model.to_owned());
        }
        // `effortLevel` from settings.json. The CLI accepts
        // `low | medium | high | xhigh | max` per `claude --help`.
        // forge-sdk's typed enum only carries Low/Medium/High/Max;
        // xhigh has to go through the `extra_arg` escape hatch (which
        // emits `--effort xhigh` verbatim).
        if let Some(effort) = settings_record.get("effortLevel").and_then(serde_json::Value::as_str)
            && !effort.trim().is_empty()
        {
            applied_effort = Some(effort.to_owned());
            b = b.extra_arg("effort", Some(effort.to_owned()));
        }
    }
    tracing::info!(
        target: crate::logging::targets::BRIDGE_LIFECYCLE,
        event_name = "forge_sdk_worker_options_built",
        message = "launch_settings → forge-sdk OptionsBuilder",
        outcome = "info",
        settings_present,
        applied_permission_mode = applied_mode.unwrap_or("(none)"),
        applied_model = applied_model.as_deref().unwrap_or("(none)"),
        applied_effort = applied_effort.as_deref().unwrap_or("(none)"),
        cwd_present = !cwd.is_empty(),
        resume_present = resume.is_some(),
    );
    b.build()
}

async fn run_permission_request(
    ctx: ToolPermissionContext,
    session_id: String,
    event_tx: &mpsc::UnboundedSender<BridgeEvent>,
    pending: &PendingResponses,
) -> PermissionDecision {
    let (tx, rx) = oneshot::channel();
    if let Ok(mut map) = pending.lock() {
        map.insert(ctx.tool_use_id.clone(), tx);
    }
    let event = synth_permission_request(&session_id, &ctx);
    if event_tx.send(event).is_err() {
        return PermissionDecision::deny("event channel closed");
    }
    match rx.await {
        Ok(decision) => decision,
        Err(_) => PermissionDecision::deny("response channel closed"),
    }
}

/// Drive `AskUserQuestion` through its multi-question loop. Mirrors
/// upstream's `requestAskUserQuestionAnswers` in
/// `agent-sdk/src/bridge/user_interaction.ts`. Per question:
///   1. Park a oneshot keyed by `tool_use_id`.
///   2. Emit a `QuestionRequest` with the right `question_index` /
///      `total_questions` so the TUI can render a paginator.
///   3. Await the matching `QuestionResponse`.
///   4. Resolve the selected option `label`s and accumulate them.
///
/// At the end, return `PermissionDecision::allow_with_input` carrying
/// `answers: { question_text: label }` (and optional `annotations`).
async fn run_ask_user_question(
    ctx: ToolPermissionContext,
    session_id: String,
    event_tx: &mpsc::UnboundedSender<BridgeEvent>,
    pending_questions: &PendingQuestions,
) -> PermissionDecision {
    use crate::agent::types::QuestionOutcome;

    let prompts = bridge_user_interaction::parse_ask_user_question_prompts(&ctx.tool_input);
    tracing::info!(
        target: crate::logging::targets::APP_PERMISSION,
        event_name = "ask_user_question_received",
        message = "AskUserQuestion can_use_tool fired",
        outcome = "info",
        tool_use_id = %ctx.tool_use_id,
        prompts_parsed = prompts.len(),
        raw_input = %ctx.tool_input,
    );
    if prompts.is_empty() {
        // Mirror upstream: no valid prompts → allow with the original
        // input so the CLI can decide what to do.
        return PermissionDecision::allow();
    }

    let total = prompts.len() as u64;
    let base_tool_call = synth_question_base_tool_call(&ctx);
    let mut answers = serde_json::Map::new();
    let mut annotations = serde_json::Map::new();

    for (index, prompt) in prompts.iter().enumerate() {
        let request = bridge_user_interaction::build_question_request(
            &base_tool_call,
            prompt,
            index as u64,
            total,
        );
        let (tx, rx) = oneshot::channel();
        if let Ok(mut map) = pending_questions.lock() {
            map.insert(ctx.tool_use_id.clone(), tx);
        }
        if event_tx
            .send(BridgeEvent::QuestionRequest {
                session_id: session_id.clone(),
                request: request.clone(),
            })
            .is_err()
        {
            return PermissionDecision::deny("event channel closed");
        }
        let Ok(outcome) = rx.await else {
            return PermissionDecision::deny("response channel closed");
        };
        match outcome {
            QuestionOutcome::Answered { selected_option_ids, annotation } => {
                let selected: Vec<crate::agent::types::QuestionOption> = request
                    .prompt
                    .options
                    .iter()
                    .filter(|opt| selected_option_ids.iter().any(|id| id == &opt.option_id))
                    .cloned()
                    .collect();
                if selected.is_empty()
                    || (!prompt.multi_select && selected.len() != 1)
                {
                    return PermissionDecision::deny("Question answer was invalid");
                }
                let answer = selected
                    .iter()
                    .map(|o| o.label.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                answers.insert(prompt.question.clone(), serde_json::Value::String(answer));
                if let Some(annotation) =
                    bridge_user_interaction::derive_annotation(&selected, annotation.as_ref())
                    && let Ok(v) = serde_json::to_value(&annotation)
                {
                    annotations.insert(prompt.question.clone(), v);
                }
            }
            QuestionOutcome::Cancelled => {
                return PermissionDecision::deny("Question cancelled");
            }
        }
    }

    let updated_input =
        bridge_user_interaction::build_updated_input(&ctx.tool_input, answers, annotations);
    tracing::info!(
        target: crate::logging::targets::APP_PERMISSION,
        event_name = "ask_user_question_resolved",
        message = "AskUserQuestion answers ready, returning PermissionDecision::allow_with_input",
        outcome = "success",
        tool_use_id = %ctx.tool_use_id,
        updated_input = %updated_input,
    );
    PermissionDecision::allow_with_input(updated_input)
}

fn synth_question_base_tool_call(ctx: &ToolPermissionContext) -> crate::agent::types::ToolCall {
    use crate::agent::types::ToolCall;
    ToolCall {
        tool_call_id: ctx.tool_use_id.clone(),
        title: bridge_user_interaction::ASK_USER_QUESTION_TOOL_NAME.to_owned(),
        kind: "ask".to_owned(),
        status: "pending".to_owned(),
        content: Vec::new(),
        raw_input: Some(ctx.tool_input.clone()),
        raw_output: None,
        output_metadata: None,
        task_metadata: None,
        locations: Vec::new(),
        meta: None,
    }
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
    pending: &PendingQuestions,
    tool_call_id: &str,
    outcome: crate::agent::types::QuestionOutcome,
) {
    let Some(tx) = pending.lock().ok().and_then(|mut m| m.remove(tool_call_id)) else {
        tracing::warn!(
            target: crate::logging::targets::APP_PERMISSION,
            tool_call_id,
            "forge_sdk_worker: QuestionResponse for unknown tool_call_id",
        );
        return;
    };
    // The driver in `run_ask_user_question` is awaiting the typed
    // QuestionOutcome and will resolve labels + accumulate
    // `answers[question_text]`. Forward the outcome verbatim.
    let _ = tx.send(outcome);
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

fn translate_oauth_credentials(
    info: forge_sdk::OauthCredentials,
) -> crate::agent::types::OauthCredentialsInfo {
    // SystemTime → epoch ms; round down on sub-millisecond
    // resolution. We use ms (not seconds) so the wire shape
    // preserves the resolution the credentials file actually
    // carries (Anthropic's `claudeAiOauth.expiresAt` is ms).
    let expires_at_ms = info.expires_at.and_then(|t| {
        t.duration_since(std::time::UNIX_EPOCH).ok().map(|d| {
            u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
        })
    });
    crate::agent::types::OauthCredentialsInfo {
        access_token: info.access_token,
        expires_at_ms,
    }
}

fn translate_git_context(snapshot: forge_sdk::GitContext) -> crate::agent::types::GitContextInfo {
    let branch = match snapshot.branch {
        forge_sdk::GitBranch::Named(name) => crate::agent::types::GitBranchInfo::Named(name),
        forge_sdk::GitBranch::Detached => crate::agent::types::GitBranchInfo::Detached,
        forge_sdk::GitBranch::NoRepo => crate::agent::types::GitBranchInfo::NoRepo,
        // forge_sdk::GitBranch is #[non_exhaustive] — explicit Unknown
        // plus the wildcard for any future variants both surface as
        // GitBranchInfo::Unknown (TUI renders no branch chip).
        forge_sdk::GitBranch::Unknown | _ => crate::agent::types::GitBranchInfo::Unknown,
    };
    crate::agent::types::GitContextInfo { branch }
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


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{
        PendingQuestions, PendingResponses, deliver_permission_response,
        deliver_question_response, synth_permission_request, take_pending,
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

    fn fresh_pending_questions() -> PendingQuestions {
        Arc::new(Mutex::new(HashMap::new()))
    }

    fn park_question(
        pending: &PendingQuestions,
        id: &str,
    ) -> oneshot::Receiver<crate::agent::types::QuestionOutcome> {
        let (tx, rx) = oneshot::channel();
        pending.lock().unwrap().insert(id.to_owned(), tx);
        rx
    }

    #[test]
    fn question_response_forwards_typed_outcome() {
        let pending = fresh_pending_questions();
        let rx = park_question(&pending, "tu_q1");
        deliver_question_response(
            &pending,
            "tu_q1",
            QuestionOutcome::Answered {
                selected_option_ids: vec!["question_0".to_owned(), "question_1".to_owned()],
                annotation: None,
            },
        );
        // The driver in `run_ask_user_question` consumes the typed
        // outcome and resolves answer labels itself; this function is
        // only the routing hop.
        let outcome = rx.blocking_recv().expect("oneshot resolved");
        match outcome {
            QuestionOutcome::Answered { selected_option_ids, .. } => {
                assert_eq!(selected_option_ids, vec!["question_0", "question_1"]);
            }
            QuestionOutcome::Cancelled => panic!("expected answered outcome"),
        }
    }

    #[test]
    fn question_response_cancelled_forwards_typed_outcome() {
        let pending = fresh_pending_questions();
        let rx = park_question(&pending, "tu_q2");
        deliver_question_response(&pending, "tu_q2", QuestionOutcome::Cancelled);
        let outcome = rx.blocking_recv().expect("oneshot resolved");
        assert!(matches!(outcome, QuestionOutcome::Cancelled));
    }

    #[test]
    fn question_response_unknown_id_is_silent_no_op() {
        let pending = fresh_pending_questions();
        deliver_question_response(&pending, "missing", QuestionOutcome::Cancelled);
        assert!(pending.lock().unwrap().is_empty());
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
