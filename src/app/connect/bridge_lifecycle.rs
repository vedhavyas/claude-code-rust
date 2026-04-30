// Copyright 2025 Simon Peter Rothgang
// SPDX-License-Identifier: Apache-2.0

//! Bridge process lifecycle: spawning, initialization handshake, event loop,
//! and connection slot management.

#[allow(unused_imports)] // BridgeLauncher used by the legacy Node-bridge path; PR #3 removes.
use crate::agent::bridge::BridgeLauncher;
#[allow(unused_imports)] // AgentConnection / BridgeClient idem; AgentBridge stays.
use crate::agent::client::{AgentBridge, AgentConnection, BridgeClient};
use crate::agent::events::ClientEvent;
use crate::agent::forge_sdk_bridge::{ForgeSdkBridge, ForgeSdkCommand};
use crate::agent::forge_sdk_worker;
use crate::agent::wire::{BridgeCommand, BridgeEvent, CommandEnvelope, EventEnvelope};
use crate::error::AppError;
use std::rc::Rc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{Instrument as _, info_span};

use super::event_dispatch::handle_bridge_event;
use super::{ConnectionSlot, StartConnectionParams, extract_app_error};

pub(super) async fn run_connection_task(
    params: StartConnectionParams,
    conn_slot_writer: Rc<std::cell::RefCell<Option<ConnectionSlot>>>,
) {
    let request_kind = if params.resume_id.is_some() { "resume" } else { "create" };
    let session_id = params.resume_id.clone().unwrap_or_default();
    let connection_span = info_span!(
        target: crate::logging::targets::BRIDGE_LIFECYCLE,
        "bridge_connection",
        request_kind,
        resume_requested = params.resume_requested,
        session_id = %session_id,
        cwd = %params.cwd_raw,
    );

    async move {
        tracing::debug!(
            target: crate::logging::targets::BRIDGE_LIFECYCLE,
            event_name = "bridge_connection_task_started",
            message = "bridge connection task started",
            outcome = "start",
            request_kind,
            resume_requested = params.resume_requested,
            session_id = %session_id,
        );

        let mut connected_once = false;
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<ForgeSdkCommand>();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel::<BridgeEvent>();

        let agent: Rc<dyn AgentBridge> =
            Rc::new(ForgeSdkBridge::new(cmd_tx.clone())) as Rc<dyn AgentBridge>;
        publish_connection_slot(&conn_slot_writer, Rc::clone(&agent));

        // Worker owns the forge_sdk::Client and drains commands.
        // Send-safe future, runs on the multi-threaded runtime alongside
        // the LocalSet-backed UI tasks.
        tokio::spawn(forge_sdk_worker::run_worker(cmd_rx, event_tx));

        // Issue the initial session command. With Node bridge this
        // happened over NDJSON via send_session_command; now it's a
        // direct trait call that the worker translates into
        // forge_sdk::Client::spawn.
        let send_result = if let Some(resume_id) = params.resume_id.clone() {
            agent.resume_session(resume_id, params.session_launch_settings.clone())
        } else {
            agent.new_session(
                params.cwd_raw.clone(),
                params.session_launch_settings.clone(),
            )
        };
        if let Err(err) = send_result {
            emit_connection_failed(
                &params.event_tx,
                format!("Failed to start forge-sdk session: {err}"),
                AppError::ConnectionFailed,
            );
            return;
        }

        forge_sdk_event_loop(&params, &mut event_rx, &agent, &mut connected_once).await;
    }
    .instrument(connection_span)
    .await;
}

// ----------------------------------------------------------------------------
// Legacy Node-bridge spawn path. The active path above uses the forge-sdk
// worker; the helpers below are unreferenced and stay only to keep this
// commit's diff focused on the swap. PR #3 deletes them alongside
// agent/bridge.rs, agent/client.rs::BridgeClient, and agent-sdk/.
// ----------------------------------------------------------------------------

#[allow(dead_code)]
fn resolve_launcher(params: &StartConnectionParams) -> Option<BridgeLauncher> {
    match crate::agent::bridge::resolve_bridge_launcher(params.bridge_script.as_deref()) {
        Ok(launcher) => Some(launcher),
        Err(err) => {
            tracing::error!(
                target: crate::logging::targets::BRIDGE_LIFECYCLE,
                event_name = "bridge_launcher_resolution_failed",
                message = "failed to resolve bridge launcher",
                outcome = "failure",
                error = %err,
            );
            let app_error = extract_app_error(&err).unwrap_or(AppError::ConnectionFailed);
            emit_connection_failed(
                &params.event_tx,
                format!("Failed to resolve bridge launcher: {err}"),
                app_error,
            );
            None
        }
    }
}

#[allow(dead_code)]
fn spawn_bridge_client(
    event_tx: &mpsc::UnboundedSender<ClientEvent>,
    launcher: &BridgeLauncher,
) -> Option<BridgeClient> {
    match BridgeClient::spawn(launcher) {
        Ok(client) => Some(client),
        Err(err) => {
            tracing::error!(
                target: crate::logging::targets::BRIDGE_LIFECYCLE,
                event_name = "bridge_spawn_failed",
                message = "failed to spawn bridge process",
                outcome = "failure",
                error = %err,
            );
            let app_error = extract_app_error(&err).unwrap_or(AppError::AdapterCrashed);
            emit_connection_failed(event_tx, format!("Failed to spawn bridge: {err}"), app_error);
            None
        }
    }
}

fn publish_connection_slot(
    conn_slot_writer: &Rc<std::cell::RefCell<Option<ConnectionSlot>>>,
    agent: Rc<dyn AgentBridge>,
) {
    *conn_slot_writer.borrow_mut() = Some(ConnectionSlot { conn: agent });
}

/// Forge-sdk event relay: drain `BridgeEvent`s emitted by the
/// forge-sdk worker and feed them into the existing
/// `handle_bridge_event` dispatcher. The dispatcher is unaware which
/// backend produced the event because the wire shape (`BridgeEvent`)
/// is identical to what the Node bridge would have produced.
async fn forge_sdk_event_loop(
    params: &StartConnectionParams,
    event_rx: &mut mpsc::UnboundedReceiver<BridgeEvent>,
    agent: &Rc<dyn AgentBridge>,
    connected_once: &mut bool,
) {
    while let Some(event) = event_rx.recv().await {
        let envelope = EventEnvelope { request_id: None, event };
        handle_bridge_event(
            &params.event_tx,
            agent,
            connected_once,
            params.resume_requested,
            envelope,
        );
    }
    tracing::info!(
        target: crate::logging::targets::BRIDGE_LIFECYCLE,
        event_name = "forge_sdk_event_loop_exited",
        message = "forge-sdk worker channel closed; connection task exiting",
        outcome = "success",
    );
}

#[allow(dead_code)]
async fn send_initialize_command(
    params: &StartConnectionParams,
    bridge: &mut BridgeClient,
) -> bool {
    let init_cmd = CommandEnvelope {
        request_id: None,
        command: BridgeCommand::Initialize {
            cwd: params.cwd_raw.clone(),
            metadata: std::collections::BTreeMap::new(),
        },
    };
    if let Err(err) = bridge.send(init_cmd).await {
        emit_connection_failed(
            &params.event_tx,
            format!("Failed to initialize bridge: {err}"),
            AppError::ConnectionFailed,
        );
        return false;
    }
    true
}

#[allow(dead_code)]
fn build_session_command(params: &StartConnectionParams) -> CommandEnvelope {
    if let Some(resume) = &params.resume_id {
        CommandEnvelope {
            request_id: None,
            command: BridgeCommand::ResumeSession {
                session_id: resume.clone(),
                launch_settings: params.session_launch_settings.clone(),
                metadata: std::collections::BTreeMap::new(),
            },
        }
    } else {
        CommandEnvelope {
            request_id: None,
            command: BridgeCommand::CreateSession {
                cwd: params.cwd_raw.clone(),
                resume: None,
                launch_settings: params.session_launch_settings.clone(),
                metadata: std::collections::BTreeMap::new(),
            },
        }
    }
}

#[allow(dead_code)]
fn log_session_connect_command_sent(params: &StartConnectionParams, command: &BridgeCommand) {
    let has_language = params.session_launch_settings.language.is_some();
    let has_settings = params.session_launch_settings.settings.is_some();
    let agent_progress_summaries_enabled =
        params.session_launch_settings.agent_progress_summaries.unwrap_or(false);
    match command {
        BridgeCommand::ResumeSession { session_id, .. } => tracing::info!(
            target: crate::logging::targets::APP_SESSION,
            event_name = "session_connect_command_sent",
            message = "session connect command sent to bridge",
            outcome = "success",
            request_kind = "resume",
            resume_requested = true,
            session_id = %session_id,
            has_language,
            has_settings,
            agent_progress_summaries_enabled,
        ),
        BridgeCommand::CreateSession { .. } => tracing::info!(
            target: crate::logging::targets::APP_SESSION,
            event_name = "session_connect_command_sent",
            message = "session connect command sent to bridge",
            outcome = "success",
            request_kind = "create",
            resume_requested = false,
            cwd = %params.cwd_raw,
            has_language,
            has_settings,
            agent_progress_summaries_enabled,
        ),
        _ => {}
    }
}

#[allow(dead_code)]
async fn send_session_command(params: &StartConnectionParams, bridge: &mut BridgeClient) -> bool {
    let command = build_session_command(params);
    if let Err(err) = bridge.send(command.clone()).await {
        emit_connection_failed(
            &params.event_tx,
            format!("Failed to create bridge session: {err}"),
            AppError::ConnectionFailed,
        );
        return false;
    }
    log_session_connect_command_sent(params, &command.command);
    true
}

#[allow(dead_code)]
async fn bridge_event_loop(
    params: &StartConnectionParams,
    bridge: &mut BridgeClient,
    agent: &Rc<dyn AgentBridge>,
    cmd_rx: &mut mpsc::UnboundedReceiver<CommandEnvelope>,
    connected_once: &mut bool,
) {
    loop {
        tokio::select! {
            Some(cmd) = cmd_rx.recv() => {
                if let Err(err) = bridge.send(cmd).await {
                    emit_connection_failed(
                        &params.event_tx,
                        format!("Failed to send bridge command: {err}"),
                        AppError::ConnectionFailed,
                    );
                    break;
                }
            }
            event = bridge.recv() => {
                match event {
                    Ok(Some(envelope)) => {
                        handle_bridge_event(
                            &params.event_tx,
                            agent,
                            connected_once,
                            params.resume_requested,
                            envelope,
                        );
                    }
                    Ok(None) => {
                        tracing::error!(
                            target: crate::logging::targets::BRIDGE_LIFECYCLE,
                            event_name = "bridge_stdout_closed",
                            message = "bridge stdout closed unexpectedly",
                            outcome = "failure",
                        );
                        emit_connection_failed(
                            &params.event_tx,
                            "Bridge process exited unexpectedly".to_owned(),
                            AppError::ConnectionFailed,
                        );
                        break;
                    }
                    Err(err) => {
                        emit_connection_failed(
                            &params.event_tx,
                            format!("Bridge communication failure: {err}"),
                            AppError::ConnectionFailed,
                        );
                        break;
                    }
                }
            }
        }
    }
}

pub(super) fn emit_connection_failed(
    event_tx: &mpsc::UnboundedSender<ClientEvent>,
    message: String,
    app_error: AppError,
) {
    let _ = event_tx.send(ClientEvent::ConnectionFailed(message));
    let _ = event_tx.send(ClientEvent::FatalError(app_error));
}

#[allow(dead_code)]
pub(super) async fn wait_for_bridge_initialized(
    bridge: &mut BridgeClient,
    event_tx: &mpsc::UnboundedSender<ClientEvent>,
    agent: &Rc<dyn AgentBridge>,
    connected_once: &mut bool,
    resume_requested: bool,
) -> Result<(), AppError> {
    let timeout = Duration::from_secs(10);
    let timeout_ms = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
    let initialize_span = info_span!(
        target: crate::logging::targets::BRIDGE_LIFECYCLE,
        "bridge_initialize",
        resume_requested,
        timeout_ms,
    );

    async {
        let started = tokio::time::Instant::now();
        loop {
            let elapsed = tokio::time::Instant::now().saturating_duration_since(started);
            let remaining = timeout.saturating_sub(elapsed);
            if remaining.is_zero() {
                tracing::error!(
                    target: crate::logging::targets::BRIDGE_LIFECYCLE,
                    event_name = "bridge_initialize_timed_out",
                    message = "bridge initialization timed out",
                    outcome = "timeout",
                    timeout_ms,
                );
                return Err(AppError::ConnectionFailed);
            }

            let event = tokio::time::timeout(remaining, bridge.recv()).await;
            match event {
                Ok(Ok(Some(envelope))) => {
                    if matches!(envelope.event, BridgeEvent::Initialized { .. }) {
                        return Ok(());
                    }
                    if matches!(envelope.event, BridgeEvent::ConnectionFailed { .. }) {
                        handle_bridge_event(
                            event_tx,
                            agent,
                            connected_once,
                            resume_requested,
                            envelope,
                        );
                        return Err(AppError::ConnectionFailed);
                    }
                    handle_bridge_event(
                        event_tx,
                        agent,
                        connected_once,
                        resume_requested,
                        envelope,
                    );
                }
                Ok(Ok(None) | Err(_)) | Err(_) => return Err(AppError::ConnectionFailed),
            }
        }
    }
    .instrument(initialize_span)
    .await
}
