// Copyright 2025 Simon Peter Rothgang
// SPDX-License-Identifier: Apache-2.0

//! Bridge process lifecycle: forge-sdk worker setup, connection slot
//! publication, event-loop relay.

use crate::agent::client::AgentBridge;
use crate::agent::events::ClientEvent;
use crate::agent::forge_sdk_bridge::{ForgeSdkBridge, ForgeSdkCommand};
use crate::agent::forge_sdk_worker;
use crate::agent::wire::{BridgeEvent, EventEnvelope};
use crate::error::AppError;
use std::rc::Rc;
use tokio::sync::mpsc;
use tracing::{Instrument as _, info_span};

use super::event_dispatch::handle_bridge_event;
use super::{ConnectionSlot, StartConnectionParams};

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
        *conn_slot_writer.borrow_mut() = Some(ConnectionSlot { conn: Rc::clone(&agent) });

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


pub(super) fn emit_connection_failed(
    event_tx: &mpsc::UnboundedSender<ClientEvent>,
    message: String,
    app_error: AppError,
) {
    let _ = event_tx.send(ClientEvent::ConnectionFailed(message));
    let _ = event_tx.send(ClientEvent::FatalError(app_error));
}

