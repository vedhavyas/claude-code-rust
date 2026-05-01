//! End-to-end smoke test: drive a real `claude` session through the
//! forge-sdk-backed `AgentBridge` worker.
//!
//! Marked `#[ignore]` because it needs a real `claude` binary on PATH
//! and burns a small amount of API budget per run. Run manually with:
//!
//! ```
//! cargo test --test forge_sdk_smoke -- --ignored --nocapture
//! ```

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::manual_assert,
)]

use std::time::Duration;

use claude_code_rust::agent::client::AgentBridge;
use claude_code_rust::agent::forge_sdk_bridge::ForgeSdkBridge;
use claude_code_rust::agent::forge_sdk_worker;
use claude_code_rust::agent::types::SessionUpdate;
use claude_code_rust::agent::wire::{BridgeEvent, SessionLaunchSettings};
use std::rc::Rc;
use tokio::sync::mpsc;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a real `claude` binary on PATH; burns API budget"]
async fn forge_sdk_e2e_round_trip() {
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();

    // Worker runs forge-sdk; lives for the duration of the test.
    let worker = tokio::spawn(forge_sdk_worker::run_worker(cmd_rx, event_tx));

    let agent: Rc<dyn AgentBridge> = Rc::new(ForgeSdkBridge::new(cmd_tx));

    // Kick off a session.
    agent
        .new_session(
            std::env::current_dir()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            SessionLaunchSettings::default(),
        )
        .expect("new_session queued");

    // Wait for Connected within 30s.
    let session_id = await_connected(&mut event_rx, Duration::from_secs(30)).await;
    eprintln!("e2e: connected to session {session_id}");

    // Send a tiny prompt.
    agent
        .prompt_text(session_id.clone(), "Reply with exactly the word OK.".to_owned())
        .expect("prompt_text queued");

    // Wait for the result frame within 60s and verify the assistant said something.
    let saw_text = await_turn_complete(&mut event_rx, Duration::from_secs(60)).await;
    assert!(saw_text, "expected at least one assistant text chunk before turn complete");

    // Tear down.
    drop(agent);
    let _ = tokio::time::timeout(Duration::from_secs(5), worker).await;
}

async fn await_connected(
    rx: &mut mpsc::UnboundedReceiver<BridgeEvent>,
    timeout: Duration,
) -> String {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline
            .saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            panic!("timed out waiting for Connected event");
        }
        let Ok(event) = tokio::time::timeout(remaining, rx.recv()).await else {
            panic!("timed out waiting for Connected event");
        };
        let Some(event) = event else {
            panic!("event channel closed before Connected");
        };
        match event {
            BridgeEvent::Connected { session_id, .. } => return session_id,
            BridgeEvent::ConnectionFailed { message } => {
                panic!("connection failed during smoke test: {message}");
            }
            other => {
                eprintln!("e2e: pre-connected event: {}", other.event_name());
            }
        }
    }
}

async fn await_turn_complete(
    rx: &mut mpsc::UnboundedReceiver<BridgeEvent>,
    timeout: Duration,
) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut saw_text = false;
    loop {
        let remaining = deadline
            .saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            panic!("timed out waiting for TurnComplete");
        }
        let Ok(event) = tokio::time::timeout(remaining, rx.recv()).await else {
            panic!("timed out waiting for TurnComplete");
        };
        let Some(event) = event else {
            panic!("event channel closed before TurnComplete");
        };
        match event {
            BridgeEvent::SessionUpdate { update, .. } => {
                if matches!(update, SessionUpdate::AgentMessageChunk { .. }) {
                    saw_text = true;
                }
                eprintln!("e2e: SessionUpdate {:?}", brief(&update));
            }
            BridgeEvent::TurnComplete { .. } => return saw_text,
            BridgeEvent::TurnError { message, .. } => {
                panic!("turn errored: {message}");
            }
            other => eprintln!("e2e: event: {}", other.event_name()),
        }
    }
}

fn brief(update: &SessionUpdate) -> &'static str {
    match update {
        SessionUpdate::AgentMessageChunk { .. } => "AgentMessageChunk",
        SessionUpdate::AgentThoughtChunk { .. } => "AgentThoughtChunk",
        SessionUpdate::ToolCall { .. } => "ToolCall",
        _ => "(other)",
    }
}
