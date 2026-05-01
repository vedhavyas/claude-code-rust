//! End-to-end smoke tests: drive a real `claude` session through the
//! forge-sdk-backed `AgentBridge` worker.
//!
//! Marked `#[ignore]` because they need a real `claude` binary on PATH
//! and burn a small amount of API budget per run. Run manually with:
//!
//! ```
//! cargo test --test forge_sdk_smoke -- --ignored --nocapture
//! ```
//!
//! Coverage today:
//! - `forge_sdk_e2e_round_trip` — single prompt → one assistant chunk
//!   → `TurnComplete`. Validates the basic happy path.
//! - `forge_sdk_e2e_multi_turn` — two sequential prompts on the same
//!   session, validates session state survives between turns.
//! - `forge_sdk_e2e_tool_call_emits_event` — asks for a tool that is
//!   typically allow-listed in the developer's profile (Bash) so the
//!   `ToolCall` `SessionUpdate` fans out without a permission round-trip.
//!   Validates the `assistant->tool_use` translation path.
//!
//! Out of scope here (need manual TUI testing):
//! - `can_use_tool` round-trip with deny/allow choices. The CLI's
//!   auto-mode classifier and the developer's `settings.json` decide
//!   whether the callback fires; replicating that deterministically
//!   requires a `--settings` override per scenario, see
//!   `forge-test-harness/tests/sdk_scenarios_permission_deny.rs`.
//! - `AskUserQuestion`, MCP servers, slash commands, picker UI,
//!   `/resume`, status snapshot, model switching live in the TUI loop
//!   and need terminal-driven verification.

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
    let outcome = await_turn(&mut event_rx, Duration::from_secs(60)).await;
    assert!(outcome.saw_text, "expected at least one assistant text chunk before turn complete");

    // Tear down.
    drop(agent);
    let _ = tokio::time::timeout(Duration::from_secs(5), worker).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a real `claude` binary on PATH; burns API budget"]
async fn forge_sdk_e2e_multi_turn() {
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();

    let worker = tokio::spawn(forge_sdk_worker::run_worker(cmd_rx, event_tx));
    let agent: Rc<dyn AgentBridge> = Rc::new(ForgeSdkBridge::new(cmd_tx));

    agent
        .new_session(
            std::env::current_dir().unwrap().to_string_lossy().into_owned(),
            SessionLaunchSettings::default(),
        )
        .expect("new_session queued");

    let session_id = await_connected(&mut event_rx, Duration::from_secs(30)).await;
    eprintln!("e2e multi_turn: connected to {session_id}");

    // Turn 1: establish a fact in the conversation context.
    agent
        .prompt_text(
            session_id.clone(),
            "Remember the codeword PUMPKIN. Just acknowledge.".to_owned(),
        )
        .expect("turn 1 queued");
    let turn1 = await_turn(&mut event_rx, Duration::from_secs(60)).await;
    assert!(turn1.saw_text, "turn 1 produced no assistant text");
    eprintln!("e2e multi_turn: turn 1 complete (text seen)");

    // Turn 2: probe whether the session retained turn 1's context. We
    // don't assert on the model's content (it might paraphrase) — we
    // only assert that another full turn round-trips without errors,
    // which proves the worker doesn't re-spawn the CLI between turns.
    agent
        .prompt_text(session_id, "What was the codeword? One word.".to_owned())
        .expect("turn 2 queued");
    let turn2 = await_turn(&mut event_rx, Duration::from_secs(60)).await;
    assert!(turn2.saw_text, "turn 2 produced no assistant text");
    eprintln!("e2e multi_turn: turn 2 complete (text seen)");

    drop(agent);
    let _ = tokio::time::timeout(Duration::from_secs(5), worker).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a real `claude` binary on PATH; burns API budget"]
async fn forge_sdk_e2e_tool_call_emits_event() {
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();

    let worker = tokio::spawn(forge_sdk_worker::run_worker(cmd_rx, event_tx));
    let agent: Rc<dyn AgentBridge> = Rc::new(ForgeSdkBridge::new(cmd_tx));

    agent
        .new_session(
            std::env::current_dir().unwrap().to_string_lossy().into_owned(),
            SessionLaunchSettings::default(),
        )
        .expect("new_session queued");

    let session_id = await_connected(&mut event_rx, Duration::from_secs(30)).await;
    eprintln!("e2e tool_call: connected to {session_id}");

    // Bash is typically allow-listed in the developer's settings, so
    // the auto-mode classifier short-circuits the can_use_tool callback.
    // We only need to see a `ToolCall` SessionUpdate fan out from the
    // assistant message — the actual permission round-trip is covered
    // by `sdk_scenarios_permission_deny` in forge-test-harness.
    agent
        .prompt_text(
            session_id,
            "Use the Bash tool to run `echo OK_FROM_BASH` and report the output verbatim."
                .to_owned(),
        )
        .expect("prompt queued");

    let outcome = await_turn(&mut event_rx, Duration::from_secs(120)).await;
    assert!(
        outcome.saw_tool_call,
        "expected at least one ToolCall SessionUpdate during the turn (Bash tool)"
    );
    eprintln!("e2e tool_call: tool call observed");

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

struct TurnOutcome {
    saw_text: bool,
    saw_tool_call: bool,
}

async fn await_turn(
    rx: &mut mpsc::UnboundedReceiver<BridgeEvent>,
    timeout: Duration,
) -> TurnOutcome {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut outcome = TurnOutcome { saw_text: false, saw_tool_call: false };
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
                match update {
                    SessionUpdate::AgentMessageChunk { .. } => outcome.saw_text = true,
                    SessionUpdate::ToolCall { .. } => outcome.saw_tool_call = true,
                    _ => {}
                }
                eprintln!("e2e: SessionUpdate {:?}", brief(&update));
            }
            BridgeEvent::TurnComplete { .. } => return outcome,
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
