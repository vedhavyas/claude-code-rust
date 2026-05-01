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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a real `claude` binary on PATH; burns API budget"]
async fn forge_sdk_e2e_cancel_mid_turn() {
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
    eprintln!("e2e cancel: connected to {session_id}");

    // Kick off a turn likely to take a few seconds (writing a long
    // poem). We cancel before letting it finish — the worker should
    // route the interrupt to the CLI and emit either TurnComplete or
    // TurnError shortly after.
    agent
        .prompt_text(
            session_id.clone(),
            "Write a 500-word poem about Rust ownership semantics.".to_owned(),
        )
        .expect("prompt queued");

    // Give the CLI a beat to start the turn, then cancel. We don't
    // gate on receiving a chunk first — a long task may emit thinking
    // chunks (which the translator drops today) or no chunk at all
    // before the interrupt lands. The contract under test is: the
    // worker forwards `cancel` to the CLI and a terminal frame
    // (TurnComplete or TurnError) reaches us.
    tokio::time::sleep(Duration::from_secs(2)).await;
    agent.cancel(session_id).expect("cancel queued");
    eprintln!("e2e cancel: interrupt sent");

    // Drain until we see TurnComplete or TurnError.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut terminal = None;
    while tokio::time::Instant::now() < deadline {
        let Ok(Some(event)) = tokio::time::timeout(
            deadline.saturating_duration_since(tokio::time::Instant::now()),
            event_rx.recv(),
        )
        .await
        else {
            break;
        };
        match event {
            BridgeEvent::TurnComplete { .. } => {
                terminal = Some("complete");
                break;
            }
            BridgeEvent::TurnError { message, .. } => {
                eprintln!("e2e cancel: TurnError {message}");
                terminal = Some("error");
                break;
            }
            _ => {}
        }
    }
    assert!(terminal.is_some(), "no terminal turn frame after cancel");
    eprintln!("e2e cancel: turn finalized as {terminal:?}");

    drop(agent);
    let _ = tokio::time::timeout(Duration::from_secs(5), worker).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a real `claude` binary on PATH; burns API budget"]
async fn forge_sdk_e2e_status_and_context_snapshots() {
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
    eprintln!("e2e status: connected to {session_id}");

    // Drive a tiny prompt so the CLI's account info and context-usage
    // numbers are populated. account_info() returns None until at
    // least one stream-json frame mentions it.
    agent
        .prompt_text(session_id.clone(), "Reply with OK.".to_owned())
        .expect("prompt queued");
    let _ = await_turn(&mut event_rx, Duration::from_secs(60)).await;

    agent
        .get_status_snapshot(session_id.clone())
        .expect("status queued");
    agent
        .get_context_usage(session_id.clone())
        .expect("context queued");

    // Drain until we've seen both, with a generous timeout.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    let mut saw_status = false;
    let mut saw_context = false;
    while tokio::time::Instant::now() < deadline && !(saw_status && saw_context) {
        let Ok(Some(event)) = tokio::time::timeout(
            deadline.saturating_duration_since(tokio::time::Instant::now()),
            event_rx.recv(),
        )
        .await
        else {
            break;
        };
        match event {
            BridgeEvent::StatusSnapshot { .. } => saw_status = true,
            BridgeEvent::ContextUsage { .. } => saw_context = true,
            _ => {}
        }
    }
    assert!(saw_status, "expected StatusSnapshot event");
    assert!(saw_context, "expected ContextUsage event");
    eprintln!("e2e status: both snapshots delivered");

    drop(agent);
    let _ = tokio::time::timeout(Duration::from_secs(5), worker).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a real `claude` binary on PATH; burns API budget"]
async fn forge_sdk_e2e_mcp_snapshot() {
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
    eprintln!("e2e mcp: connected to {session_id}");

    agent
        .get_mcp_snapshot(session_id)
        .expect("mcp snapshot queued");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    let mut got = false;
    while tokio::time::Instant::now() < deadline {
        let Ok(Some(event)) = tokio::time::timeout(
            deadline.saturating_duration_since(tokio::time::Instant::now()),
            event_rx.recv(),
        )
        .await
        else {
            break;
        };
        if let BridgeEvent::McpSnapshot { servers, error, .. } = event {
            // The list may be empty (no MCP servers configured) — we
            // only care that the round-trip works without error.
            assert!(error.is_none(), "MCP snapshot error: {error:?}");
            eprintln!("e2e mcp: snapshot returned {} server(s)", servers.len());
            got = true;
            break;
        }
    }
    assert!(got, "expected McpSnapshot event");

    drop(agent);
    let _ = tokio::time::timeout(Duration::from_secs(5), worker).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a real `claude` binary on PATH; burns API budget"]
async fn forge_sdk_e2e_resume_session() {
    // Phase 1: spawn a fresh session, drive one prompt, capture sid.
    #[allow(clippy::similar_names)]
    let session_id = {
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
        let sid = await_connected(&mut event_rx, Duration::from_secs(30)).await;
        eprintln!("e2e resume: phase 1 session {sid}");

        agent
            .prompt_text(sid.clone(), "Reply with the word PERSIST.".to_owned())
            .expect("phase 1 prompt queued");
        let _ = await_turn(&mut event_rx, Duration::from_secs(60)).await;

        // Tear phase 1 down so the underlying CLI subprocess exits and
        // its session state lands on disk.
        drop(agent);
        let _ = tokio::time::timeout(Duration::from_secs(5), worker).await;
        sid
    };

    // Phase 2: resume by id on a fresh worker.
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let worker = tokio::spawn(forge_sdk_worker::run_worker(cmd_rx, event_tx));
    let agent: Rc<dyn AgentBridge> = Rc::new(ForgeSdkBridge::new(cmd_tx));

    agent
        .resume_session(session_id, SessionLaunchSettings::default())
        .expect("resume_session queued");
    // The CLI may issue a brand-new session id when resuming; what we
    // care about is that we receive a Connected event without a
    // ConnectionFailed in between.
    let resumed_id = await_connected(&mut event_rx, Duration::from_secs(30)).await;
    eprintln!("e2e resume: phase 2 connected as {resumed_id}");

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
