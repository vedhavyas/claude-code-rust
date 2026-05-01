//! Bridge translation layer — Rust port of upstream's
//! `agent-sdk/src/bridge/*.ts` modules. Each submodule mirrors one
//! upstream file by the same name.
//!
//! Goal: produce byte-identical `BridgeEvent` output to upstream's
//! Node bridge so the lifted TUI renders the same on both backends.
//!
//! Entry points are called from `forge_sdk_translate.rs` (live SDK
//! `Message` translation) and `forge_sdk_worker.rs` (lifecycle hooks
//! at session spawn / command dispatch).
//!
//! Doc comments cite upstream identifiers (`tool_use_id`, `gitDiff`,
//! `agentType`, etc.) verbatim so the port is diff-applicable when
//! upstream evolves. The `clippy::doc_markdown` lint flags those as
//! "missing backticks" — silenced module-wide.

#![allow(clippy::doc_markdown)]

pub mod agents;
pub mod cache_policy;
pub mod commands;
pub mod events;
pub mod history;
pub mod message_handlers;
pub mod session_lifecycle;
pub mod shared;
pub mod state;
pub mod state_parsing;
pub mod tool_calls;
pub mod tooling;
pub mod user_interaction;
