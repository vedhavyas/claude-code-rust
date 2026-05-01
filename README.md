# Claude Code Rust

A native Rust terminal interface for Claude Code, driven by
`forge-sdk` against the same `claude` CLI binary upstream uses.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## About

A native Rust terminal interface for Claude Code built on
[Ratatui](https://ratatui.rs/). The TUI talks directly to the
`claude` CLI binary via [`forge-sdk`](../forge/crates/forge-sdk/),
a Rust wrapper that handles the stream-json wire protocol. No
Node.js bridge, no subprocess JSON shim — single Rust process
end-to-end.

## Requisites

- `claude` CLI binary on PATH (existing Claude Code authentication
  in `~/.claude/config.json` carries over).

## Install

`cargo install` (when published) or build from source:

```bash
cargo build --release
./target/release/claude-rs
```

## Usage

```bash
claude-rs
```

## Why

The stock Claude Code TUI runs on Node.js with React Ink. This
causes real problems:

- **Memory**: 200-400MB baseline vs ~20-50MB for a native binary
- **Startup**: 2-5 seconds vs under 100ms
- **Scrollback**: Broken virtual scrolling that loses history
- **Input latency**: Event queue delays on keystroke handling
- **Copy/paste**: Custom implementation instead of native terminal support

Claude Code Rust fixes all of these by compiling to a single native
binary with direct terminal control via Crossterm and an in-process
`forge-sdk` driving the agent.

## Architecture

Two layers:

**Presentation** (Rust/Ratatui) — single binary with an async
event loop (Tokio) handling keyboard input and agent events
concurrently. Virtual-scrolled chat history with syntax-highlighted
code blocks.

**Agent driver** (`forge-sdk` in-process) — wraps the `claude` CLI
subprocess, handles the stream-json wire protocol, fans out
permissions and questions through a single `AgentBridge` trait. See
`src/agent/forge_sdk_bridge.rs` and `src/agent/forge_sdk_worker.rs`
for the integration.

## Status

This project is pre-1.0 and under active development. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

## Limitations

Startup is still constrained by the upstream Claude Agent SDK runtime that this TUI wraps. The Rust interface itself is fast, but end-to-end readiness can still take noticeable time before a session is fully available. Improving that remains an active area of work.

## License

This project is licensed under the [Apache License 2.0](LICENSE). Apache-2.0 was chosen to keep usage and redistribution straightforward for individual users, downstream packagers, and commercial adopters.

## Disclaimer and Legal Notice

This project is not affiliated with, endorsed by, or supported by Anthropic.

A quick note on where this project stands, since I know people worry about this kind of thing: claude-code-rust is a terminal UI that I wrote from scratch in Rust. It is not a fork, copy or port of the latest Claude Code source leak -- it talks to Anthropic's official [Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/agent-sdk) as a runtime dependency instead, the same way any other third-party tool would. No Anthropic source code was read or used as reference at any point during development.

The project uses your existing Claude Code subscription via the Agent SDK and the Agent SDK's terms allow building on top of it. Other community projects do the same. As far as I can tell, using this project is fine -- but I am a single maintainer, not a lawyer. If anything changes on Anthropic's end, I will update this section and adjust the project accordingly.

This project's source code is licensed under [Apache-2.0](LICENSE). The Agent SDK itself is proprietary and governed by [Anthropic's Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms).

For official Claude documentation, see [https://claude.ai/docs](https://claude.ai/docs).
