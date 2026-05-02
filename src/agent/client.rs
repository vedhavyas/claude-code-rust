// Copyright 2025 Simon Peter Rothgang
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::agent::wire::SessionLaunchSettings;

/// Behavioural seam between the TUI and the agent backend.
///
/// Pinning callers to a trait rather than a concrete struct lets
/// alternative backends (forge-sdk in-process, future remote daemons,
/// stub implementations for tests) plug in without changing call sites
/// across `app/*`. The current production implementation is
/// [`crate::agent::forge_sdk_bridge::ForgeSdkBridge`].
///
/// Two method shapes coexist on this trait:
///
/// - **Fire-and-forget commands** — most session-lifecycle methods
///   (`prompt_text`, `cancel`, `set_mode`, …) return
///   `anyhow::Result<()>` and emit results back through the existing
///   [`crate::agent::wire::BridgeEvent`] stream.
/// - **Direct-return accessors** — synchronous reads/writes that
///   return their result directly (`config_dir`, `oauth_credentials`,
///   `settings_documents`, `write_settings_document`,
///   `project_memory_path`) plus one async method (`oauth_usage`,
///   does HTTPS). The `ForgeSdkBridge` impl delegates these to
///   `forge_sdk::*` free functions today; a remote-daemon impl
///   would do RPC over the same trait shape.
///
/// `#[async_trait(?Send)]` matches the
/// `Rc<dyn AgentBridge>` single-threaded App-state model — futures
/// returned by the trait don't need a `Send` bound.
#[async_trait(?Send)]
pub trait AgentBridge {
    fn prompt_text(&self, session_id: String, text: String) -> anyhow::Result<PromptResponse>;

    fn prompt_with_images(
        &self,
        session_id: String,
        text: String,
        images: Vec<crate::app::clipboard_image::ImageAttachment>,
    ) -> anyhow::Result<PromptResponse>;

    fn cancel(&self, session_id: String) -> anyhow::Result<()>;

    fn set_mode(&self, session_id: String, mode: String) -> anyhow::Result<()>;

    fn set_model(&self, session_id: String, model: String) -> anyhow::Result<()>;

    fn generate_session_title(
        &self,
        session_id: String,
        description: String,
    ) -> anyhow::Result<()>;

    fn rename_session(&self, session_id: String, title: String) -> anyhow::Result<()>;

    fn get_status_snapshot(&self, session_id: String) -> anyhow::Result<()>;

    fn get_oauth_credentials_snapshot(&self, session_id: String) -> anyhow::Result<()>;

    fn get_context_usage(&self, session_id: String) -> anyhow::Result<()>;

    fn reload_plugins(&self, session_id: String) -> anyhow::Result<()>;

    fn get_mcp_snapshot(&self, session_id: String) -> anyhow::Result<()>;

    fn respond_to_elicitation(
        &self,
        session_id: String,
        elicitation_request_id: String,
        action: crate::agent::types::ElicitationAction,
        content: Option<serde_json::Value>,
    ) -> anyhow::Result<()>;

    fn reconnect_mcp_server(
        &self,
        session_id: String,
        server_name: String,
    ) -> anyhow::Result<()>;

    fn toggle_mcp_server(
        &self,
        session_id: String,
        server_name: String,
        enabled: bool,
    ) -> anyhow::Result<()>;

    fn set_mcp_servers(
        &self,
        session_id: String,
        servers: std::collections::BTreeMap<String, crate::agent::types::McpServerConfig>,
    ) -> anyhow::Result<()>;

    fn authenticate_mcp_server(
        &self,
        session_id: String,
        server_name: String,
    ) -> anyhow::Result<()>;

    fn clear_mcp_auth(&self, session_id: String, server_name: String) -> anyhow::Result<()>;

    fn submit_mcp_oauth_callback_url(
        &self,
        session_id: String,
        server_name: String,
        callback_url: String,
    ) -> anyhow::Result<()>;

    fn new_session(
        &self,
        cwd: String,
        launch_settings: SessionLaunchSettings,
    ) -> anyhow::Result<()>;

    fn resume_session(
        &self,
        session_id: String,
        launch_settings: SessionLaunchSettings,
    ) -> anyhow::Result<()>;

    fn permission_response(
        &self,
        session_id: String,
        tool_call_id: String,
        outcome: crate::agent::types::PermissionOutcome,
    ) -> anyhow::Result<()>;

    fn question_response(
        &self,
        session_id: String,
        tool_call_id: String,
        outcome: crate::agent::types::QuestionOutcome,
    ) -> anyhow::Result<()>;

    /// Start watching `cwd`'s `.git` machinery for branch changes.
    /// Snapshots flow back via `BridgeEvent::GitContextSnapshot`
    /// (initial state queued before the call returns; subsequent
    /// snapshots only on actual branch change). Calling again with
    /// the same `session_id` aborts and replaces any existing
    /// watcher for that session.
    fn start_git_context_watch(
        &self,
        session_id: String,
        cwd: PathBuf,
    ) -> anyhow::Result<()>;

    /// Stop the git-context watcher for `session_id`. No-op when no
    /// watcher is active. Watchers also stop automatically when the
    /// session closes / the bridge worker shuts down.
    fn stop_git_context_watch(&self, session_id: String) -> anyhow::Result<()>;

    // ---- Direct-return accessors (lifted from forge_sdk::* free fns) ----

    /// Resolve the Claude config directory. Honours
    /// `$CLAUDE_CONFIG_DIR` (when set + non-empty) else falls back
    /// to `$HOME/.claude`.
    fn config_dir(&self) -> PathBuf;

    /// Resolve the project's auto-memory file:
    /// `<config_dir>/projects/<project_key>/memory/MEMORY.md`. The
    /// returned path may not exist on disk; callers decide.
    fn project_memory_path(&self, cwd: &Path) -> PathBuf;

    /// Read OAuth credentials from `<config_dir>/.credentials.json`
    /// or, on macOS, the matching keychain entry. Returns `None`
    /// when no credentials are present.
    fn oauth_credentials(&self) -> Option<forge_sdk::OauthCredentials>;

    /// Read all three settings documents (user, project-local,
    /// preferences) from disk. Each field is `None` when the
    /// underlying file is missing or unreadable.
    fn settings_documents(&self, cwd: &Path) -> forge_sdk::SettingsDocuments;

    /// Atomically write a settings document to the path
    /// [`forge_sdk::SettingsTarget`] resolves to.
    ///
    /// # Errors
    ///
    /// Returns [`forge_sdk::Error`] when the underlying write fails
    /// — see [`forge_sdk::write_settings_document`] for the failure
    /// modes.
    fn write_settings_document(
        &self,
        target: &forge_sdk::SettingsTarget,
        document: &serde_json::Value,
    ) -> Result<(), forge_sdk::Error>;

    /// Fetch the OAuth usage payload from
    /// `api.anthropic.com/api/oauth/usage`. The bearer token is
    /// resolved internally; it never crosses the trait boundary.
    ///
    /// # Errors
    ///
    /// See [`forge_sdk::OauthUsageError`].
    async fn oauth_usage(
        &self,
    ) -> Result<forge_sdk::OauthUsage, forge_sdk::OauthUsageError>;
}

#[derive(Debug, Clone)]
pub struct PromptResponse {
    pub stop_reason: String,
}
