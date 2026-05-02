//! `AgentBridge` impl backed by `forge-sdk` running in-process.
//!
//! Mirrors the role of [`super::client::AgentConnection`] (which forwards
//! commands as NDJSON to the Node bridge subprocess in `agent-sdk/`) but
//! drives a `forge_sdk::Client` directly from Rust. No subprocess fork,
//! no JSON serialization round-trip, no Node runtime.
//!
//! ## Architecture
//!
//! ```text
//!     TUI                 ForgeSdkBridge       worker task         forge_sdk::Client
//!      |  trait method       |                      |                   |
//!      |-------------------->|                      |                   |
//!      |                     | ForgeSdkCommand      |                   |
//!      |                     |--------------------->|                   |
//!      |                     |                      | client.method()   |
//!      |                     |                      |------------------>|
//!      |                     |                      |                   |
//!      |   BridgeEvent       |                      | client.next_event |
//!      |<-- (via event_tx) --+----------------------+<------------------|
//! ```
//!
//! The trait impl is fire-and-forget: each method sends a
//! [`ForgeSdkCommand`] on an mpsc and returns immediately. The worker
//! task drains the channel, calls async `forge_sdk::Client` methods,
//! and emits results back to the TUI via the existing `BridgeEvent`
//! shape so [`crate::app::connect::event_dispatch`] doesn't need to
//! know which backend is running.
//!
//! Status: scaffolding. The trait impl ships in this commit; the
//! command -> SDK translation worker and the SDK message ->
//! `BridgeEvent` translator land in follow-up commits.

#![allow(dead_code)] // worker + dispatch land in follow-up commits

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::agent::client::{AgentBridge, PromptResponse};
use crate::agent::types::{
    ElicitationAction, McpServerConfig, PermissionOutcome, QuestionOutcome,
};
use crate::agent::wire::SessionLaunchSettings;

/// Internal command shape pushed onto the worker's queue. Mirrors the
/// trait surface of [`AgentBridge`] minus the (sync) plumbing — the
/// worker translates each variant into one or more
/// `forge_sdk::Client` async calls.
#[derive(Debug, Clone, PartialEq)]
pub enum ForgeSdkCommand {
    Prompt {
        session_id: String,
        chunks: Vec<crate::agent::types::PromptChunk>,
    },
    Cancel {
        session_id: String,
    },
    SetMode {
        session_id: String,
        mode: String,
    },
    SetModel {
        session_id: String,
        model: String,
    },
    GenerateSessionTitle {
        session_id: String,
        description: String,
    },
    RenameSession {
        session_id: String,
        title: String,
    },
    GetStatusSnapshot {
        session_id: String,
    },
    GetOauthCredentialsSnapshot {
        session_id: String,
    },
    StartGitContextWatch {
        session_id: String,
        cwd: PathBuf,
    },
    StopGitContextWatch {
        session_id: String,
    },
    GetContextUsage {
        session_id: String,
    },
    ReloadPlugins {
        session_id: String,
    },
    GetMcpSnapshot {
        session_id: String,
    },
    RespondToElicitation {
        session_id: String,
        elicitation_request_id: String,
        action: ElicitationAction,
        content: Option<Value>,
    },
    ReconnectMcpServer {
        session_id: String,
        server_name: String,
    },
    ToggleMcpServer {
        session_id: String,
        server_name: String,
        enabled: bool,
    },
    SetMcpServers {
        session_id: String,
        servers: BTreeMap<String, McpServerConfig>,
    },
    AuthenticateMcpServer {
        session_id: String,
        server_name: String,
    },
    ClearMcpAuth {
        session_id: String,
        server_name: String,
    },
    SubmitMcpOauthCallbackUrl {
        session_id: String,
        server_name: String,
        callback_url: String,
    },
    NewSession {
        cwd: String,
        launch_settings: SessionLaunchSettings,
    },
    ResumeSession {
        session_id: String,
        launch_settings: SessionLaunchSettings,
    },
    PermissionResponse {
        session_id: String,
        tool_call_id: String,
        outcome: PermissionOutcome,
    },
    QuestionResponse {
        session_id: String,
        tool_call_id: String,
        outcome: QuestionOutcome,
    },
}

/// Forge-SDK-backed implementation of [`AgentBridge`].
///
/// Constructed by [`spawn_forge_sdk_bridge`] (lands in a follow-up
/// commit alongside the worker that drains [`ForgeSdkCommand`]s into
/// `forge_sdk::Client` calls). The TUI side holds an
/// `Rc<dyn AgentBridge>` and is unaware which backend handles the
/// commands.
#[derive(Clone)]
pub struct ForgeSdkBridge {
    command_tx: mpsc::UnboundedSender<ForgeSdkCommand>,
}

impl ForgeSdkBridge {
    /// Construct a bridge handle around an existing command channel.
    /// The receiving end is consumed by the worker task spawned in
    /// [`spawn_forge_sdk_bridge`].
    #[must_use]
    pub fn new(command_tx: mpsc::UnboundedSender<ForgeSdkCommand>) -> Self {
        Self { command_tx }
    }

    fn send(&self, cmd: ForgeSdkCommand) -> anyhow::Result<()> {
        self.command_tx
            .send(cmd)
            .map_err(|_| anyhow::anyhow!("forge-sdk bridge worker has exited"))
    }
}

#[async_trait(?Send)]
impl AgentBridge for ForgeSdkBridge {
    fn prompt_text(
        &self,
        session_id: String,
        text: String,
    ) -> anyhow::Result<PromptResponse> {
        self.prompt_with_images(session_id, text, Vec::new())
    }

    fn prompt_with_images(
        &self,
        session_id: String,
        text: String,
        images: Vec<crate::app::clipboard_image::ImageAttachment>,
    ) -> anyhow::Result<PromptResponse> {
        let mut chunks = Vec::with_capacity(1 + images.len());
        for img in images {
            if let Err(reason) =
                crate::app::clipboard_image::validate_image(&img.data, &img.mime_type)
            {
                tracing::warn!(
                    target: crate::logging::targets::APP_INPUT,
                    "forge-sdk bridge: skipping invalid image: {reason}"
                );
                continue;
            }
            chunks.push(crate::agent::types::PromptChunk {
                kind: "image".to_owned(),
                value: serde_json::json!({
                    "data": img.data,
                    "mime_type": img.mime_type,
                }),
            });
        }
        chunks.push(crate::agent::types::PromptChunk {
            kind: "text".to_owned(),
            value: Value::String(text),
        });
        self.send(ForgeSdkCommand::Prompt { session_id, chunks })?;
        Ok(PromptResponse { stop_reason: "end_turn".to_owned() })
    }

    fn cancel(&self, session_id: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::Cancel { session_id })
    }

    fn set_mode(&self, session_id: String, mode: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::SetMode { session_id, mode })
    }

    fn set_model(&self, session_id: String, model: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::SetModel { session_id, model })
    }

    fn generate_session_title(
        &self,
        session_id: String,
        description: String,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::GenerateSessionTitle { session_id, description })
    }

    fn rename_session(&self, session_id: String, title: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::RenameSession { session_id, title })
    }

    fn get_status_snapshot(&self, session_id: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::GetStatusSnapshot { session_id })
    }

    fn get_oauth_credentials_snapshot(&self, session_id: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::GetOauthCredentialsSnapshot { session_id })
    }

    fn start_git_context_watch(
        &self,
        session_id: String,
        cwd: PathBuf,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::StartGitContextWatch { session_id, cwd })
    }

    fn stop_git_context_watch(&self, session_id: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::StopGitContextWatch { session_id })
    }

    fn get_context_usage(&self, session_id: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::GetContextUsage { session_id })
    }

    fn reload_plugins(&self, session_id: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::ReloadPlugins { session_id })
    }

    fn get_mcp_snapshot(&self, session_id: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::GetMcpSnapshot { session_id })
    }

    fn respond_to_elicitation(
        &self,
        session_id: String,
        elicitation_request_id: String,
        action: ElicitationAction,
        content: Option<Value>,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::RespondToElicitation {
            session_id,
            elicitation_request_id,
            action,
            content,
        })
    }

    fn reconnect_mcp_server(
        &self,
        session_id: String,
        server_name: String,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::ReconnectMcpServer { session_id, server_name })
    }

    fn toggle_mcp_server(
        &self,
        session_id: String,
        server_name: String,
        enabled: bool,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::ToggleMcpServer { session_id, server_name, enabled })
    }

    fn set_mcp_servers(
        &self,
        session_id: String,
        servers: BTreeMap<String, McpServerConfig>,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::SetMcpServers { session_id, servers })
    }

    fn authenticate_mcp_server(
        &self,
        session_id: String,
        server_name: String,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::AuthenticateMcpServer { session_id, server_name })
    }

    fn clear_mcp_auth(&self, session_id: String, server_name: String) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::ClearMcpAuth { session_id, server_name })
    }

    fn submit_mcp_oauth_callback_url(
        &self,
        session_id: String,
        server_name: String,
        callback_url: String,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::SubmitMcpOauthCallbackUrl {
            session_id,
            server_name,
            callback_url,
        })
    }

    fn new_session(
        &self,
        cwd: String,
        launch_settings: SessionLaunchSettings,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::NewSession { cwd, launch_settings })
    }

    fn resume_session(
        &self,
        session_id: String,
        launch_settings: SessionLaunchSettings,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::ResumeSession { session_id, launch_settings })
    }

    fn permission_response(
        &self,
        session_id: String,
        tool_call_id: String,
        outcome: PermissionOutcome,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::PermissionResponse { session_id, tool_call_id, outcome })
    }

    fn question_response(
        &self,
        session_id: String,
        tool_call_id: String,
        outcome: QuestionOutcome,
    ) -> anyhow::Result<()> {
        self.send(ForgeSdkCommand::QuestionResponse { session_id, tool_call_id, outcome })
    }

    // ---- Direct-return accessors (delegate to forge_sdk::*) ----

    fn config_dir(&self) -> PathBuf {
        forge_sdk::claude_config_dir()
    }

    fn project_memory_path(&self, cwd: &Path) -> PathBuf {
        forge_sdk::project_memory_path(cwd)
    }

    fn oauth_credentials(&self) -> Option<forge_sdk::OauthCredentials> {
        forge_sdk::oauth_credentials()
    }

    fn settings_documents(&self, cwd: &Path) -> forge_sdk::SettingsDocuments {
        forge_sdk::settings_documents(cwd)
    }

    fn write_settings_document(
        &self,
        target: &forge_sdk::SettingsTarget,
        document: &Value,
    ) -> Result<(), forge_sdk::Error> {
        forge_sdk::write_settings_document(target, document)
    }

    async fn oauth_usage(
        &self,
    ) -> Result<forge_sdk::OauthUsage, forge_sdk::OauthUsageError> {
        forge_sdk::oauth_usage().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_routes_command_through_channel() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let bridge = ForgeSdkBridge::new(tx);

        bridge
            .cancel("session-1".to_owned())
            .expect("cancel command queued");

        let cmd = rx.try_recv().expect("worker receives command");
        match cmd {
            ForgeSdkCommand::Cancel { session_id } => assert_eq!(session_id, "session-1"),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn closed_channel_surfaces_error() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let bridge = ForgeSdkBridge::new(tx);

        let err = bridge.cancel("session-1".to_owned()).unwrap_err();
        assert!(err.to_string().contains("worker has exited"));
    }

    #[test]
    fn prompt_text_packages_a_text_chunk() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let bridge = ForgeSdkBridge::new(tx);

        bridge
            .prompt_text("session-1".to_owned(), "hello".to_owned())
            .expect("prompt queued");

        match rx.try_recv().expect("command") {
            ForgeSdkCommand::Prompt { session_id, chunks } => {
                assert_eq!(session_id, "session-1");
                assert_eq!(chunks.len(), 1);
                assert_eq!(chunks[0].kind, "text");
                assert_eq!(chunks[0].value, Value::String("hello".to_owned()));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }
}
