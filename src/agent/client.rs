// Copyright 2025 Simon Peter Rothgang
// SPDX-License-Identifier: Apache-2.0

use crate::agent::wire::SessionLaunchSettings;

/// Behavioural seam between the TUI and the agent backend.
///
/// Pinning callers to a trait rather than a concrete struct lets
/// alternative backends (forge-sdk in-process, future remote daemons,
/// stub implementations for tests) plug in without changing call sites
/// across `app/*`. The current production implementation is
/// [`crate::agent::forge_sdk_bridge::ForgeSdkBridge`].
///
/// All methods are fire-and-forget — outcomes flow back through the
/// existing [`crate::agent::wire::BridgeEvent`] stream rather than the
/// trait return value.
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
}

#[derive(Debug, Clone)]
pub struct PromptResponse {
    pub stop_reason: String,
}
