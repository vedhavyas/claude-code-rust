//! Per-session bridge translation state. Mirrors upstream's
//! `SessionState` struct in `agent-sdk/src/bridge/session_lifecycle.ts`.
//!
//! Today the worker is single-session; the wrapping `BridgeSessionStore`
//! is multi-session-ready so a future tab strip doesn't need a state
//! rewrite.

use std::collections::HashMap;
use std::time::Instant;

use crate::agent::types::{
    AvailableAgent, AvailableCommand, AvailableModel, CurrentModel, FastModeState, ModeState,
    SessionUpdate, ToolCall,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectEventKind {
    Connected,
    SessionReplaced,
}

/// Mirrors upstream's permission-mode enum used internally by the
/// bridge — distinct from the wire-string `current_mode_id` shipped on
/// `ModeState`. Used to track which modes are supported / runtime-
/// unavailable per session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PermissionMode {
    Default,
    AcceptEdits,
    Plan,
    DontAsk,
    Auto,
    BypassPermissions,
}

impl PermissionMode {
    #[must_use]
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AcceptEdits => "acceptEdits",
            Self::Plan => "plan",
            Self::DontAsk => "dontAsk",
            Self::Auto => "auto",
            Self::BypassPermissions => "bypassPermissions",
        }
    }

    #[must_use]
    pub fn from_wire(s: &str) -> Option<Self> {
        Some(match s {
            "default" | "ask" => Self::Default,
            "acceptEdits" | "accept_edits" => Self::AcceptEdits,
            "plan" => Self::Plan,
            "dontAsk" | "dont_ask" | "deny" => Self::DontAsk,
            "auto" => Self::Auto,
            "bypassPermissions" | "bypass_permissions" => Self::BypassPermissions,
            _ => return None,
        })
    }

    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::AcceptEdits => "Accept Edits",
            Self::Plan => "Plan",
            Self::DontAsk => "Don't Ask",
            Self::Auto => "Auto",
            Self::BypassPermissions => "Bypass Permissions",
        }
    }
}

#[derive(Debug)]
pub struct BridgeSession {
    // Identity / connect
    pub session_id: String,
    pub cwd: String,
    pub connected: bool,
    pub connect_event: ConnectEventKind,

    // Model resolution cache (mirrors `session.model`,
    // `session.requestedModelId`, `session.resolvedRuntimeModelId`,
    // `session.currentModel`, `session.availableModels` upstream).
    pub model_id: String,
    pub requested_model_id: Option<String>,
    pub resolved_runtime_model_id: Option<String>,
    pub current_model: Option<CurrentModel>,
    pub available_models: Vec<AvailableModel>,

    // Mode resolution
    pub mode: Option<PermissionMode>,
    pub supported_mode_ids: Vec<PermissionMode>,
    pub runtime_unavailable_mode_ids: Vec<PermissionMode>,
    pub supports_bypass_permissions_mode: bool,
    pub mode_state: Option<ModeState>,

    // Last-seen fast mode (for change detection on emit)
    pub fast_mode_state: FastModeState,

    // Slash commands + agents catalogue (for change-on-emit)
    pub available_commands: Vec<AvailableCommand>,
    pub available_agents: Vec<AvailableAgent>,
    pub last_agents_signature: Option<String>,

    // Tool call store + cross-message wiring
    pub tool_calls: HashMap<String, ToolCall>,
    pub task_tool_use_ids: HashMap<String, String>, // task_id -> tool_use_id

    // MCP cooldowns (per server name)
    pub mcp_status_revalidated_at: HashMap<String, Instant>,

    // Auth — emit AuthRequired at most once per session
    pub auth_hint_sent: bool,

    // Last assistant error subtype — survives across messages so a
    // subsequent Result can classify correctly.
    pub last_assistant_error: Option<String>,

    // Resume history collected during connect handshake to attach to
    // the first Connected event (None for fresh sessions).
    pub resume_updates: Option<Vec<SessionUpdate>>,
}

impl BridgeSession {
    #[must_use]
    pub fn new(session_id: String, cwd: String) -> Self {
        Self {
            session_id,
            cwd,
            connected: false,
            connect_event: ConnectEventKind::Connected,
            model_id: String::new(),
            requested_model_id: None,
            resolved_runtime_model_id: None,
            current_model: None,
            available_models: Vec::new(),
            mode: None,
            supported_mode_ids: Vec::new(),
            runtime_unavailable_mode_ids: Vec::new(),
            supports_bypass_permissions_mode: false,
            mode_state: None,
            fast_mode_state: FastModeState::Off,
            available_commands: Vec::new(),
            available_agents: Vec::new(),
            last_agents_signature: None,
            tool_calls: HashMap::new(),
            task_tool_use_ids: HashMap::new(),
            mcp_status_revalidated_at: HashMap::new(),
            auth_hint_sent: false,
            last_assistant_error: None,
            resume_updates: None,
        }
    }
}

#[derive(Debug, Default)]
pub struct BridgeSessionStore {
    sessions: HashMap<String, BridgeSession>,
}

impl BridgeSessionStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, session: BridgeSession) {
        self.sessions.insert(session.session_id.clone(), session);
    }

    #[must_use]
    pub fn get_mut(&mut self, session_id: &str) -> Option<&mut BridgeSession> {
        self.sessions.get_mut(session_id)
    }

    #[must_use]
    pub fn get(&self, session_id: &str) -> Option<&BridgeSession> {
        self.sessions.get(session_id)
    }

    pub fn remove(&mut self, session_id: &str) -> Option<BridgeSession> {
        self.sessions.remove(session_id)
    }

    /// Convenience for the common single-session case.
    #[must_use]
    pub fn first_mut(&mut self) -> Option<&mut BridgeSession> {
        self.sessions.values_mut().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_mode_round_trips_through_wire() {
        for mode in [
            PermissionMode::Default,
            PermissionMode::AcceptEdits,
            PermissionMode::Plan,
            PermissionMode::DontAsk,
            PermissionMode::Auto,
            PermissionMode::BypassPermissions,
        ] {
            assert_eq!(PermissionMode::from_wire(mode.as_wire()), Some(mode));
        }
    }

    #[test]
    fn permission_mode_aliases() {
        assert_eq!(PermissionMode::from_wire("ask"), Some(PermissionMode::Default));
        assert_eq!(PermissionMode::from_wire("accept_edits"), Some(PermissionMode::AcceptEdits));
        assert_eq!(PermissionMode::from_wire("dont_ask"), Some(PermissionMode::DontAsk));
        assert_eq!(PermissionMode::from_wire("deny"), Some(PermissionMode::DontAsk));
        assert_eq!(PermissionMode::from_wire("bypass_permissions"), Some(PermissionMode::BypassPermissions));
    }

    #[test]
    fn store_insert_get_mut_remove() {
        let mut store = BridgeSessionStore::new();
        store.insert(BridgeSession::new("s1".to_owned(), "/tmp".to_owned()));
        assert!(store.get("s1").is_some());
        store.get_mut("s1").unwrap().connected = true;
        assert!(store.get("s1").unwrap().connected);
        assert!(store.remove("s1").is_some());
        assert!(store.get("s1").is_none());
    }
}
