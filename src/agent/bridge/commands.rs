//! Slash command + mode-state helpers. Mirrors upstream's
//! `agent-sdk/src/bridge/commands.ts`.
//!
//! Stage 1 only ports the mode-state builder + supported-modes
//! computation; the `set_mode` / `set_model` command-side handlers
//! land in Stage 4 alongside the live emit-after-set follow-ups.

use crate::agent::types::{ModeInfo, ModeState};

use super::state::{BridgeSession, PermissionMode};

const BASE_SUPPORTED_MODE_IDS: [PermissionMode; 4] = [
    PermissionMode::Default,
    PermissionMode::AcceptEdits,
    PermissionMode::Plan,
    PermissionMode::DontAsk,
];

fn current_model_supports_auto_mode(session: &BridgeSession) -> bool {
    session.current_model.as_ref().is_some_and(|m| m.supports_auto_mode == Some(true))
}

fn unique_mode_ids(modes: Vec<PermissionMode>) -> Vec<PermissionMode> {
    // Mirror upstream's MODE_OPTIONS ordering: default, acceptEdits,
    // plan, dontAsk, auto, bypassPermissions. Filter to only present
    // ids while preserving canonical order.
    const CANONICAL_ORDER: [PermissionMode; 6] = [
        PermissionMode::Default,
        PermissionMode::AcceptEdits,
        PermissionMode::Plan,
        PermissionMode::DontAsk,
        PermissionMode::Auto,
        PermissionMode::BypassPermissions,
    ];
    let mut seen: std::collections::HashSet<PermissionMode> =
        std::collections::HashSet::with_capacity(modes.len());
    for m in modes {
        seen.insert(m);
    }
    CANONICAL_ORDER.into_iter().filter(|m| seen.contains(m)).collect()
}

fn computed_supported_mode_ids(session: &BridgeSession) -> Vec<PermissionMode> {
    let mut supported: Vec<PermissionMode> = BASE_SUPPORTED_MODE_IDS.to_vec();
    if current_model_supports_auto_mode(session) {
        supported.push(PermissionMode::Auto);
    }
    if session.supports_bypass_permissions_mode {
        supported.push(PermissionMode::BypassPermissions);
    }
    if let Some(mode) = session.mode {
        supported.push(mode);
    }
    unique_mode_ids(supported)
}

/// Mirrors `refreshSupportedModesForSession(session)`. Recomputes
/// `session.supported_mode_ids`, filtered by runtime-unavailable list
/// (but keeping the current mode if it's still set).
pub fn refresh_supported_modes_for_session(session: &mut BridgeSession) {
    let computed = computed_supported_mode_ids(session);
    let current = session.mode;
    let unavailable = session.runtime_unavailable_mode_ids.clone();
    session.supported_mode_ids = computed
        .into_iter()
        .filter(|m| current == Some(*m) || !unavailable.contains(m))
        .collect();
}

fn mode_info_for_id(mode: PermissionMode) -> ModeInfo {
    ModeInfo {
        id: mode.as_wire().to_owned(),
        name: mode.display_name().to_owned(),
        description: None,
    }
}

fn available_modes_for_session(session: &BridgeSession) -> Vec<ModeInfo> {
    session.supported_mode_ids.iter().copied().map(mode_info_for_id).collect()
}

/// Mirrors `buildModeState(session, mode)`. Pure builder used both at
/// connect time and after `set_permission_mode`.
#[must_use]
pub fn build_mode_state(session: &BridgeSession, mode: PermissionMode) -> ModeState {
    ModeState {
        current_mode_id: mode.as_wire().to_owned(),
        current_mode_name: mode.display_name().to_owned(),
        available_modes: available_modes_for_session(session),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::types::CurrentModel;

    fn fresh() -> BridgeSession {
        BridgeSession::new("s".to_owned(), "/tmp".to_owned())
    }

    #[test]
    fn base_supported_modes_default_to_four() {
        let mut session = fresh();
        refresh_supported_modes_for_session(&mut session);
        assert_eq!(
            session.supported_mode_ids,
            vec![
                PermissionMode::Default,
                PermissionMode::AcceptEdits,
                PermissionMode::Plan,
                PermissionMode::DontAsk
            ]
        );
    }

    #[test]
    fn auto_mode_appears_when_current_model_supports_it() {
        let mut session = fresh();
        session.current_model = Some(CurrentModel {
            requested_id: None,
            resolved_id: "x".to_owned(),
            display_name_short: "X".to_owned(),
            display_name_long: "X".to_owned(),
            catalog_id: None,
            supports_effort: false,
            supported_effort_levels: Vec::new(),
            supports_fast_mode: None,
            supports_auto_mode: Some(true),
            supports_adaptive_thinking: None,
            is_authoritative: true,
        });
        refresh_supported_modes_for_session(&mut session);
        assert!(session.supported_mode_ids.contains(&PermissionMode::Auto));
    }

    #[test]
    fn bypass_appears_when_session_supports_it() {
        let mut session = fresh();
        session.supports_bypass_permissions_mode = true;
        refresh_supported_modes_for_session(&mut session);
        assert!(session.supported_mode_ids.contains(&PermissionMode::BypassPermissions));
    }

    #[test]
    fn current_mode_survives_runtime_unavailable_filter() {
        let mut session = fresh();
        session.mode = Some(PermissionMode::Plan);
        session.runtime_unavailable_mode_ids = vec![PermissionMode::Plan];
        refresh_supported_modes_for_session(&mut session);
        assert!(session.supported_mode_ids.contains(&PermissionMode::Plan));
    }

    #[test]
    fn build_mode_state_uses_session_supported_list() {
        let mut session = fresh();
        refresh_supported_modes_for_session(&mut session);
        let state = build_mode_state(&session, PermissionMode::AcceptEdits);
        assert_eq!(state.current_mode_id, "acceptEdits");
        assert_eq!(state.current_mode_name, "Accept Edits");
        assert_eq!(state.available_modes.len(), 4);
        assert_eq!(state.available_modes[0].id, "default");
    }
}
