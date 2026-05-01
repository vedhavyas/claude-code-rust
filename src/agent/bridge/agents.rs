//! Available agents change detection. Mirrors upstream's
//! `agent-sdk/src/bridge/agents.ts`.

use serde_json::Value;

use crate::agent::types::{AvailableAgent, SessionUpdate};
use crate::agent::wire::BridgeEvent;

use super::state::BridgeSession;

/// Mirrors `mapAvailableAgents(value: unknown)` — accepts the raw
/// JSON `agents` array from the initialize response or system/init,
/// dedupes by name (preserving the first non-empty description /
/// model on conflicts), sorts alphabetically.
#[must_use]
pub fn map_available_agents(value: Option<&Value>) -> Vec<AvailableAgent> {
    let Some(arr) = value.and_then(Value::as_array) else { return Vec::new() };

    let mut by_name: std::collections::BTreeMap<String, AvailableAgent> =
        std::collections::BTreeMap::new();

    for entry in arr {
        let Some(record) = entry.as_object() else { continue };
        let Some(name) = record.get("name").and_then(Value::as_str) else { continue };
        let name = name.trim().to_owned();
        if name.is_empty() {
            continue;
        }
        let description = record
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let model = record
            .get("model")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_owned);

        by_name
            .entry(name.clone())
            .and_modify(|existing| {
                if existing.description.trim().is_empty() && !description.trim().is_empty() {
                    existing.description.clone_from(&description);
                }
                if existing.model.is_none() && model.is_some() {
                    existing.model.clone_from(&model);
                }
            })
            .or_insert_with(|| AvailableAgent { name, description, model });
    }

    by_name.into_values().collect()
}

/// Mirrors `mapAvailableAgentsFromNames(value)` — agents listed as
/// plain strings (no descriptions). Used when the CLI emits a names-
/// only list.
#[must_use]
pub fn map_available_agents_from_names(value: Option<&Value>) -> Vec<AvailableAgent> {
    let Some(arr) = value.and_then(Value::as_array) else { return Vec::new() };
    let mut by_name: std::collections::BTreeMap<String, AvailableAgent> =
        std::collections::BTreeMap::new();
    for entry in arr {
        let Some(name) = entry.as_str() else { continue };
        let name = name.trim().to_owned();
        if name.is_empty() {
            continue;
        }
        by_name
            .entry(name.clone())
            .or_insert_with(|| AvailableAgent { name, description: String::new(), model: None });
    }
    by_name.into_values().collect()
}

/// Mirrors `emitAvailableAgentsIfChanged(session, agents)` — diffs by
/// JSON signature and emits a `SessionUpdate::AvailableAgentsUpdate`
/// only when the list actually changed.
pub fn emit_available_agents_if_changed(
    session: &mut BridgeSession,
    agents: Vec<AvailableAgent>,
    out: &mut Vec<BridgeEvent>,
) {
    let signature = serde_json::to_string(&agents).unwrap_or_default();
    if session.last_agents_signature.as_deref() == Some(signature.as_str()) {
        return;
    }
    session.last_agents_signature = Some(signature);
    session.available_agents.clone_from(&agents);
    out.push(BridgeEvent::SessionUpdate {
        session_id: session.session_id.clone(),
        update: SessionUpdate::AvailableAgentsUpdate { agents },
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn map_agents_dedupes_and_sorts() {
        let v = json!([
            { "name": "z", "description": "zd" },
            { "name": "a", "description": "" },
            { "name": "a", "description": "ad", "model": "haiku" },
            { "name": "  ", "description": "drop empty" },
        ]);
        let agents = map_available_agents(Some(&v));
        assert_eq!(agents.len(), 2);
        assert_eq!(agents[0].name, "a");
        assert_eq!(agents[0].description, "ad"); // filled from second entry
        assert_eq!(agents[0].model.as_deref(), Some("haiku"));
        assert_eq!(agents[1].name, "z");
    }

    #[test]
    fn map_agents_from_names_dedupes_and_sorts() {
        let v = json!(["b", "a", "a", " ", "c"]);
        let agents = map_available_agents_from_names(Some(&v));
        assert_eq!(agents.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), vec!["a", "b", "c"]);
    }

    #[test]
    fn emit_agents_only_on_change() {
        let mut session = BridgeSession::new("s".to_owned(), "/tmp".to_owned());
        let agents = vec![AvailableAgent { name: "x".to_owned(), description: "d".to_owned(), model: None }];

        let mut out = Vec::new();
        emit_available_agents_if_changed(&mut session, agents.clone(), &mut out);
        assert_eq!(out.len(), 1);

        let mut out = Vec::new();
        emit_available_agents_if_changed(&mut session, agents.clone(), &mut out);
        assert!(out.is_empty(), "second emit with identical signature should be a no-op");

        let mut out = Vec::new();
        emit_available_agents_if_changed(
            &mut session,
            vec![AvailableAgent { name: "y".to_owned(), description: "d".to_owned(), model: None }],
            &mut out,
        );
        assert_eq!(out.len(), 1);
    }
}
