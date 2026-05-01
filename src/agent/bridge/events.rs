//! Thin emit helpers — most modules now push directly onto the
//! caller-provided `Vec<BridgeEvent>`. This module only carries the
//! cross-cutting helpers that mirror upstream's
//! `agent-sdk/src/bridge/events.ts` `failConnection` shape and the
//! `emit_session_update` convenience.

use crate::agent::types::SessionUpdate;
use crate::agent::wire::BridgeEvent;

/// Convenience: push a `SessionUpdate` for `session_id` onto the out
/// buffer. Mirrors `emitSessionUpdate(sessionId, update)` upstream.
pub fn emit_session_update(out: &mut Vec<BridgeEvent>, session_id: &str, update: SessionUpdate) {
    out.push(BridgeEvent::SessionUpdate { session_id: session_id.to_owned(), update });
}
