//! JSON walking helpers. Mirrors upstream's `agent-sdk/src/bridge/shared.ts`.
//!
//! TS uses unknown / Record<string, unknown>; in Rust we walk
//! `serde_json::Value` so these helpers wrap the common patterns.

use serde_json::{Map, Value};

/// Equivalent of upstream's `asRecordOrNull(value)` — returns the
/// inner object map only when `value` is a JSON object (not an array,
/// not a primitive, not null).
#[must_use]
pub fn as_record_or_null(value: Option<&Value>) -> Option<&Map<String, Value>> {
    value?.as_object()
}

/// Same as [`as_record_or_null`] but takes the value by reference
/// directly. Convenience for existing borrows.
#[must_use]
pub fn record(value: &Value) -> Option<&Map<String, Value>> {
    value.as_object()
}

/// Borrow a string field, trimmed — returns `None` when missing or empty.
#[must_use]
pub fn string_field<'a>(record: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    let s = record.get(key)?.as_str()?.trim();
    if s.is_empty() { None } else { Some(s) }
}

/// First number field with a finite f64 value, looked up across the
/// alias keys upstream uses (`max_retries` / `maxRetries`).
#[must_use]
pub fn number_field(record: &Map<String, Value>, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(v) = record.get(*key).and_then(Value::as_f64)
            && v.is_finite()
        {
            return Some(v);
        }
    }
    None
}

/// Convenience: `number_field` cast to u64 (truncating). Returns None if
/// the value is negative or non-finite.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn u64_field(record: &Map<String, Value>, keys: &[&str]) -> Option<u64> {
    number_field(record, keys).and_then(|n| if n >= 0.0 { Some(n as u64) } else { None })
}
