//! Stream-message state parsers: rate-limit / api-retry / fast-mode /
//! runtime-session / settings-parse-error. Mirrors upstream's
//! `agent-sdk/src/bridge/state_parsing.ts`.

use serde_json::{Map, Value};

use crate::agent::types::{
    ApiRetryError, FastModeState, RateLimitStatus, RuntimeSessionState, SessionUpdate,
    SettingsParseErrorUpdate,
};

use super::shared::{number_field, record};

#[must_use]
pub fn parse_fast_mode_state(value: Option<&Value>) -> Option<FastModeState> {
    match value?.as_str()? {
        "off" => Some(FastModeState::Off),
        "cooldown" => Some(FastModeState::Cooldown),
        "on" => Some(FastModeState::On),
        _ => None,
    }
}

#[must_use]
pub fn parse_rate_limit_status(value: Option<&Value>) -> Option<RateLimitStatus> {
    match value?.as_str()? {
        "allowed" => Some(RateLimitStatus::Allowed),
        "allowed_warning" => Some(RateLimitStatus::AllowedWarning),
        "rejected" => Some(RateLimitStatus::Rejected),
        _ => None,
    }
}

#[must_use]
pub fn parse_runtime_session_state(value: Option<&Value>) -> Option<RuntimeSessionState> {
    match value?.as_str()? {
        "idle" => Some(RuntimeSessionState::Idle),
        "running" => Some(RuntimeSessionState::Running),
        "requires_action" => Some(RuntimeSessionState::RequiresAction),
        _ => None,
    }
}

#[must_use]
pub fn parse_api_retry_error(value: Option<&Value>) -> ApiRetryError {
    match value.and_then(Value::as_str) {
        Some("authentication_failed") => ApiRetryError::AuthenticationFailed,
        Some("billing_error") => ApiRetryError::BillingError,
        Some("rate_limit") => ApiRetryError::RateLimit,
        Some("invalid_request") => ApiRetryError::InvalidRequest,
        Some("server_error") => ApiRetryError::ServerError,
        Some("max_output_tokens") => ApiRetryError::MaxOutputTokens,
        _ => ApiRetryError::Unknown,
    }
}

/// Mirrors upstream's `buildRateLimitUpdate(rateLimitInfo)`. Returns
/// `Some(SessionUpdate::RateLimitUpdate { ... })` when the input has a
/// recognised `status`, populating optional fields when present.
#[must_use]
pub fn build_rate_limit_update(rate_limit_info: Option<&Value>) -> Option<SessionUpdate> {
    let info = rate_limit_info.and_then(record)?;
    let status = parse_rate_limit_status(info.get("status"))?;

    Some(SessionUpdate::RateLimitUpdate {
        status,
        resets_at: number_field(info, &["resetsAt"]),
        utilization: number_field(info, &["utilization"]),
        rate_limit_type: info
            .get("rateLimitType")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        overage_status: parse_rate_limit_status(info.get("overageStatus")),
        overage_resets_at: number_field(info, &["overageResetsAt"]),
        overage_disabled_reason: info
            .get("overageDisabledReason")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        is_using_overage: info.get("isUsingOverage").and_then(Value::as_bool),
        surpassed_threshold: number_field(info, &["surpassedThreshold"]),
    })
}

/// Mirrors upstream's `buildApiRetryUpdate(message)`. Returns the
/// `ApiRetryUpdate` `SessionUpdate` when all three required numeric
/// fields parse correctly.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn build_api_retry_update(message: &Map<String, Value>) -> Option<SessionUpdate> {
    let attempt = number_field(message, &["attempt"])? as u64;
    let max_retries = number_field(message, &["max_retries", "maxRetries"])? as u64;
    let retry_delay_ms = number_field(message, &["retry_delay_ms", "retryDelayMs"])? as u64;
    let error_status = number_field(message, &["error_status", "errorStatus"])
        .and_then(|n| u16::try_from(n as i64).ok());
    let error = parse_api_retry_error(message.get("error"));

    Some(SessionUpdate::ApiRetryUpdate {
        attempt,
        max_retries,
        retry_delay_ms,
        error_status,
        error,
    })
}

/// Mirrors `normalizeSettingsParseError(value)`.
#[must_use]
pub fn normalize_settings_parse_error(value: &Value) -> Option<SettingsParseErrorUpdate> {
    let r = record(value)?;
    let message = r.get("message").and_then(Value::as_str)?.trim();
    if message.is_empty() {
        return None;
    }
    let path = r.get("path").and_then(Value::as_str).unwrap_or("").to_owned();
    let file = r
        .get("file")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_owned);
    Some(SettingsParseErrorUpdate { file, path, message: message.to_owned() })
}

/// Accepts either a single record or an array; returns all valid entries.
#[must_use]
pub fn normalize_settings_parse_errors(value: &Value) -> Vec<SettingsParseErrorUpdate> {
    if let Some(arr) = value.as_array() {
        return arr.iter().filter_map(normalize_settings_parse_error).collect();
    }
    normalize_settings_parse_error(value).into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fast_mode_state_parses_known_values() {
        assert_eq!(parse_fast_mode_state(Some(&json!("on"))), Some(FastModeState::On));
        assert_eq!(parse_fast_mode_state(Some(&json!("off"))), Some(FastModeState::Off));
        assert_eq!(parse_fast_mode_state(Some(&json!("cooldown"))), Some(FastModeState::Cooldown));
        assert_eq!(parse_fast_mode_state(Some(&json!("nope"))), None);
        assert_eq!(parse_fast_mode_state(None), None);
    }

    #[test]
    fn rate_limit_status_parses_known_values() {
        assert_eq!(
            parse_rate_limit_status(Some(&json!("allowed_warning"))),
            Some(RateLimitStatus::AllowedWarning),
        );
        assert_eq!(parse_rate_limit_status(Some(&json!("rejected"))), Some(RateLimitStatus::Rejected));
        assert_eq!(parse_rate_limit_status(Some(&json!("nope"))), None);
    }

    #[test]
    fn api_retry_error_falls_back_to_unknown() {
        assert!(matches!(parse_api_retry_error(None), ApiRetryError::Unknown));
        assert!(matches!(
            parse_api_retry_error(Some(&json!("rate_limit"))),
            ApiRetryError::RateLimit
        ));
    }

    #[test]
    fn rate_limit_update_requires_status() {
        let v = json!({"resetsAt": 100.0});
        assert!(build_rate_limit_update(Some(&v)).is_none());

        let v = json!({"status": "allowed", "resetsAt": 100.0, "utilization": 0.5});
        let u = build_rate_limit_update(Some(&v)).expect("update built");
        if let SessionUpdate::RateLimitUpdate { status, resets_at, utilization, .. } = u {
            assert_eq!(status, RateLimitStatus::Allowed);
            assert_eq!(resets_at, Some(100.0));
            assert_eq!(utilization, Some(0.5));
        } else {
            panic!("expected RateLimitUpdate");
        }
    }

    #[test]
    fn api_retry_update_requires_three_numeric_fields() {
        let map: Map<String, Value> =
            serde_json::from_value(json!({"attempt": 1.0, "max_retries": 5.0})).unwrap();
        assert!(build_api_retry_update(&map).is_none());

        let map: Map<String, Value> = serde_json::from_value(json!({
            "attempt": 1.0,
            "max_retries": 5.0,
            "retry_delay_ms": 1000.0,
            "error": "rate_limit",
            "error_status": 429.0,
        }))
        .unwrap();
        let u = build_api_retry_update(&map).expect("update built");
        if let SessionUpdate::ApiRetryUpdate {
            attempt,
            max_retries,
            retry_delay_ms,
            error_status,
            error,
        } = u
        {
            assert_eq!(attempt, 1);
            assert_eq!(max_retries, 5);
            assert_eq!(retry_delay_ms, 1000);
            assert_eq!(error_status, Some(429));
            assert!(matches!(error, ApiRetryError::RateLimit));
        } else {
            panic!("expected ApiRetryUpdate");
        }
    }

    #[test]
    fn settings_parse_error_drops_empty_message() {
        assert!(normalize_settings_parse_error(&json!({"message": "  "})).is_none());
        let entry = normalize_settings_parse_error(&json!({
            "message": "broken json",
            "path": "/etc/x.json",
            "file": "x.json",
        }))
        .expect("entry");
        assert_eq!(entry.message, "broken json");
        assert_eq!(entry.path, "/etc/x.json");
        assert_eq!(entry.file.as_deref(), Some("x.json"));
    }
}
