// Copyright 2025 Simon Peter Rothgang
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Map, Value};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::{
    DefaultPermissionMode, OutputStyle, PreferredNotifChannel, SettingId, SettingKind, SettingSpec,
    setting_spec,
};
use crate::agent::model::EffortLevel;

const SETTINGS_FILENAME: &str = "settings.json";
const LOCAL_SETTINGS_FILENAME: &str = "settings.local.json";
const PREFERENCES_FILENAME: &str = ".claude.json";
const CLAUDE_DIR: &str = ".claude";
const CLAUDE_CODE_DISABLE_1M_CONTEXT_ENV: &str = "CLAUDE_CODE_DISABLE_1M_CONTEXT";
const ANTHROPIC_DEFAULT_OPUS_MODEL_ENV: &str = "ANTHROPIC_DEFAULT_OPUS_MODEL";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistedSettingValue {
    Missing,
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsPaths {
    pub settings: PathBuf,
    pub local_settings: PathBuf,
    pub preferences: PathBuf,
}

pub struct LoadedSettingsDocuments {
    pub paths: SettingsPaths,
    pub settings_document: Value,
    pub local_settings_document: Value,
    pub preferences_document: Value,
}

pub fn load(
    home_override: Option<&Path>,
    project_root_override: Option<&Path>,
    bridge: Option<&dyn crate::agent::client::AgentBridge>,
) -> Result<LoadedSettingsDocuments, String> {
    let paths = resolve_paths(home_override, project_root_override, bridge)?;

    // Production path delegates to the AgentBridge so the same
    // `$CLAUDE_CONFIG_DIR`-respecting reader is used everywhere. Test
    // fixtures pass home_override / project_root_override (and `None`
    // for `bridge`) and bypass the bridge — env vars are
    // process-global and would race across parallel test runs.
    let (settings_document, local_settings_document, preferences_document) = match bridge {
        Some(bridge) if home_override.is_none() && project_root_override.is_none() => {
            let cwd = std::env::current_dir()
                .map_err(|err| format!("Failed to resolve current directory: {err}"))?;
            let docs = bridge.settings_documents(&cwd);
            (
                docs.user.unwrap_or_else(empty_object),
                docs.project_local.unwrap_or_else(empty_object),
                docs.preferences.unwrap_or_else(empty_object),
            )
        }
        _ => (
            read_json_or_empty(&paths.settings),
            read_json_or_empty(&paths.local_settings),
            read_json_or_empty(&paths.preferences),
        ),
    };

    Ok(LoadedSettingsDocuments {
        paths,
        settings_document,
        local_settings_document,
        preferences_document,
    })
}

pub fn save(path: &Path, document: &Value) -> Result<(), String> {
    let parent = path.parent().ok_or_else(|| "Settings path has no parent directory".to_owned())?;
    std::fs::create_dir_all(parent)
        .map_err(|err| format!("Failed to create settings directory: {err}"))?;

    let normalized = normalized_root(document);
    let temp_path = unique_temp_path(parent, path.file_name().and_then(std::ffi::OsStr::to_str));
    let mut temp = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .map_err(|err| format!("Failed to create settings temp file: {err}"))?;
    serde_json::to_writer_pretty(&mut temp, &normalized)
        .map_err(|err| format!("Failed to serialize settings: {err}"))?;
    temp.write_all(b"\n").map_err(|err| format!("Failed to finalize settings file: {err}"))?;
    temp.flush().map_err(|err| format!("Failed to flush settings file: {err}"))?;
    temp.sync_all().map_err(|err| format!("Failed to sync settings file: {err}"))?;
    drop(temp);
    std::fs::rename(&temp_path, path)
        .map_err(|err| format!("Failed to move settings file into place: {err}"))?;
    Ok(())
}

pub fn read_persisted_setting(
    document: &Value,
    spec: &SettingSpec,
) -> Result<PersistedSettingValue, ()> {
    let Some(value) = read_json_path(document, spec.json_path) else {
        return Ok(PersistedSettingValue::Missing);
    };

    match spec.kind {
        SettingKind::Bool => match value {
            Value::Bool(flag) => Ok(PersistedSettingValue::Bool(*flag)),
            _ => Err(()),
        },
        SettingKind::Enum | SettingKind::DynamicEnum | SettingKind::Text => match value {
            Value::String(text) => Ok(PersistedSettingValue::String(text.clone())),
            _ => Err(()),
        },
    }
}

pub fn write_persisted_setting(
    document: &mut Value,
    spec: &SettingSpec,
    value: PersistedSettingValue,
) {
    match value {
        PersistedSettingValue::Missing => remove_json_path(document, spec.json_path),
        PersistedSettingValue::Bool(flag) => {
            set_json_path(document, spec.json_path, Value::Bool(flag));
        }
        PersistedSettingValue::String(text) => {
            set_json_path(document, spec.json_path, Value::String(text));
        }
    }
}

pub fn fast_mode(document: &Value) -> Result<bool, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::FastMode))? {
        PersistedSettingValue::Missing => Ok(false),
        PersistedSettingValue::Bool(value) => Ok(value),
        PersistedSettingValue::String(_) => Err(()),
    }
}

pub fn set_fast_mode(document: &mut Value, enabled: bool) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::FastMode),
        PersistedSettingValue::Bool(enabled),
    );
}

pub fn always_thinking_enabled(document: &Value) -> Result<bool, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::AlwaysThinking))? {
        PersistedSettingValue::Missing => Ok(false),
        PersistedSettingValue::Bool(value) => Ok(value),
        PersistedSettingValue::String(_) => Err(()),
    }
}

pub fn set_always_thinking_enabled(document: &mut Value, enabled: bool) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::AlwaysThinking),
        PersistedSettingValue::Bool(enabled),
    );
}

pub fn thinking_effort_level(document: &Value) -> Result<EffortLevel, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::ThinkingEffort))? {
        PersistedSettingValue::Missing => Ok(EffortLevel::Medium),
        PersistedSettingValue::Bool(_) => Err(()),
        PersistedSettingValue::String(value) => EffortLevel::from_stored(&value).ok_or(()),
    }
}

pub fn set_thinking_effort_level(document: &mut Value, level: EffortLevel) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::ThinkingEffort),
        PersistedSettingValue::String(level.as_stored().to_owned()),
    );
}

pub fn spinner_tips_enabled(document: &Value) -> Result<bool, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::ShowTips))? {
        PersistedSettingValue::Missing => Ok(true),
        PersistedSettingValue::Bool(value) => Ok(value),
        PersistedSettingValue::String(_) => Err(()),
    }
}

pub fn set_spinner_tips_enabled(document: &mut Value, enabled: bool) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::ShowTips),
        PersistedSettingValue::Bool(enabled),
    );
}

pub fn terminal_progress_bar_enabled(document: &Value) -> Result<bool, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::TerminalProgressBar))? {
        PersistedSettingValue::Missing => Ok(true),
        PersistedSettingValue::Bool(value) => Ok(value),
        PersistedSettingValue::String(_) => Err(()),
    }
}

pub fn set_terminal_progress_bar_enabled(document: &mut Value, enabled: bool) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::TerminalProgressBar),
        PersistedSettingValue::Bool(enabled),
    );
}

pub fn prefers_reduced_motion(document: &Value) -> Result<bool, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::ReduceMotion))? {
        PersistedSettingValue::Missing => Ok(false),
        PersistedSettingValue::Bool(value) => Ok(value),
        PersistedSettingValue::String(_) => Err(()),
    }
}

pub fn set_prefers_reduced_motion(document: &mut Value, enabled: bool) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::ReduceMotion),
        PersistedSettingValue::Bool(enabled),
    );
}

pub fn output_style(document: &Value) -> Result<OutputStyle, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::OutputStyle))? {
        PersistedSettingValue::Missing => Ok(OutputStyle::Default),
        PersistedSettingValue::Bool(_) => Err(()),
        PersistedSettingValue::String(value) => OutputStyle::from_stored(&value).ok_or(()),
    }
}

pub fn set_output_style(document: &mut Value, style: OutputStyle) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::OutputStyle),
        PersistedSettingValue::String(style.as_stored().to_owned()),
    );
}

pub fn set_model(document: &mut Value, model: Option<&str>) {
    let value = model.map_or(PersistedSettingValue::Missing, |model| {
        PersistedSettingValue::String(model.to_owned())
    });
    write_persisted_setting(document, setting_spec(SettingId::Model), value);
}

#[cfg(test)]
pub fn model(document: &Value) -> Result<Option<String>, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::Model))? {
        PersistedSettingValue::Missing => Ok(None),
        PersistedSettingValue::Bool(_) => Err(()),
        PersistedSettingValue::String(value) => Ok(Some(value)),
    }
}

#[cfg(test)]
pub fn default_permission_mode(document: &Value) -> Result<DefaultPermissionMode, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::DefaultPermissionMode))? {
        PersistedSettingValue::Missing => Ok(DefaultPermissionMode::Default),
        PersistedSettingValue::Bool(_) => Err(()),
        PersistedSettingValue::String(value) => {
            DefaultPermissionMode::from_stored(&value).ok_or(())
        }
    }
}

pub fn set_default_permission_mode(document: &mut Value, mode: DefaultPermissionMode) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::DefaultPermissionMode),
        PersistedSettingValue::String(mode.as_stored().to_owned()),
    );
}

pub fn language(document: &Value) -> Result<Option<String>, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::Language))? {
        PersistedSettingValue::Missing => Ok(None),
        PersistedSettingValue::Bool(_) => Err(()),
        PersistedSettingValue::String(value) => Ok(Some(value)),
    }
}

pub fn set_language(document: &mut Value, value: Option<&str>) {
    let persisted = value
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map_or(PersistedSettingValue::Missing, |text| {
            PersistedSettingValue::String(text.to_owned())
        });
    write_persisted_setting(document, setting_spec(SettingId::Language), persisted);
}

pub fn disable_1m_context(document: &Value) -> Result<bool, ()> {
    match read_json_path(document, &["env", CLAUDE_CODE_DISABLE_1M_CONTEXT_ENV]) {
        None => Ok(false),
        Some(Value::String(value)) => Ok(value == "1"),
        Some(_) => Err(()),
    }
}

pub fn set_disable_1m_context(document: &mut Value, disabled: bool) {
    if disabled {
        set_json_path(
            document,
            &["env", CLAUDE_CODE_DISABLE_1M_CONTEXT_ENV],
            Value::String("1".to_owned()),
        );
    } else {
        remove_json_path(document, &["env", CLAUDE_CODE_DISABLE_1M_CONTEXT_ENV]);
    }
}

pub fn opus_version_pin(document: &Value) -> Result<Option<String>, ()> {
    match read_json_path(document, &["env", ANTHROPIC_DEFAULT_OPUS_MODEL_ENV]) {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(()),
    }
}

pub fn set_opus_version_pin(document: &mut Value, model: Option<&str>) {
    if let Some(model) = model {
        set_json_path(
            document,
            &["env", ANTHROPIC_DEFAULT_OPUS_MODEL_ENV],
            Value::String(model.to_owned()),
        );
    } else {
        remove_json_path(document, &["env", ANTHROPIC_DEFAULT_OPUS_MODEL_ENV]);
    }
}

pub fn respect_gitignore(document: &Value) -> Result<bool, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::RespectGitignore))? {
        PersistedSettingValue::Missing => Ok(true),
        PersistedSettingValue::Bool(value) => Ok(value),
        PersistedSettingValue::String(_) => Err(()),
    }
}

pub fn set_respect_gitignore(document: &mut Value, enabled: bool) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::RespectGitignore),
        PersistedSettingValue::Bool(enabled),
    );
}

pub fn preferred_notification_channel(document: &Value) -> Result<PreferredNotifChannel, ()> {
    match read_persisted_setting(document, setting_spec(SettingId::Notifications))? {
        PersistedSettingValue::Missing => Ok(PreferredNotifChannel::default()),
        PersistedSettingValue::Bool(_) => Err(()),
        PersistedSettingValue::String(value) => {
            PreferredNotifChannel::from_stored(&value).ok_or(())
        }
    }
}

pub fn set_preferred_notification_channel(document: &mut Value, channel: PreferredNotifChannel) {
    write_persisted_setting(
        document,
        setting_spec(SettingId::Notifications),
        PersistedSettingValue::String(channel.as_stored().to_owned()),
    );
}

fn resolve_paths(
    home_override: Option<&Path>,
    project_root_override: Option<&Path>,
    bridge: Option<&dyn crate::agent::client::AgentBridge>,
) -> Result<SettingsPaths, String> {
    let home = if let Some(path) = home_override {
        path.to_path_buf()
    } else {
        dirs::home_dir().ok_or_else(|| "Failed to resolve home directory".to_owned())?
    };
    let project_root = if let Some(path) = project_root_override {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|err| format!("Failed to resolve current directory: {err}"))?
    };

    // User settings live under <config_dir>, which honours
    // $CLAUDE_CONFIG_DIR — delegate to AgentBridge so the env var is
    // resolved in exactly one place (and a remote-daemon bridge can
    // surface its own `<config_dir>`). The home_override case (used
    // by tests) and the no-bridge case (early init / disconnected)
    // both bypass the bridge.
    let settings = match (home_override, bridge) {
        (None, Some(bridge)) => bridge.config_dir().join(SETTINGS_FILENAME),
        (Some(_), _) | (None, None) => home.join(CLAUDE_DIR).join(SETTINGS_FILENAME),
    };

    Ok(SettingsPaths {
        settings,
        local_settings: project_root.join(CLAUDE_DIR).join(LOCAL_SETTINGS_FILENAME),
        preferences: home.join(PREFERENCES_FILENAME),
    })
}

/// Map the TUI's `SettingFile` enum onto forge-sdk's
/// `SettingsTarget`. Used by `persist_setting_change` to delegate
/// writes to the SDK while keeping `SettingFile` as the TUI-domain
/// type that callers reason about.
pub fn settings_target_for(file: super::SettingFile, cwd: PathBuf) -> forge_sdk::SettingsTarget {
    match file {
        super::SettingFile::Settings => forge_sdk::SettingsTarget::User,
        super::SettingFile::LocalSettings => forge_sdk::SettingsTarget::ProjectLocal { cwd },
        super::SettingFile::Preferences => forge_sdk::SettingsTarget::Preferences,
    }
}

fn empty_object() -> Value {
    Value::Object(Map::new())
}

fn read_json_or_empty(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .filter(Value::is_object)
        .unwrap_or_else(empty_object)
}

fn unique_temp_path(parent: &Path, filename_hint: Option<&str>) -> PathBuf {
    let stamp =
        SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_nanos());
    let filename = filename_hint.unwrap_or(SETTINGS_FILENAME);
    parent.join(format!(".{filename}.{stamp}.tmp"))
}

fn read_json_path<'a>(document: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = document;
    for key in path {
        current = current.as_object()?.get(*key)?;
    }
    Some(current)
}

fn set_json_path(document: &mut Value, path: &[&str], value: Value) {
    let Some((last_key, parents)) = path.split_last() else {
        return;
    };

    let mut current = ensure_object_mut(document);
    for key in parents {
        let child = current.entry((*key).to_owned()).or_insert_with(|| Value::Object(Map::new()));
        if !child.is_object() {
            *child = Value::Object(Map::new());
        }
        current = match child {
            Value::Object(object) => object,
            _ => unreachable!("child must be an object after normalization"),
        };
    }

    current.insert((*last_key).to_owned(), value);
}

fn remove_json_path(document: &mut Value, path: &[&str]) {
    if let Value::Object(object) = document {
        remove_from_object_path(object, path);
    }
}

fn remove_from_object_path(object: &mut Map<String, Value>, path: &[&str]) -> bool {
    let Some((head, tail)) = path.split_first() else {
        return object.is_empty();
    };

    if tail.is_empty() {
        object.remove(*head);
        return object.is_empty();
    }

    let should_remove_child = if let Some(child) = object.get_mut(*head) {
        match child {
            Value::Object(child_object) => remove_from_object_path(child_object, tail),
            _ => true,
        }
    } else {
        false
    };

    if should_remove_child {
        object.remove(*head);
    }

    object.is_empty()
}

fn normalized_root(document: &Value) -> Value {
    match document {
        Value::Object(object) => Value::Object(object.clone()),
        _ => Value::Object(Map::new()),
    }
}

fn ensure_object_mut(document: &mut Value) -> &mut Map<String, Value> {
    if !document.is_object() {
        *document = Value::Object(Map::new());
    }

    match document {
        Value::Object(object) => object,
        _ => unreachable!("document must be an object after normalization"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::setting_spec;

    #[test]
    fn load_missing_files_returns_empty_objects() {
        let dir = tempfile::tempdir().expect("tempdir");

        let loaded = load(Some(dir.path()), Some(dir.path()), None).expect("load");

        assert_eq!(loaded.settings_document, Value::Object(Map::new()));
        assert_eq!(loaded.local_settings_document, Value::Object(Map::new()));
        assert_eq!(loaded.preferences_document, Value::Object(Map::new()));
        assert_eq!(loaded.paths.settings, dir.path().join(".claude").join("settings.json"));
        assert_eq!(
            loaded.paths.local_settings,
            dir.path().join(".claude").join("settings.local.json")
        );
        assert_eq!(loaded.paths.preferences, dir.path().join(".claude.json"));
    }

    #[test]
    fn load_malformed_files_returns_empty_objects_silently() {
        // forge-sdk's read path treats malformed JSON the same as a
        // missing file — empty object, no notice, no backup. This is a
        // deliberate simplification of the previous "rename to .bak +
        // surface a banner" UX.
        let dir = tempfile::tempdir().expect("tempdir");
        let settings_path = dir.path().join(".claude").join("settings.json");
        let preferences_path = dir.path().join(".claude.json");
        std::fs::create_dir_all(settings_path.parent().expect("settings parent"))
            .expect("create settings dir");
        std::fs::write(&settings_path, r#"{"fastMode":true}"#).expect("write settings");
        std::fs::write(&preferences_path, "{ not-json").expect("write malformed");

        let loaded = load(Some(dir.path()), Some(dir.path()), None).expect("load");

        assert_eq!(fast_mode(&loaded.settings_document), Ok(true));
        assert_eq!(loaded.preferences_document, Value::Object(Map::new()));
    }

    #[test]
    fn persisted_setting_readers_apply_defaults() {
        let document = Value::Object(Map::new());

        assert_eq!(default_permission_mode(&document), Ok(DefaultPermissionMode::Default));
        assert_eq!(respect_gitignore(&document), Ok(true));
        assert_eq!(terminal_progress_bar_enabled(&document), Ok(true));
        assert_eq!(output_style(&document), Ok(OutputStyle::Default));
        assert_eq!(model(&document), Ok(None));
        assert_eq!(language(&document), Ok(None));
        assert_eq!(preferred_notification_channel(&document), Ok(PreferredNotifChannel::Iterm2));
    }

    #[test]
    fn persisted_setting_readers_reject_invalid_values() {
        let invalid_notification = serde_json::json!({
            "preferredNotifChannel": "not-a-channel"
        });
        let invalid_output_style = serde_json::json!({
            "outputStyle": "Verbose"
        });
        let invalid_gitignore = serde_json::json!({
            "respectGitignore": "yes"
        });
        let invalid_model = serde_json::json!({
            "model": true
        });
        let invalid_language = serde_json::json!({
            "language": true
        });
        let invalid_permission_mode = serde_json::json!({
            "permissions": {
                "defaultMode": "not-a-mode"
            }
        });

        assert_eq!(preferred_notification_channel(&invalid_notification), Err(()));
        assert_eq!(output_style(&invalid_output_style), Err(()));
        assert_eq!(respect_gitignore(&invalid_gitignore), Err(()));
        assert_eq!(model(&invalid_model), Err(()));
        assert_eq!(language(&invalid_language), Err(()));
        assert_eq!(default_permission_mode(&invalid_permission_mode), Err(()));
    }

    #[test]
    fn save_persists_settings_values_without_dropping_neighboring_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.json");
        let mut document = serde_json::json!({
            "fastMode": false,
            "permissions": {
                "defaultMode": "default",
                "keep": true
            },
            "model": "old-model",
            "language": "English",
            "unknown": {
                "keep": true
            }
        });

        set_fast_mode(&mut document, true);
        set_default_permission_mode(&mut document, DefaultPermissionMode::Plan);
        set_model(&mut document, Some("sonnet"));
        set_language(&mut document, Some("German"));

        save(&path, &document).expect("save");
        let saved: Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json");

        assert_eq!(fast_mode(&saved), Ok(true));
        assert_eq!(default_permission_mode(&saved), Ok(DefaultPermissionMode::Plan));
        assert_eq!(model(&saved), Ok(Some("sonnet".to_owned())));
        assert_eq!(language(&saved), Ok(Some("German".to_owned())));
        assert_eq!(saved["permissions"]["keep"], Value::Bool(true));
        assert_eq!(saved["unknown"]["keep"], Value::Bool(true));
    }

    #[test]
    fn save_roundtrips_auto_permission_mode() {
        let mut document = Value::Object(Map::new());
        set_default_permission_mode(&mut document, DefaultPermissionMode::Auto);

        assert_eq!(default_permission_mode(&document), Ok(DefaultPermissionMode::Auto));
    }

    #[test]
    fn set_language_trims_and_removes_whitespace_only_values() {
        let mut document = Value::Object(Map::new());
        set_language(&mut document, Some("  German  "));
        assert_eq!(language(&document), Ok(Some("German".to_owned())));

        set_language(&mut document, Some("   "));
        assert_eq!(language(&document), Ok(None));
    }

    #[test]
    fn save_preserves_unknown_keys_and_updates_output_style() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".claude").join("settings.local.json");
        std::fs::create_dir_all(path.parent().expect("settings parent")).expect("create dir");
        let mut document = serde_json::json!({
            "outputStyle": "Default",
            "spinnerTipsEnabled": true
        });
        set_output_style(&mut document, OutputStyle::Learning);

        save(&path, &document).expect("save");
        let saved: Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json");

        assert_eq!(output_style(&saved), Ok(OutputStyle::Learning));
        assert_eq!(saved["spinnerTipsEnabled"], Value::Bool(true));
    }

    #[test]
    fn set_disable_1m_context_preserves_neighboring_env_keys() {
        let mut document = serde_json::json!({
            "env": {
                "KEEP_ME": "yes"
            },
            "other": true
        });

        set_disable_1m_context(&mut document, true);
        assert_eq!(disable_1m_context(&document), Ok(true));
        assert_eq!(document["env"]["KEEP_ME"], Value::String("yes".to_owned()));

        set_disable_1m_context(&mut document, false);
        assert_eq!(disable_1m_context(&document), Ok(false));
        assert_eq!(document["env"]["KEEP_ME"], Value::String("yes".to_owned()));
        assert_eq!(document["other"], Value::Bool(true));
    }

    #[test]
    fn set_disable_1m_context_removes_empty_env_object() {
        let mut document = serde_json::json!({
            "env": {
                "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1"
            }
        });

        set_disable_1m_context(&mut document, false);

        assert_eq!(disable_1m_context(&document), Ok(false));
        assert!(document.get("env").is_none());
    }

    #[test]
    fn opus_version_pin_returns_none_when_unset() {
        let document = Value::Object(Map::new());

        assert_eq!(opus_version_pin(&document), Ok(None));
    }

    #[test]
    fn opus_version_pin_returns_string_when_set() {
        let document = serde_json::json!({
            "env": {
                "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-7"
            }
        });

        assert_eq!(opus_version_pin(&document), Ok(Some("claude-opus-4-7".to_owned())));
    }

    #[test]
    fn opus_version_pin_errors_on_non_string_value() {
        let document = serde_json::json!({
            "env": {
                "ANTHROPIC_DEFAULT_OPUS_MODEL": true
            }
        });

        assert_eq!(opus_version_pin(&document), Err(()));
    }

    #[test]
    fn set_opus_version_pin_preserves_neighboring_env_keys() {
        let mut document = serde_json::json!({
            "env": {
                "KEEP_ME": "yes"
            },
            "other": true
        });

        set_opus_version_pin(&mut document, Some("claude-opus-4-6"));
        assert_eq!(opus_version_pin(&document), Ok(Some("claude-opus-4-6".to_owned())));
        assert_eq!(document["env"]["KEEP_ME"], Value::String("yes".to_owned()));

        set_opus_version_pin(&mut document, None);
        assert_eq!(opus_version_pin(&document), Ok(None));
        assert_eq!(document["env"]["KEEP_ME"], Value::String("yes".to_owned()));
        assert_eq!(document["other"], Value::Bool(true));
    }

    #[test]
    fn set_opus_version_pin_removes_empty_env_object() {
        let mut document = serde_json::json!({
            "env": {
                "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-7"
            }
        });

        set_opus_version_pin(&mut document, None);

        assert_eq!(opus_version_pin(&document), Ok(None));
        assert!(document.get("env").is_none());
    }

    #[test]
    fn save_persists_preferences_values_without_dropping_neighboring_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".claude.json");
        let mut document = serde_json::json!({
            "preferredNotifChannel": "iterm2",
            "respectGitignore": true,
            "terminalProgressBarEnabled": true,
            "theme": "dark"
        });

        set_preferred_notification_channel(&mut document, PreferredNotifChannel::TerminalBell);
        set_respect_gitignore(&mut document, false);
        set_terminal_progress_bar_enabled(&mut document, false);

        save(&path, &document).expect("save");
        let saved: Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json");

        assert_eq!(preferred_notification_channel(&saved), Ok(PreferredNotifChannel::TerminalBell));
        assert_eq!(respect_gitignore(&saved), Ok(false));
        assert_eq!(terminal_progress_bar_enabled(&saved), Ok(false));
        assert_eq!(saved["theme"], Value::String("dark".to_owned()));
    }

    #[test]
    fn write_persisted_setting_removes_nested_path_and_prunes_empty_parent() {
        let mut document = serde_json::json!({
            "permissions": {
                "defaultMode": "plan"
            },
            "keep": true
        });

        write_persisted_setting(
            &mut document,
            setting_spec(SettingId::DefaultPermissionMode),
            PersistedSettingValue::Missing,
        );

        assert_eq!(
            document,
            serde_json::json!({
                "keep": true
            })
        );
    }

    #[test]
    fn read_persisted_setting_uses_json_path_metadata() {
        let document = serde_json::json!({
            "permissions": {
                "defaultMode": "plan"
            }
        });

        let value =
            read_persisted_setting(&document, setting_spec(SettingId::DefaultPermissionMode));

        assert_eq!(value, Ok(PersistedSettingValue::String("plan".to_owned())));
    }
}
