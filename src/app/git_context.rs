// Copyright 2025 Simon Peter Rothgang
// SPDX-License-Identifier: Apache-2.0

//! Thin TUI-side cache for the bridge-pushed git context snapshots.
//!
//! Filesystem reads, the `.git/HEAD` walker, the `notify::Watcher`,
//! and the 75ms debounce all live in `forge_sdk::git`. The TUI starts
//! a watcher per session via `AgentBridge::start_git_context_watch`
//! and consumes `BridgeEvent::GitContextSnapshot` events; this module
//! is the App-side cache those events feed.

use crate::agent::types::{GitBranchInfo, GitContextInfo};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum BranchDisplayState {
    Named(String),
    Detached,
    #[default]
    NoRepo,
    Unknown,
}

impl BranchDisplayState {
    #[must_use]
    pub(crate) fn as_deref(&self) -> Option<&str> {
        match self {
            Self::Named(branch) => Some(branch.as_str()),
            Self::Detached | Self::NoRepo | Self::Unknown => None,
        }
    }
}

impl From<GitBranchInfo> for BranchDisplayState {
    fn from(info: GitBranchInfo) -> Self {
        match info {
            GitBranchInfo::Named(name) => Self::Named(name),
            GitBranchInfo::Detached => Self::Detached,
            GitBranchInfo::NoRepo => Self::NoRepo,
            GitBranchInfo::Unknown => Self::Unknown,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct GitContextState {
    branch: BranchDisplayState,
}

impl GitContextState {
    #[must_use]
    pub(crate) fn branch_name(&self) -> Option<&str> {
        self.branch.as_deref()
    }

    /// Apply a bridge-pushed snapshot. Returns `true` when the
    /// resolved branch state changed (i.e. the caller should mark
    /// `needs_redraw`).
    pub(crate) fn apply_snapshot(&mut self, info: GitContextInfo) -> bool {
        let next = BranchDisplayState::from(info.branch);
        if next != self.branch {
            self.branch = next;
            return true;
        }
        false
    }

    #[cfg(test)]
    pub(crate) fn set_branch_for_test(&mut self, branch: Option<&str>) {
        self.branch = match branch {
            Some(branch) => BranchDisplayState::Named(branch.to_owned()),
            None => BranchDisplayState::NoRepo,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::{BranchDisplayState, GitContextState};
    use crate::agent::types::{GitBranchInfo, GitContextInfo};

    #[test]
    fn default_is_no_repo() {
        let state = GitContextState::default();
        assert_eq!(state.branch_name(), None);
    }

    #[test]
    fn apply_snapshot_named_returns_true_on_change() {
        let mut state = GitContextState::default();
        let changed = state.apply_snapshot(GitContextInfo {
            branch: GitBranchInfo::Named("main".to_owned()),
        });
        assert!(changed);
        assert_eq!(state.branch_name(), Some("main"));
    }

    #[test]
    fn apply_snapshot_returns_false_on_same() {
        let mut state = GitContextState::default();
        state.apply_snapshot(GitContextInfo {
            branch: GitBranchInfo::Named("main".to_owned()),
        });
        let changed = state.apply_snapshot(GitContextInfo {
            branch: GitBranchInfo::Named("main".to_owned()),
        });
        assert!(!changed);
    }

    #[test]
    fn apply_snapshot_detached_yields_no_branch_name() {
        let mut state = GitContextState::default();
        state.apply_snapshot(GitContextInfo { branch: GitBranchInfo::Detached });
        assert_eq!(state.branch_name(), None);
        assert_eq!(state.branch, BranchDisplayState::Detached);
    }

    #[test]
    fn set_branch_for_test_updates_state() {
        let mut state = GitContextState::default();
        state.set_branch_for_test(Some("feature/x"));
        assert_eq!(state.branch_name(), Some("feature/x"));
        state.set_branch_for_test(None);
        assert_eq!(state.branch_name(), None);
    }
}
