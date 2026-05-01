//! Cache split policy constants. Mirrors upstream's
//! `agent-sdk/src/bridge/cache_policy.ts`. Used by tooling.rs for
//! persisted-output preview-line skipping.

#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, Copy)]
pub struct CacheSplitPolicy {
    pub soft_limit_bytes: usize,
    pub hard_limit_bytes: usize,
    pub preview_limit_bytes: usize,
}

pub const CACHE_SPLIT_POLICY: CacheSplitPolicy = CacheSplitPolicy {
    soft_limit_bytes: 1536,
    hard_limit_bytes: 4096,
    preview_limit_bytes: 2048,
};

#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn preview_kilobyte_label(policy: CacheSplitPolicy) -> String {
    let kb = policy.preview_limit_bytes as f64 / 1024.0;
    if kb.fract() == 0.0 { format!("{}KB", kb as u64) } else { format!("{kb:.1}KB") }
}

#[cfg(test)]
mod tests {
    use super::{CACHE_SPLIT_POLICY, preview_kilobyte_label};

    #[test]
    fn default_policy_label_is_2kb() {
        assert_eq!(preview_kilobyte_label(CACHE_SPLIT_POLICY), "2KB");
    }
}
