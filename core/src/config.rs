//! Configuration for auto-routed and GPU-backed scans.

/// Default maximum input size per GPU chunk in bytes (128 MB).
pub const DEFAULT_MAX_INPUT_SIZE: usize = 128 * 1024 * 1024;

/// Default maximum input size for regex DFA scans on GPU (16 MB).
///
/// The regex DFA shader has $O(N^2)$ worst-case complexity (each thread
/// scans from its position to the end of the input). Large inputs with
/// pathological patterns can cause GPU TDR (Timeout Detection and Recovery).
pub const DEFAULT_MAX_REGEX_INPUT_SIZE: usize = 16 * 1024 * 1024;

/// Default chunk size for GPU scans.
pub const DEFAULT_CHUNK_SIZE: usize = DEFAULT_MAX_INPUT_SIZE;

/// Minimum overlap to preserve matches at chunk boundaries.
///
/// This value is a floor; the actual overlap used per-scan is expanded to at
/// least the length of the longest literal pattern so that matches spanning
/// chunk boundaries are never silently dropped.
pub const DEFAULT_CHUNK_OVERLAP: usize = 4096;

/// Minimum input size before GPU dispatch is preferred over CPU.
///
/// Below this threshold, the overhead of GPU buffer allocation and
/// kernel launch exceeds the speedup from parallel execution. Determined
/// empirically on NVIDIA A100 with warpstate 0.1.0 benchmarks.
pub const DEFAULT_GPU_THRESHOLD: usize = 65_536;

/// Maximum number of matches supported per scan before returning `BufferOverflow`.
///
/// This limit keeps the GPU buffer memory size deterministic and small enough
/// to avoid out-of-memory errors on smaller GPU devices, balancing typical workload
/// needs with memory constraints.
pub const DEFAULT_MAX_MATCHES: u32 = 1_000_000;

/// Builder-backed configuration shared by [`crate::AutoMatcher`] and [`crate::GpuMatcher`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoMatcherConfig {
    pub(crate) gpu_threshold: usize,
    pub(crate) gpu_max_input_size: usize,
    pub(crate) gpu_max_regex_input_size: usize,
    pub(crate) max_matches: u32,
    pub(crate) chunk_size: usize,
    pub(crate) chunk_overlap: usize,
    pub(crate) auto_tune_threshold: bool,
    pub(crate) max_scan_depth: Option<u32>,
}

impl Default for AutoMatcherConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: DEFAULT_GPU_THRESHOLD,
            gpu_max_input_size: DEFAULT_MAX_INPUT_SIZE,
            gpu_max_regex_input_size: DEFAULT_MAX_REGEX_INPUT_SIZE,
            max_matches: DEFAULT_MAX_MATCHES,
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            auto_tune_threshold: true,
            max_scan_depth: None,
        }
    }
}

impl AutoMatcherConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the routing threshold where GPU becomes eligible.
    pub fn gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_threshold = threshold;
        self
    }

    /// Set the maximum input size eligible for routing to the GPU.
    pub fn gpu_max_input_size(mut self, bytes: usize) -> Self {
        self.gpu_max_input_size = bytes;
        self
    }

    /// Set the maximum input size for regex DFA scans on GPU.
    pub fn gpu_max_regex_input_size(mut self, bytes: usize) -> Self {
        self.gpu_max_regex_input_size = bytes;
        self
    }

    /// Set the maximum number of GPU matches buffered per chunk.
    pub fn max_matches(mut self, max_matches: u32) -> Self {
        self.max_matches = max_matches;
        self
    }

    /// Set the chunk size used by the GPU scanner.
    pub fn chunk_size(mut self, bytes: usize) -> Self {
        self.chunk_size = bytes.max(1);
        self
    }

    /// Set chunk overlap used to preserve matches across chunk boundaries.
    pub fn chunk_overlap(mut self, bytes: usize) -> Self {
        self.chunk_overlap = bytes;
        self
    }

    /// Enable or disable first-scan threshold auto-tuning.
    pub fn auto_tune_threshold(mut self, enabled: bool) -> Self {
        self.auto_tune_threshold = enabled;
        self
    }

    /// Set max scan depth for GPU regex matchers.
    pub fn max_scan_depth(mut self, depth: Option<u32>) -> Self {
        self.max_scan_depth = depth;
        self
    }

    /// Get the configured routing threshold.
    pub fn configured_gpu_threshold(&self) -> usize {
        self.gpu_threshold
    }

    /// Get the configured GPU max input size.
    pub fn configured_gpu_max_input_size(&self) -> usize {
        self.gpu_max_input_size
    }

    /// Get the configured GPU max regex input size.
    pub fn configured_gpu_max_regex_input_size(&self) -> usize {
        self.gpu_max_regex_input_size
    }

    /// Get the configured max match buffer size.
    pub fn configured_max_matches(&self) -> u32 {
        self.max_matches
    }

    /// Get the configured chunk size.
    pub fn configured_chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the configured chunk overlap.
    pub fn configured_chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }

    /// Whether threshold auto-tuning is enabled.
    pub fn is_auto_tune_threshold_enabled(&self) -> bool {
        self.auto_tune_threshold
    }

    /// Get the configured max scan depth.
    pub fn configured_max_scan_depth(&self) -> Option<u32> {
        self.max_scan_depth
    }

    /// Mutate the GPU routing threshold in-place.
    ///
    /// Unlike [`gpu_threshold`](Self::gpu_threshold) (which consumes `self`),
    /// this borrows mutably — avoiding a clone when modifying post-construction.
    pub fn set_gpu_threshold(&mut self, threshold: usize) {
        self.gpu_threshold = threshold;
    }

    /// Mutate the GPU max input size in-place.
    pub fn set_gpu_max_input_size(&mut self, bytes: usize) {
        self.gpu_max_input_size = bytes;
    }

    /// Mutate the GPU max regex input size in-place.
    pub fn set_gpu_max_regex_input_size(&mut self, bytes: usize) {
        self.gpu_max_regex_input_size = bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_are_stable() {
        let config = AutoMatcherConfig::default();
        assert_eq!(config.configured_gpu_threshold(), DEFAULT_GPU_THRESHOLD);
        assert_eq!(
            config.configured_gpu_max_input_size(),
            DEFAULT_MAX_INPUT_SIZE
        );
        assert_eq!(config.configured_max_matches(), DEFAULT_MAX_MATCHES);
        assert!(config.is_auto_tune_threshold_enabled());
    }

    #[test]
    fn config_builder_overrides_values() {
        let config = AutoMatcherConfig::new()
            .gpu_threshold(1024)
            .gpu_max_input_size(2048)
            .max_matches(4096)
            .chunk_size(8192)
            .chunk_overlap(128)
            .auto_tune_threshold(false);

        assert_eq!(config.configured_gpu_threshold(), 1024);
        assert_eq!(config.configured_gpu_max_input_size(), 2048);
        assert_eq!(config.configured_max_matches(), 4096);
        assert_eq!(config.configured_chunk_size(), 8192);
        assert_eq!(config.configured_chunk_overlap(), 128);
        assert!(!config.is_auto_tune_threshold_enabled());
    }

    // === Adversarial Config Tests ===

    /// Test that defaults are sane and non-zero
    #[test]
    fn config_defaults_sane() {
        let config = AutoMatcherConfig::default();
        assert!(config.configured_gpu_threshold() > 0);
        assert!(config.configured_gpu_max_input_size() > 0);
        assert!(config.configured_max_matches() > 0);
        assert!(config.configured_chunk_size() > 0);
    }

    /// Test that custom values are correctly stored
    #[test]
    fn config_custom_values() {
        let config = AutoMatcherConfig::new()
            .gpu_threshold(0)
            .gpu_max_input_size(1)
            .max_matches(1)
            .chunk_size(1)
            .chunk_overlap(0);

        assert_eq!(config.configured_gpu_threshold(), 0);
        assert_eq!(config.configured_gpu_max_input_size(), 1);
        assert_eq!(config.configured_max_matches(), 1);
        assert_eq!(config.configured_chunk_size(), 1);
        assert_eq!(config.configured_chunk_overlap(), 0);
    }

    /// Test zero input size is allowed (edge case)
    #[test]
    fn config_zero_input_size() {
        let config = AutoMatcherConfig::new().gpu_max_input_size(0);
        assert_eq!(config.configured_gpu_max_input_size(), 0);
    }

    /// Test huge chunk size is handled
    #[test]
    fn config_huge_chunk_size() {
        let huge = usize::MAX;
        let config = AutoMatcherConfig::new().chunk_size(huge);
        assert_eq!(config.configured_chunk_size(), huge);
    }

    /// Test that chunk_size max(1) logic works
    #[test]
    fn config_chunk_size_minimum_one() {
        let config = AutoMatcherConfig::new().chunk_size(0);
        // chunk_size method uses .max(1) to ensure at least 1
        assert_eq!(config.configured_chunk_size(), 1);
    }

    /// Test max_scan_depth getter
    #[test]
    fn config_max_scan_depth() {
        let config = AutoMatcherConfig::new().max_scan_depth(Some(100));
        assert_eq!(config.configured_max_scan_depth(), Some(100));

        let config_none = AutoMatcherConfig::new().max_scan_depth(None);
        assert_eq!(config_none.configured_max_scan_depth(), None);
    }

    /// Test clone preserves all values
    #[test]
    fn config_clone_preserves_values() {
        let config = AutoMatcherConfig::new()
            .gpu_threshold(1234)
            .gpu_max_input_size(5678)
            .max_matches(9999)
            .auto_tune_threshold(false);

        let cloned = config.clone();
        assert_eq!(cloned.configured_gpu_threshold(), 1234);
        assert_eq!(cloned.configured_gpu_max_input_size(), 5678);
        assert_eq!(cloned.configured_max_matches(), 9999);
        assert!(!cloned.is_auto_tune_threshold_enabled());
    }

    /// Test equality
    #[test]
    fn config_equality() {
        let a = AutoMatcherConfig::new().gpu_threshold(100);
        let b = AutoMatcherConfig::new().gpu_threshold(100);
        let c = AutoMatcherConfig::new().gpu_threshold(200);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
