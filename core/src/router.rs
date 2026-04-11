//! GPU router for warpstate internet-scale scan backends.

use parking_lot::Mutex;
#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use std::time::Instant;

use crate::config::AutoMatcherConfig;
use crate::error::Result;
use crate::Match;
use crate::PatternSet;

#[cfg(feature = "gpu")]
use crate::gpu::GpuMatcher;
#[cfg(feature = "gpu")]
use crate::gpu::SharedDeviceQueue;

/// Type of GPU backend spawned for async and auto routing.
#[cfg(feature = "gpu")]
#[non_exhaustive]
pub enum GpuBackend {
    /// The consolidated wgpu backend.
    Standard(GpuMatcher),
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard(m) => write!(f, "Standard({m:?})"),
        }
    }
}

#[cfg(feature = "gpu")]
impl GpuBackend {
    /// Perform an asynchronous scan across the input data using the underlying GPU backend.
    pub async fn scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        match self {
            Self::Standard(m) => m.scan(data).await,
        }
    }
}

#[derive(Debug)]
struct RouterState {
    gpu_threshold: usize,
    #[cfg(feature = "gpu")]
    device_queue: Option<SharedDeviceQueue>,
    #[cfg(feature = "gpu")]
    gpu: Option<Arc<GpuBackend>>,
    #[cfg(feature = "gpu")]
    last_gpu_init_attempt: Option<Instant>,
}

/// A matcher that routes eligible input to GPU backends and fails loudly when GPU is unavailable.
#[derive(Debug)]
pub struct AutoMatcher {
    patterns: PatternSet,
    config: AutoMatcherConfig,
    /// Lock-free threshold for hot-path routing decisions.
    /// Updated atomically by auto-tune; read without locking on every scan().
    gpu_threshold_atomic: std::sync::atomic::AtomicUsize,
    state: Mutex<RouterState>,
}

impl AutoMatcher {
    /// Create an auto-routing matcher with default configuration.
    pub async fn new(patterns: &PatternSet) -> Result<Self> {
        Self::with_config(patterns, AutoMatcherConfig::default()).await
    }

    /// Create an auto-routing matcher with explicit configuration.
    pub async fn with_config(patterns: &PatternSet, mut config: AutoMatcherConfig) -> Result<Self> {
        let max_pattern_length = patterns
            .ir()
            .offsets
            .iter()
            .map(|&(_, len)| len as usize)
            .max()
            .unwrap_or(0);
        config.chunk_overlap = config.configured_chunk_overlap().max(max_pattern_length);

        let threshold = config.configured_gpu_threshold();
        Ok(Self {
            patterns: patterns.clone(),
            gpu_threshold_atomic: std::sync::atomic::AtomicUsize::new(threshold),
            state: Mutex::new(RouterState {
                gpu_threshold: threshold,
                #[cfg(feature = "gpu")]
                device_queue: None,
                #[cfg(feature = "gpu")]
                gpu: None,
                #[cfg(feature = "gpu")]
                last_gpu_init_attempt: None,
            }),
            config,
        })
    }

    /// Synchronous constructor.
    ///
    /// The GPU feature path drives initialization with a small, executor-independent
    /// blocking executor. It must not be called from latency-sensitive async tasks.
    pub fn new_blocking(patterns: &PatternSet) -> Result<Self> {
        #[cfg(feature = "gpu")]
        {
            pollster::block_on(Self::new(patterns))
        }
        #[cfg(not(feature = "gpu"))]
        {
            // No async work without GPU — construct directly.
            Self::with_config_blocking(patterns, AutoMatcherConfig::default())
        }
    }

    /// Synchronous constructor with config.
    ///
    /// The GPU feature path blocks the current thread until initialization finishes.
    pub fn with_config_blocking(patterns: &PatternSet, config: AutoMatcherConfig) -> Result<Self> {
        #[cfg(feature = "gpu")]
        {
            pollster::block_on(Self::with_config(patterns, config))
        }
        #[cfg(not(feature = "gpu"))]
        {
            // No async work without GPU — construct directly.
            let mut config = config;
            let max_pattern_length = patterns
                .ir()
                .offsets
                .iter()
                .map(|&(_, len)| len as usize)
                .max()
                .unwrap_or(0);
            config.chunk_overlap = config.configured_chunk_overlap().max(max_pattern_length);
            let threshold = config.configured_gpu_threshold();
            Ok(Self {
                patterns: patterns.clone(),
                gpu_threshold_atomic: std::sync::atomic::AtomicUsize::new(threshold),
                state: Mutex::new(RouterState {
                    gpu_threshold: threshold,
                }),
                config,
            })
        }
    }

    /// Backward-compatible constructor for ad-hoc options.
    pub async fn with_options(
        patterns: &PatternSet,
        gpu_threshold: usize,
        gpu_max_input_size: usize,
    ) -> Result<Self> {
        Self::with_config(
            patterns,
            AutoMatcherConfig::new()
                .gpu_threshold(gpu_threshold)
                .gpu_max_input_size(gpu_max_input_size),
        )
        .await
    }

    /// Override the routing threshold.
    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_threshold_atomic
            .store(threshold, std::sync::atomic::Ordering::Relaxed);
        self.state.get_mut().gpu_threshold = threshold;
        self.config.set_gpu_threshold(threshold);
        self
    }

    /// Override the max GPU input size.
    pub fn with_gpu_max_input_size(mut self, max_size: usize) -> Self {
        self.config.set_gpu_max_input_size(max_size);
        self
    }

    /// Get the configured GPU routing threshold.
    /// Lock-free read — no mutex acquisition on the hot path.
    pub fn gpu_threshold(&self) -> usize {
        self.gpu_threshold_atomic
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the configured GPU maximum input size.
    pub fn gpu_max_input_size(&self) -> usize {
        self.config.configured_gpu_max_input_size()
    }

    /// Whether a GPU backend is available.
    pub fn has_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            lock_router_state(&self.state).gpu.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Scan data, automatically choosing the best backend.
    pub async fn scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = data;
            Err(crate::error::Error::NoGpuAdapter)
        }

        #[cfg(feature = "gpu")]
        {
            let threshold = self.gpu_threshold();
            if data.len() < threshold || data.len() > self.config.configured_gpu_max_input_size() {
                return Err(crate::error::Error::NoGpuAdapter);
            }

            let Some(gpu) = self.ensure_gpu_backend().await? else {
                return Err(crate::error::Error::NoGpuAdapter);
            };

            gpu.scan(data).await
        }
    }

    /// Synchronous scan for callers without an async runtime.
    ///
    /// Uses an executor-independent blocking poll on the current thread.
    pub fn scan_blocking(&self, input: &[u8]) -> Result<Vec<Match>> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = input;
            Err(crate::error::Error::NoGpuAdapter)
        }
        #[cfg(feature = "gpu")]
        {
            pollster::block_on(self.scan(input))
        }
    }

    /// Force CPU scan regardless of input size.
    pub fn scan_cpu(&self, data: &[u8]) -> Result<Vec<Match>> {
        self.patterns.scan(data)
    }

    /// Force GPU scan regardless of routing threshold. Returns error if no GPU.
    pub async fn scan_gpu(&self, data: &[u8]) -> Result<Vec<Match>> {
        #[cfg(feature = "gpu")]
        {
            match self.ensure_gpu_backend().await? {
                Some(gpu) => gpu.scan(data).await,
                None => Err(crate::error::Error::NoGpuAdapter),
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = data;
            Err(crate::error::Error::NoGpuAdapter)
        }
    }

    #[cfg(feature = "gpu")]
    async fn ensure_gpu_backend(&self) -> Result<Option<Arc<GpuBackend>>> {
        {
            let mut state = lock_router_state(&self.state);
            if let Some(gpu) = &state.gpu {
                return Ok(Some(Arc::clone(gpu)));
            }
            if let Some(last_attempt) = state.last_gpu_init_attempt {
                if last_attempt.elapsed().as_secs() < 60 {
                    return Ok(None);
                }
            }
            state.last_gpu_init_attempt = Some(Instant::now());
        }

        let device_queue = match crate::gpu::acquire_device().await {
            Ok(device_queue) => device_queue,
            Err(_) => return Ok(None),
        };
        let backend = self.init_gpu_backend(Arc::clone(&device_queue));
        let mut state = lock_router_state(&self.state);
        state.device_queue = Some(device_queue);
        if let Some(gpu) = backend {
            let gpu = Arc::new(gpu);
            state.gpu = Some(Arc::clone(&gpu));
            return Ok(Some(gpu));
        }
        Ok(None)
    }

    #[cfg(feature = "gpu")]
    fn init_gpu_backend(&self, device_queue: SharedDeviceQueue) -> Option<GpuBackend> {
        match GpuMatcher::from_device(&device_queue, &self.patterns, self.config.clone()) {
            Ok(standard) => {
                tracing::debug!("init_gpu_backend: chose consolidated GpuMatcher");
                Some(GpuBackend::Standard(standard))
            }
            Err(error) => {
                tracing::warn!(?error, "init_gpu_backend: GpuMatcher initialization failed");
                None
            }
        }
    }
}

fn lock_router_state(state: &Mutex<RouterState>) -> parking_lot::MutexGuard<'_, RouterState> {
    state.lock()
}

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use crate::config::DEFAULT_GPU_THRESHOLD;
    use crate::PatternSet;

    use super::*;

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(future)
    }

    #[test]
    fn default_threshold_is_64kb() {
        assert_eq!(DEFAULT_GPU_THRESHOLD, 65_536);
    }

    #[test]
    fn auto_matcher_with_config_propagates_values() {
        let ps = PatternSet::builder().literal("test").build().unwrap();
        let config = AutoMatcherConfig::new()
            .gpu_threshold(8192)
            .gpu_max_input_size(64 * 1024 * 1024)
            .max_matches(2048);

        let matcher = block_on(AutoMatcher::with_config(&ps, config)).unwrap();
        assert_eq!(matcher.gpu_threshold(), 8192);
        assert_eq!(matcher.gpu_max_input_size(), 64 * 1024 * 1024);
    }

    #[test]
    fn cpu_scan_path_always_works() {
        let ps = PatternSet::builder().literal("test").build().unwrap();
        let matcher = block_on(AutoMatcher::new(&ps)).unwrap();
        let results = matcher.scan_cpu(b"test input").unwrap();
        assert_eq!(results.len(), 1);
    }
}
