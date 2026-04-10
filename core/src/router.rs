//! Auto-routing between CPU and GPU backends.

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
use crate::algebraic::AlgebraicDfaMatcher;
#[cfg(feature = "gpu")]
use crate::gpu::GpuMatcher;
#[cfg(feature = "gpu")]
use crate::gpu::SharedDeviceQueue;
#[cfg(feature = "gpu")]
use crate::gpu_smem::SmemDfaMatcher;
#[cfg(feature = "gpu")]
use crate::matcher::BlockMatcher;
#[cfg(feature = "gpu")]
use crate::persistent::PersistentMatcher;

/// Type of GPU backend spawned for async and auto routing.
///
/// Tried in order of preference:
/// 1. Algebraic (parallel DFA via prefix scan — O(log n), fastest)
/// 2. SmemDfa (shared memory staged DFA — good for medium state counts)
/// 3. Persistent (double-buffered standard DFA — most general)
/// 4. Standard (single-buffer DFA — fallback)
#[allow(clippy::large_enum_variant)]
#[cfg(feature = "gpu")]
pub enum GpuBackend {
    /// Algebraic parallel DFA via prefix scan — O(log n) for small DFAs.
    Algebraic(AlgebraicDfaMatcher),
    /// Shared-memory staged DFA — faster than global memory.
    Smem(SmemDfaMatcher),
    /// The persistent, double-buffered GPU backend.
    Persistent(PersistentMatcher),
    /// The standard GPU backend.
    Standard(GpuMatcher),
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Algebraic(_) => write!(f, "Algebraic"),
            Self::Smem(_) => write!(f, "Smem"),
            Self::Persistent(_) => write!(f, "Persistent"),
            Self::Standard(m) => write!(f, "Standard({m:?})"),
        }
    }
}

#[cfg(feature = "gpu")]
impl GpuBackend {
    /// Perform an asynchronous scan across the input data using the underlying GPU backend.
    pub async fn scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        match self {
            Self::Algebraic(m) => m.scan_block(data).await.map_err(Into::into),
            Self::Smem(m) => m.scan_block(data).await.map_err(Into::into),
            Self::Persistent(m) => m.scan_block(data).await.map_err(Into::into),
            Self::Standard(m) => m.scan(data).await,
        }
    }
}

#[derive(Debug)]
struct RouterState {
    gpu_threshold: usize,
    tuned: bool,
    tune_samples: usize,
    speedup_sum: f64,
    #[cfg(feature = "gpu")]
    device_queue: Option<SharedDeviceQueue>,
    #[cfg(feature = "gpu")]
    gpu: Option<Arc<GpuBackend>>,
    #[cfg(feature = "gpu")]
    last_gpu_init_attempt: Option<Instant>,
}

/// A matcher that automatically routes between CPU and GPU.
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
                tuned: false,
                tune_samples: 0,
                speedup_sum: 0.0,
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

    /// Get the current GPU threshold, including auto-tuned updates.
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
            self.patterns.scan(data)
        }

        #[cfg(feature = "gpu")]
        {
            let threshold = self.gpu_threshold();
            if data.len() < threshold || data.len() > self.config.configured_gpu_max_input_size() {
                return self.patterns.scan(data);
            }

            let Some(gpu) = self.ensure_gpu_backend().await? else {
                return self.patterns.scan(data);
            };

            if self.config.is_auto_tune_threshold_enabled() {
                let should_tune = {
                    let state = lock_router_state(&self.state);
                    !state.tuned
                };
                if should_tune {
                    return self.auto_tune_and_scan(gpu.as_ref(), data).await;
                }
            }

            gpu.scan(data).await
        }
    }

    /// Synchronous scan for callers without an async runtime.
    ///
    /// Internally creates a minimal tokio current-thread runtime. For
    /// repeated calls, prefer using the async [scan](Self::scan) method
    /// with a shared runtime.
    pub fn scan_blocking(&self, input: &[u8]) -> Result<Vec<Match>> {
        pollster::block_on(self.scan(input))
    }

    #[cfg(feature = "gpu")]
    async fn auto_tune_and_scan(&self, gpu: &GpuBackend, data: &[u8]) -> Result<Vec<Match>> {
        let cpu_start = Instant::now();
        let cpu_matches = self.patterns.scan(data)?;
        let cpu_elapsed = cpu_start.elapsed();

        let gpu_start = Instant::now();
        let gpu_result = gpu.scan(data).await;
        let gpu_elapsed = gpu_start.elapsed();

        let mut state = lock_router_state(&self.state);

        let cpu_time = cpu_elapsed.as_secs_f64();
        let gpu_time = gpu_elapsed.as_secs_f64();

        let speedup = if gpu_time > 0.0 {
            cpu_time / gpu_time
        } else {
            1.0
        };

        state.tune_samples += 1;
        state.speedup_sum += speedup;

        if state.tune_samples >= 3 {
            state.tuned = true;
            let avg_speedup = state.speedup_sum / state.tune_samples as f64;
            if avg_speedup >= 1.0 && gpu_result.is_ok() {
                state.gpu_threshold = data.len().min(state.gpu_threshold);
            } else {
                state.gpu_threshold = data.len().saturating_add(1).max(state.gpu_threshold);
            }
            // Publish to atomic for lock-free reads on subsequent scan() calls.
            self.gpu_threshold_atomic
                .store(state.gpu_threshold, std::sync::atomic::Ordering::Relaxed);
        }
        drop(state);

        match gpu_result {
            Ok(gpu_matches) if gpu_elapsed <= cpu_elapsed => Ok(gpu_matches),
            Ok(_) | Err(_) => Ok(cpu_matches),
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
        if let Ok(algebraic) =
            AlgebraicDfaMatcher::from_device(Arc::clone(&device_queue), &self.patterns)
        {
            tracing::debug!("init_gpu_backend: chose AlgebraicDfaMatcher");
            Some(GpuBackend::Algebraic(algebraic))
        } else if let Ok(smem) = SmemDfaMatcher::from_device(
            Arc::clone(&device_queue),
            &self.patterns,
            self.config.clone(),
        ) {
            tracing::debug!("init_gpu_backend: chose SmemDfaMatcher");
            Some(GpuBackend::Smem(smem))
        } else if let Ok(persistent) = PersistentMatcher::from_device(
            Arc::clone(&device_queue),
            &self.patterns,
            self.config.clone(),
        ) {
            tracing::debug!("init_gpu_backend: chose PersistentMatcher");
            Some(GpuBackend::Persistent(persistent))
        } else if let Ok(standard) =
            GpuMatcher::from_device(&device_queue, &self.patterns, self.config.clone())
        {
            tracing::debug!("init_gpu_backend: chose GpuMatcher");
            Some(GpuBackend::Standard(standard))
        } else {
            tracing::warn!("init_gpu_backend: all backend initializations failed");
            None
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
