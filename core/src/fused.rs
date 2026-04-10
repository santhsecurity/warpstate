//! Fused CPU-side prefiltering for literal-heavy scans.
//!
//! `FusedScanner` combines optional byte-frequency prefiltering with optional
//! SIMD prefix verification, then executes `PatternSet::scan_with` directly on
//! merged candidate regions. This keeps the pipeline streaming and avoids
//! materializing intermediate candidate vectors between stages.

use crate::cpu::{check_input_size, estimated_match_capacity, finish_matches};
use crate::{Error, Match, PatternSet, Result};

#[cfg(feature = "jit")]
use std::sync::OnceLock;

const FUSED_WINDOW_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone)]
struct FusedSievePlan {
    literal_patterns: Box<[Box<[u8]>]>,
    case_insensitive: bool,
}

impl FusedSievePlan {
    fn from_pattern_set(patterns: &PatternSet) -> Option<Self> {
        let ir = patterns.ir();
        if !ir.regex_dfas().is_empty() || ir.offsets.is_empty() {
            return None;
        }

        let literal_patterns = ir
            .offsets
            .iter()
            .filter_map(|&(start, len)| {
                let start = start as usize;
                let end = start.saturating_add(len as usize);
                ir.packed_bytes
                    .get(start..end)
                    .map(|bytes| bytes.to_vec().into_boxed_slice())
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        if literal_patterns.is_empty() {
            return None;
        }

        Some(Self {
            literal_patterns,
            case_insensitive: ir.case_insensitive,
        })
    }

    fn window_has_candidate(&self, haystack: &[u8]) -> Result<bool> {
        let mut chunk_refs = [&[][..]; 16];

        for chunk in self.literal_patterns.chunks(16) {
            for (index, pattern) in chunk.iter().enumerate() {
                chunk_refs[index] = pattern.as_ref();
            }

            let result = if self.case_insensitive {
                simdsieve::SimdSieve::new_case_insensitive(haystack, &chunk_refs[..chunk.len()])
            } else {
                simdsieve::SimdSieve::new(haystack, &chunk_refs[..chunk.len()])
            };

            let mut sieve = result.map_err(|error| Error::PatternCompilationFailed {
                reason: format!(
                    "failed to construct fused SIMD sieve: {error}. Fix: provide at least one non-empty literal pattern and keep SIMD chunks at 16 literals or fewer."
                ),
            })?;

            if sieve.next().is_some() {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

/// Fused literal scanner with optional byte-frequency and SIMD prefix stages.
///
/// The fused path preserves `PatternSet::scan` semantics for literal-only
/// pattern sets. If the compiled pattern set contains regex DFAs, the optional
/// prefilters are disabled automatically and scanning falls back to the
/// underlying `PatternSet`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "fused")]
/// # {
/// use ebpfsieve::{ByteFrequencyFilter, ByteThreshold};
/// use warpstate::{FusedScanner, PatternSet};
///
/// let patterns = PatternSet::builder().literal("needle").build()?;
/// let filter = ByteFrequencyFilter::new([ByteThreshold::new(b'n', 1)])?;
/// let scanner = FusedScanner::new(patterns, Some(filter));
/// let matches = scanner.scan(b"needle in a haystack")?;
/// assert_eq!(matches.len(), 1);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FusedScanner {
    freq_filter: Option<ebpfsieve::ByteFrequencyFilter>,
    sieve: Option<FusedSievePlan>,
    patterns: PatternSet,
    /// JIT-compiled literal scanner. Lazily initialized on first use when the
    /// `jit` feature is enabled and the pattern set is JIT-eligible.
    /// Wrapped in Arc because JitDfa holds executable memory and doesn't impl Clone.
    #[cfg(feature = "jit")]
    jit_scanner: OnceLock<Option<std::sync::Arc<dfajit::JitDfa>>>,
}

impl Clone for FusedScanner {
    fn clone(&self) -> Self {
        Self {
            freq_filter: self.freq_filter.clone(),
            sieve: self.sieve.clone(),
            patterns: self.patterns.clone(),
            #[cfg(feature = "jit")]
            jit_scanner: {
                let new_lock = OnceLock::new();
                if let Some(val) = self.jit_scanner.get() {
                    let _ = new_lock.set(val.clone());
                }
                new_lock
            },
        }
    }
}

impl FusedScanner {
    /// Build a fused scanner around an existing compiled `PatternSet`.
    ///
    /// `freq_filter` is only applied when the pattern set is literal-only.
    /// Mixed literal/regex sets keep exact behavior by bypassing the fused
    /// prefilter stages.
    #[must_use]
    pub fn new(patterns: PatternSet, freq_filter: Option<ebpfsieve::ByteFrequencyFilter>) -> Self {
        let sieve = FusedSievePlan::from_pattern_set(&patterns);
        let freq_filter = if sieve.is_some() { freq_filter } else { None };

        let scanner = Self {
            freq_filter,
            sieve,
            patterns,
            #[cfg(feature = "jit")]
            jit_scanner: OnceLock::new(),
        };

        // Eagerly compile JIT when the feature is enabled — the first scan
        // should hit native code, not the interpreted path.
        #[cfg(feature = "jit")]
        {
            let _ = scanner.get_or_init_jit();
        }

        scanner
    }

    /// Attempt to JIT-compile the pattern set for native execution.
    ///
    /// When the `jit` feature is enabled, this builds a `dfajit::JitDfa` from
    /// the literal patterns. Subsequent scans use the JIT-compiled code path
    /// which eliminates all virtual dispatch between pipeline stages.
    ///
    /// Returns `true` if JIT compilation succeeded.
    #[cfg(feature = "jit")]
    pub fn try_compile_jit(&self) -> bool {
        self.get_or_init_jit().is_some()
    }

    /// Returns `true` if a JIT-compiled scanner is available.
    #[cfg(feature = "jit")]
    pub fn has_jit(&self) -> bool {
        self.jit_scanner.get().is_some_and(|opt| opt.is_some())
    }

    #[cfg(feature = "jit")]
    fn get_or_init_jit(&self) -> &Option<std::sync::Arc<dfajit::JitDfa>> {
        self.jit_scanner.get_or_init(|| {
            let ir = self.patterns.ir();
            if ir.offsets.is_empty() {
                return None;
            }

            // Build literal patterns from the packed bytes
            let mut literal_patterns: Vec<Vec<u8>> = Vec::with_capacity(ir.offsets.len());
            for &(start, len) in &ir.offsets {
                let s = start as usize;
                let e = s.saturating_add(len as usize);
                if let Some(bytes) = ir.packed_bytes.get(s..e) {
                    literal_patterns.push(bytes.to_vec());
                }
            }

            if literal_patterns.is_empty() {
                return None;
            }

            let pattern_refs: Vec<&[u8]> = literal_patterns.iter().map(|p| p.as_slice()).collect();
            match dfajit::JitDfa::from_patterns(&pattern_refs) {
                Ok(jit) => {
                    tracing::debug!(
                        "FusedScanner: JIT compiled {} literal patterns",
                        literal_patterns.len()
                    );
                    Some(std::sync::Arc::new(jit))
                }
                Err(e) => {
                    tracing::debug!("FusedScanner: JIT compilation failed: {e}");
                    None
                }
            }
        })
    }

    /// Scan and materialize non-overlapping matches.
    ///
    /// When a JIT-compiled scanner is available (via `try_compile_jit`), the
    /// scan uses native x86_64 code with zero virtual dispatch. Otherwise,
    /// falls back to the interpreted fused pipeline.
    ///
    /// # Errors
    ///
    /// Returns the same scanning errors as `PatternSet::scan`.
    pub fn scan(&self, data: &[u8]) -> Result<Vec<Match>> {
        check_input_size(data)?;

        // JIT fast path: use compiled native code (eagerly initialized at construction)
        #[cfg(feature = "jit")]
        if let Some(jit) = self.get_or_init_jit() {
            let mut buf = vec![Match::from_parts(0, 0, 0); estimated_match_capacity(data.len())];
            let n = jit.scan(data, &mut buf);
            buf.truncate(n);
            return finish_matches(buf);
        }

        let mut matches = Vec::with_capacity(estimated_match_capacity(data.len()));
        self.scan_with(data, |matched| {
            matches.push(matched);
            true
        })?;
        finish_matches(matches)
    }

    /// Stream matches through `visitor` without allocating candidate regions.
    ///
    /// Returning `false` from `visitor` stops the scan early.
    ///
    /// # Errors
    ///
    /// Returns the same scanning errors as `PatternSet::scan_with`.
    pub fn scan_with<F>(&self, data: &[u8], mut visitor: F) -> Result<()>
    where
        F: FnMut(Match) -> bool,
    {
        check_input_size(data)?;

        if data.is_empty() || (self.freq_filter.is_none() && self.sieve.is_none()) {
            return self.patterns.scan_with(data, visitor);
        }

        let overlap = self.patterns.max_pattern_len().saturating_sub(1);
        let stride = FUSED_WINDOW_BYTES.saturating_sub(overlap).max(1);
        let mut window_start = 0usize;
        let mut pending_region: Option<(usize, usize)> = None;

        while window_start < data.len() {
            let window_end = (window_start + FUSED_WINDOW_BYTES).min(data.len());
            let window = &data[window_start..window_end];

            let mut accepted = true;
            if let Some(filter) = &self.freq_filter {
                accepted = filter.matches_bytes(window);
            }
            if accepted {
                if let Some(sieve) = &self.sieve {
                    accepted = sieve.window_has_candidate(window)?;
                }
            }

            if accepted {
                let region_start = window_start.saturating_sub(overlap);
                let region_end = window_end.saturating_add(overlap).min(data.len());

                if let Some((pending_start, pending_end)) = pending_region {
                    if region_start <= pending_end {
                        pending_region = Some((pending_start, pending_end.max(region_end)));
                    } else {
                        if !self.scan_region(data, pending_start, pending_end, &mut visitor)? {
                            return Ok(());
                        }
                        pending_region = Some((region_start, region_end));
                    }
                } else {
                    pending_region = Some((region_start, region_end));
                }
            }

            if window_end == data.len() {
                break;
            }
            window_start = window_start.saturating_add(stride);
        }

        if let Some((region_start, region_end)) = pending_region {
            let _ = self.scan_region(data, region_start, region_end, &mut visitor)?;
        }

        Ok(())
    }

    fn scan_region<F>(
        &self,
        data: &[u8],
        region_start: usize,
        region_end: usize,
        visitor: &mut F,
    ) -> Result<bool>
    where
        F: FnMut(Match) -> bool,
    {
        let mut keep_scanning = true;
        self.patterns
            .scan_with(&data[region_start..region_end], |matched| {
                let start = region_start + matched.start as usize;
                let end = region_start + matched.end as usize;
                // Safe: check_input_size guarantees data.len() <= u32::MAX,
                // so region offsets + match offsets fit in u32.
                let Ok(start_u32) = u32::try_from(start) else {
                    return false; // Stop scanning if offset overflows
                };
                let Ok(end_u32) = u32::try_from(end) else {
                    return false;
                };
                keep_scanning = visitor(Match {
                    pattern_id: matched.pattern_id,
                    start: start_u32,
                    end: end_u32,
                    padding: matched.padding,
                });
                keep_scanning
            })?;
        Ok(keep_scanning)
    }
}
