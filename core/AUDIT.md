# Deep Audit: warpstate/core

**Scope:** Pattern matching engine that feeds vyre — scans the entire software supply chain.  
**Date:** 2026-04-10  
**Auditor:** Kimi Code CLI  

---

## Executive Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Correctness | 5 | 4 | 3 | 1 |
| Safety | 0 | 0 | 1 | 3 |
| Async/Sync | 2 | 1 | 1 | 0 |
| Memory | 2 | 2 | 2 | 1 |
| Performance | 0 | 2 | 3 | 2 |
| API | 0 | 1 | 3 | 2 |

**Most severe issues:**
1. Aho-Corasick literal matcher uses `LeftmostFirst` instead of `LeftmostLongest`, causing false negatives when shorter literals shadow longer ones.
2. GPU synchronous API (`scan_blocking`) panics at runtime due to `tokio::task::yield_now()` being called under `pollster::block_on`.
3. Compact-DFA construction maps truncated states to the unanchored start state (0) instead of a dead state, causing spurious matches or restart loops.
4. Variable-length regex fallback in `scan_suffix_from_state` computes match start as `end - 0`, reporting zero-length garbage matches.
5. Serialized index loader trusts length prefixes from untrusted bytes, allowing OOM/DoS via tiny malformed files.

---

## 1. CORRECTNESS

### C1. Aho-Corasick `LeftmostFirst` causes false negatives  🔴 CRITICAL
- **File:** `src/pattern/compiler.rs` (lines 258–268)
- **Issue:** The comment explicitly states *"Use `LeftmostLongest` to ensure all patterns are found. `LeftmostFirst` can silently skip patterns when shorter patterns at the same position shadow longer ones."* The code, however, sets `.match_kind(aho_corasick::MatchKind::LeftmostFirst)`.
- **Impact:** If a short literal is inserted before a longer one that shares the same start position, the longer match is silently discarded. At internet scale this means patterns are missed.
- **Fix:** Change to `LeftmostLongest` (or `LeftmostFirst` with documented acceptance of the shadowing behavior, but the comment says it is unacceptable).

### C2. Compact-DFA truncation maps missing states to start state 0  🔴 CRITICAL
- **File:** `src/dfa/builder.rs` (lines 104, 167)
- **Issue:** When the compact BFS queue hits `MAX_DFA_STATES`, the loop breaks early. Any later transition to an unvisited state is mapped to `0` via `state_map.get(&target).copied().unwrap_or(0)`. State `0` is the *unanchored start state*, not a dead state.
- **Impact:** A truncated compact DFA can restart the search from the beginning instead of rejecting the input, producing spurious matches or infinite restart loops.
- **Fix:** Map missing states to a dedicated dead-state ID (or return an error and discard the compact DFA).

### C3. Variable-length regex fallback reports zero-length garbage matches  🔴 CRITICAL
- **File:** `src/dfa/scan.rs` (lines 290–300, `scan_suffix_from_state`)
- **Issue:** `scan_suffix_from_state` calls `collect_fixed_length_matches_at`, which computes `start = end - pat_len`. For variable-length regexes `fixed_regex_length` returns `None`, so `pat_len` defaults to `0`. The reported match becomes a zero-length match at the end offset.
- **Impact:** Any variable-length regex that escapes the algebraic GPU region and falls back to suffix scanning returns completely wrong match positions.
- **Fix:** Do not call `collect_fixed_length_matches_at` for patterns where `fixed_regex_length` is `None`; use a proper reverse scan or full regex search.

### C4. Backward start search capped at 256 bytes  🔴 CRITICAL
- **File:** `src/dfa/scan.rs` (lines 376, 440, `scan_native_with_compact` / `scan_native_with_dfa`)
- **Issue:** Both fallback paths limit the backwards start-position search to `match_end.saturating_sub(256)`. For any regex match longer than 256 bytes, the code defaults to `best = match_end.saturating_sub(1).max(pos)`, which is almost never the real start.
- **Impact:** Long matches on the compact/native DFA fallback path get incorrect start offsets. A single byte shift at internet scale means billions of misattributed matches.
- **Fix:** Remove the arbitrary 256-byte cap or replace it with a full reverse-DFA walk.

### C5. Literal GPU shaders ignore `gid.y` in 2D dispatches  🔴 CRITICAL
- **File:** `src/gpu/dispatch.rs` (`compute_workgroups`), `src/shader.rs`
- **Issue:** For inputs > ~16 MiB, `compute_workgroups` emits 2D dispatches (`workgroups_y > 1`). The literal prefilter and verify shaders only read `gid.x`, not `gid.y`. Every workgroup row re-processes the same byte positions.
- **Impact:** Duplicate matches are emitted, spuriously exhausting the match buffer and producing inconsistent results between CPU and GPU.
- **Fix:** Update literal shaders to compute global position as `pos = gid.x + (gid.y * 65535u * WORKGROUP_SIZE)`, matching the SMEM regex shader.

### C6. Compiled index prefilter bounds not validated — slice panic  🔴 CRITICAL
- **File:** `src/compiled_index/query.rs` (`prefilter_candidates`), `src/compiled_index/mod.rs` (`read_prefilter_table`)
- **Issue:** `prefilter_candidates()` slices `literals.literal_prefilter_table.entries[start..end]` where `start`/`end` come from serialized `bucket_ranges`. The parser never validates that these ranges point inside the `entries` array.
- **Impact:** A crafted index causes a panic (slice out-of-bounds) at scan time.
- **Fix:** Validate every `bucket_ranges` entry against `entries.len()` during parsing.

### C7. Big-endian serialization corruption  🔴 CRITICAL
- **File:** `src/compiled_index/builder.rs` (`push_u32_slice`, `push_u32_array_256`)
- **Issue:** `bytemuck::cast_slice(values)` writes **native-endian** bytes. The loader (`mod.rs`) reads them back with `u32::from_le_bytes`.
- **Impact:** An index built on a big-endian machine and loaded on a little-endian machine (or vice versa) will have byte-swapped DFA transition tables, match pointers, pattern lengths, and byte-classes. This produces crashes or guaranteed false negatives.
- **Fix:** Explicitly write little-endian bytes (e.g., `value.to_le_bytes()`) in the builder, or read native-endian in the loader.

### C8. Case-insensitive literals lose flag in combined DFA path  🟠 HIGH
- **File:** `src/pattern/mod.rs` (`compiled_regex_dfa`)
- **Issue:** When converting literals to regexes for the combined DFA path, the code calls `regex::escape(literal)` but does **not** wrap the result in a case-insensitive group when `ir.case_insensitive` is true.
- **Impact:** Case-insensitive literals scanned via the combined DFA path only match exact case, producing false negatives.
- **Fix:** Wrap escaped literals in `(?i:…)` when the case-insensitive flag is set.

### C9. Compact-DFA only stores one pattern ID per accept state  🟠 HIGH
- **File:** `src/dfa/builder.rs` (line 148)
- **Issue:** In a true multi-pattern DFA a single accept state can match multiple patterns. The compact table discards all but `match_pattern(state, 0)`.
- **Impact:** If the compact path is ever used for multi-pattern scans, patterns are lost (false negatives).
- **Fix:** Store a bitmap or small vec of pattern IDs per accept state, or disable the compact path for multi-pattern sets.

### C10. CPU/GPU semantic divergence on overlapping literals  🟠 HIGH
- **Files:** `src/cpu/mod.rs`, `src/cpu/scan.rs`, `src/gpu/mod.rs`, `src/hash_scan.rs`
- **Issue:** The CPU Aho-Corasick path is strictly non-overlapping (`LeftmostFirst` / `LeftmostLongest`). The GPU literal verify shader is inherently overlapping (every verified hash hit is emitted). HashScanner and several fast paths also advance by the shortest match, dropping overlaps.
- **Impact:** Fallback from GPU to CPU (or vice versa) can produce different match sets for inputs with overlapping literal patterns. Parity tests may pass on simple inputs but fail in production.
- **Fix:** Unify on a single documented semantics (preferably overlapping, since the GPU already does it) and make the CPU path match.

### C11. Batch API silently drops matches for items > 4 GB  🟠 HIGH
- **File:** `src/batch.rs` (`decoalesce`)
- **Issue:** `decoalesce()` skips matches where `local_start` or `local_end` do not fit in `u32`:
  ```rust
  let Ok(start) = u32::try_from(local_start) else { continue };
  ```
  While `coalesce()` limits the total buffer to `u32::MAX`, it does not limit individual items. A single item can exceed 4 GB.
- **Impact:** Valid matches in a single large batched item are silently dropped — false negatives.
- **Fix:** Either reject items > 4 GB at planning time, or use `u64` offsets in the batch API.

### C12. `scan_native_with_compact` overwrites earlier pattern IDs  🟠 HIGH
- **File:** `src/dfa/scan.rs` (line 440)
- **Issue:** `scan_native_with_compact` only remembers the *last* match pattern encountered while walking forward (`match_pat = compact.match_pattern[state as usize]`). If two different patterns match at the same end position, the earlier one is overwritten and never reported.
- **Impact:** False negatives when multiple patterns share an end position on the compact fallback path.
- **Fix:** Collect all matching pattern IDs at the accept state, not just one.

### C13. Non-overlapping semantics differ between native regex and mixed literal+regex sets  🟡 MEDIUM
- **File:** `src/dfa/scan.rs` (`scan_native_with_dfa`)
- **Issue:** `scan_native_with_dfa` advances `pos = match_end.max(pos + 1)`, giving non-overlapping semantics *per pattern*. However, `cpu/scan.rs` iterates over all `regex_dfas` independently and appends their matches. Matches from different regexes can therefore overlap, contradicting `PatternSet::scan` documented non-overlapping semantics for mixed sets.
- **Impact:** Inconsistent overlap behavior depending on how patterns are partitioned.
- **Fix:** Merge all regex matches through a single non-overlapping pipeline, or document the divergence.

### C14. `scan_single_literal_with_finder` saturates end past buffer  🟡 MEDIUM
- **File:** `src/cpu/scan.rs`
- **Issue:** `scan_single_literal_with_finder` uses `saturating_add` for `end = pos + needle_len`. If `pos` is near `u32::MAX` and `needle_len` is large, `end` clamps to `u32::MAX`, producing a match that extends past the input buffer.
- **Impact:** Invalid match ranges returned to callers.
- **Fix:** Use `checked_add` and reject the match if it overflows.

### C15. `FusedScanner` quadratic behavior with long patterns  🟡 MEDIUM
- **File:** `src/fused.rs`
- **Issue:** When `max_pattern_len > FUSED_WINDOW_BYTES` (4096), `stride` becomes 1. The window loop iterates `data.len()` times, repeatedly merging overlapping candidate regions.
- **Impact:** On multi-megabyte files this looks like a hang.
- **Fix:** Cap `stride` at a minimum > 1, or refuse to build a `FusedScanner` with patterns longer than the window.

### C16. `specialize.rs` fast path drops overlapping matches  🟡 MEDIUM
- **File:** `src/specialize.rs` (`ScanStrategy::MultiMemchr`)
- **Issue:** `scan_multi_memchr` collects all matches, sorts them, then emits only non-overlapping matches (`m.start >= last_end`). If the underlying semantics allow overlapping literal matches, this fast path silently drops them.
- **Impact:** False negatives on overlapping inputs.
- **Fix:** Align the fast path with the canonical semantics, or disable it when overlap is required.

### C17. `query.rs` `scan_literals` discards overlapping matches  🟡 MEDIUM
- **File:** `src/compiled_index/query.rs`
- **Issue:** `scan_literals()` advances `pos` to `found.end` after finding the "best" match, discarding overlapping matches at the same position (e.g., `"a"` and `"ab"` on `"ab"`).
- **Impact:** Divergence from full `PatternSet::scan()` semantics.
- **Fix:** Document the non-overlapping contract or implement overlap.

### C18. `RegexDFA::from_serialized_parts` rebuilds DFA on every scan call  🟢 LOW
- **File:** `src/dfa/mod.rs`
- **Issue:** `from_serialized_parts` stores `native_dfa: None`. Every subsequent call to `scan_native_without_jit_with` re-deserializes the DFA from bytes on the hot path.
- **Impact:** Unnecessary CPU overhead on the fallback path.
- **Fix:** Eagerly deserialize and cache the native DFA object.

---

## 2. SAFETY

### S1. `unsafe` block inventory

| # | File | Line | Kind | Safety Comment | Sound |
|---|------|------|------|----------------|-------|
| 1 | `src/pipeline.rs` | 84 | `unsafe fn` | Yes | Yes |
| 2 | `src/pipeline.rs` | 88 | `unsafe { ... }` | Yes | Yes |
| 3 | `src/router.rs` | 287 | `unsafe { ... }` | Yes | Yes* |
| 4 | `src/dma.rs` | 116 | `unsafe { ... }` | Yes | Yes |
| 5 | `src/dma.rs` | 174 | `unsafe fn` | Yes | Yes |
| 6 | `src/dma.rs` | 180 | `unsafe impl` | Yes | Yes |
| 7 | `src/dfa/mod.rs` | 63 | `unsafe { ... }` | Yes | Yes† |
| 8 | `src/dfa/mod.rs` | 96 | `unsafe { ... }` | Yes | Yes† |
| 9 | `src/dfa/mod.rs` | 108 | `unsafe { ... }` | Yes | Yes† |
| 10 | `src/dfa/mod.rs` | 115 | `unsafe { ... }` | Yes | Yes† |
| 11 | `src/dfa/mod.rs` | 232 | `unsafe { ... }` | Yes | Yes |
| 12 | `src/dfa/mod.rs` | 248 | `unsafe { ... }` | Yes | Yes |
| 13 | `src/dfa/mod.rs` | 255 | `unsafe { ... }` | Yes | Yes† |
| 14 | `src/dfa/mod.rs` | 437 | `unsafe { ... }` | **No** | Yes |
| 15 | `src/dfa/mod.rs` | 451 | `unsafe { ... }` | Yes | Yes |
| 16 | `src/dfa/scan.rs` | 420 | `unsafe { ... }` | **No** | Yes‡ |

\* Sound because `join()` guarantees the original `self` outlives the dereferenced pointer.  
† Sound under the invariant that DFA states/indices are always in-bounds (guaranteed by construction).  
‡ `_mm_prefetch` tolerates invalid addresses, and the pointer arithmetic is in-bounds in practice; a safety comment would improve maintainability.

### S2. Missing safety comment on `mmap` FFI call  🟡 MEDIUM
- **File:** `src/dfa/mod.rs` (line 437)
- **Issue:** The `libc::mmap` call is inside an `unsafe` block but carries no `// SAFETY:` comment. While `mmap` with these flags is well-defined, the crate safety policy requires comments on every unsafe block.
- **Fix:** Add a brief `// SAFETY:` comment explaining the arguments.

### S3. Missing safety comment on `_mm_prefetch`  🟢 LOW
- **File:** `src/dfa/scan.rs` (line 420)
- **Issue:** The `_mm_prefetch` intrinsic is safe to call with invalid addresses, but the `as_ptr().add(idx)` pointer arithmetic is only valid if `state` is in-bounds. No safety comment is present.
- **Fix:** Add `// SAFETY: state is guaranteed in-bounds by DFA construction, so the prefetch address lies within the transition table.`

---

## 3. ASYNC / SYNC

### A1. `scan_blocking` panics from `tokio::task::yield_now` under `pollster`  🔴 CRITICAL
- **Files:** `src/gpu/readback.rs`, `src/gpu_smem/scan.rs`, `src/gpu_dfa.rs`
- **Issue:** `await_buffer_map` and `read_matches` call `tokio::task::yield_now().await` inside polling loops. `GpuMatcher::scan_blocking` drives the future with `pollster::block_on`, which is **not** a Tokio runtime. `yield_now()` panics at runtime.
- **Impact:** The entire synchronous GPU scanning API (`GpuMatcher::scan_blocking`, `GpuScanner::scan`) is unusable outside a Tokio runtime. This is a known issue and is **not fixed**.
- **Fix:** Replace `tokio::task::yield_now` with `std::thread::yield_now` (synchronous) or use an executor-agnostic yield such as `futures::pending()` plus `device.poll(Maintain::Poll)`.

### A2. `router.rs::scan_blocking` unsound raw-pointer cast + thread-per-call  🔴 CRITICAL
- **File:** `src/router.rs` (lines 280–300)
- **Issue:** When called inside a tokio runtime, `scan_blocking` casts `&self` to a raw `usize`, sends it across `std::thread::spawn`, and dereferences it in the new thread. It also constructs a *new* `tokio::runtime::Builder::new_current_thread()` on **every call**.
- **Impact:**
  - Spawns an OS thread per scan call, easily exhausting system threads under load.
  - The safety comment claims "AutoMatcher is Send+Sync" but there is no explicit `unsafe impl Send/Sync`. If any wrapped type (e.g., wgpu resources) is `!Send`, this is undefined behavior.
- **Fix:** Do not spawn a thread per call. Use `tokio::task::spawn_blocking` if inside tokio, or avoid tokio-specific yielding so `pollster` can be used directly.

### A3. `new_blocking` / `with_config_blocking` panic inside tokio  🟠 HIGH
- **File:** `src/router.rs`
- **Issue:** The doc comments claim these constructors are "safe from inside or outside async runtimes." For the `#[cfg(feature = "gpu")]` path, they call `pollster::block_on(...)`. `pollster::block_on` will **panic** when called inside an existing tokio runtime. There is no tokio detection here, unlike `scan_blocking`.
- **Impact:** Constructors crash when invoked from within a tokio runtime.
- **Fix:** Either detect tokio and use `spawn_blocking`, or document the panic hazard accurately.

### A4. `pipeline.rs` boxes every future via `async_trait`  🟡 MEDIUM
- **File:** `src/pipeline.rs`
- **Issue:** The `Matcher` impl for `StreamPipeline<T>` uses `#[async_trait::async_trait]`, which boxes every future.
- **Impact:** A guaranteed heap allocation per `scan` call in a performance-critical matching engine.
- **Fix:** Migrate to RPITIT (return-position impl trait in trait) now that the MSRV supports it.

### A5. `stream.rs` silently truncates matches  🟡 MEDIUM
- **File:** `src/stream.rs` (`feed`)
- **Issue:** `feed()` stops the callback when `matches.len() >= cpu::MAX_CPU_MATCHES` and returns `Ok(matches)` without any indication that results were truncated.
- **Impact:** High-match streams silently drop data.
- **Fix:** Return a distinct error or a `truncated: bool` flag when the cap is reached.

---

## 4. MEMORY

### M1. Allocation-of-Death in compiled index loader  🔴 CRITICAL
- **File:** `src/compiled_index/mod.rs` (`Cursor` helpers: `read_u32_vec`, `read_u32_pairs`, `read_names`, etc.)
- **Issue:** All cursor helpers call `Vec::with_capacity(count)` where `count` is read directly from the untrusted byte stream **before** verifying that the buffer contains that many elements. A 20-byte malformed file can declare `prefix_meta_len = 0xFFFFFFFF`, triggering a ~64 GB allocation attempt.
- **Impact:** DoS / OOM / panic on small malicious inputs.
- **Fix:** Validate `count * element_size <= remaining_bytes` before allocating.

### M2. Double copy on load + no file size cap  🟠 HIGH
- **File:** `src/compiled_index/mod.rs`
- **Issue:** `CompiledPatternIndex::load(data)` copies the entire file into a `Vec<u8>`. During `parse()`, it then copies literal bytes (`packed_bytes = serialized[range].to_vec()`), offsets, names, etc. `load_from_file()` uses `fs::read()` with no size cap.
- **Impact:** Double (or triple) memory usage on load; multi-gigabyte files can exhaust RAM.
- **Fix:** Use zero-copy slices into the original buffer where possible, and cap file size before reading.

### M3. `GpuBufferPool` can retain unbounded VRAM  🟠 HIGH
- **File:** `src/gpu/device.rs`
- **Issue:** The pool limits entries to 64 but evicts the *smallest* buffer on overflow. This can cause the pool to hoard up to 64 × largest size-class buffers (e.g., 64 × 128 MiB ≈ 8 GiB of VRAM).
- **Impact:** GPU OOM on adapters with limited memory.
- **Fix:** Evict by total pooled size, not by entry count, or add a bytes budget.

### M4. No DFA size limit validation in builder API  🟡 MEDIUM
- **File:** `src/dfa/builder.rs`
- **Issue:** `dfa_size_limit` is passed straight to `regex_automata` without validation. A caller can pass `usize::MAX`. While `regex_automata` has its own limits, the API does not surface them well.
- **Fix:** Clamp `dfa_size_limit` to a reasonable maximum and document the behavior.

### M5. `HashScanner` table can grow to 4.3 GB  🟡 MEDIUM
- **File:** `src/hash_scan.rs`
- **Issue:** `LengthGroupAligned::new()` caps `table_size` at `1 << 28` (~268 M entries, ~4.3 GB). This prevents unbounded growth but is still large enough to OOM many systems.
- **Fix:** Lower the cap or make it proportional to available memory.

### M6. `RegexDFA::clone` retries huge pages  🟢 LOW
- **File:** `src/dfa/mod.rs`
- **Issue:** `clone` always attempts `MAP_HUGETLB` again even if the original DFA used standard backing. If huge pages are exhausted, the clone silently downgrades to standard pages.
- **Impact:** Unexpected performance variance.
- **Fix:** Record the original backing strategy and reuse it on clone.

---

## 5. PERFORMANCE

### P1. `scan_multi_regex_kway` linear scan for minimum start position  🟠 HIGH
- **File:** `src/dfa/scan.rs`
- **Issue:** The function does a linear scan (`O(k * m)`) to find the minimum start position on every iteration. For many patterns with many matches this is quadratic in the number of matches.
- **Fix:** Use a binary heap for `O(m log k)` merging.

### P2. Compact-DFA fallback is O(n²)  🟠 HIGH
- **File:** `src/dfa/scan.rs` (`scan_native_with_compact`)
- **Issue:** For every starting position `pos` it walks the DFA forward, and for variable-length patterns it then walks backwards up to 256 bytes.
- **Impact:** The compact DFA fallback is much slower than the `regex::bytes::Regex` fast path it is supposed to replace.
- **Fix:** Replace the nested loop with a single forward pass or a proper two-way DFA scan.

### P3. Auto-tuning scans data twice for first 3 calls  🟡 MEDIUM
- **File:** `src/router.rs` (`auto_tune_and_scan`)
- **Issue:** For the first 3 calls (`tune_samples < 3`), `auto_tune_and_scan` runs **both** CPU and GPU scans.
- **Impact:** A significant performance footgun that is not highlighted in the public API docs.
- **Fix:** Document the behavior or defer tuning until an explicit warmup call.

### P4. `literal_count()` re-parses offset vector on every call  🟡 MEDIUM
- **File:** `src/compiled_index/mod.rs`
- **Issue:** `literal_count()` re-parses the offset vector from `serialized` on every call (used by `scan()` in `query.rs`).
- **Impact:** Unexpected hidden cost for callers.
- **Fix:** Cache the count at load time.

### P5. Regex compilation is independent per pattern  🟢 LOW
- **File:** `src/dfa/builder.rs`
- **Issue:** Each regex is compiled independently. This prevents combinatorial DFA explosion but forces the CPU scan to run N separate regex searches and merge the results.
- **Impact:** Sub-optimal throughput for large regex sets.
- **Fix:** Consider a combined DFA for small regex sets where combinatorial explosion is manageable.

### P6. JIT safety check is overly conservative  🟢 LOW
- **File:** `src/dfa/jit_bridge.rs`
- **Issue:** `patterns_are_jit_safe` uses naive substring checks (`pattern.contains('^')`, etc.), rejecting perfectly safe patterns such as `[^abc]` or `\^`.
- **Impact:** JIT disabled unnecessarily for common character-class patterns.
- **Fix:** Parse the regex HIR and check for anchors at the true root level only.

---

## 6. API

### API1. `scan_blocking` docs are false  🟠 HIGH
- **File:** `src/router.rs`
- **Issue:** Doc comments claim constructors are "safe from inside or outside async runtimes," but the GPU path uses `pollster::block_on`, which panics inside tokio.
- **Fix:** Fix the docs to reflect the actual runtime requirements and panic hazards.

### API2. `ZeroCopyScan::mapped_parts()` returns raw pointer from safe fn  🟡 MEDIUM
- **File:** `src/pipeline.rs`
- **Issue:** `mapped_parts()` is a safe `pub fn` that returns `(*mut u8, usize)`. Exposing a raw pointer without requiring `unsafe` on the caller is a footgun.
- **Fix:** Mark the function `unsafe fn` or return a safe wrapper type.

### API3. Config builder allows silent footguns  🟡 MEDIUM
- **File:** `src/config.rs`
- **Issue:**
  - `gpu_threshold(0)` forces GPU for all inputs (including tiny ones) with no warning.
  - `gpu_max_input_size(0)` silently disables GPU.
  - `chunk_size(0)` clamps to `1`, which is defensive but undocumented.
  - No check that `chunk_overlap < chunk_size`.
- **Fix:** Add builder validation or doc warnings for these values.

### API4. `GpuBackend` is public but large  🟡 MEDIUM
- **File:** `src/router.rs`
- **Issue:** `GpuBackend` is `pub` and uses `#[allow(clippy::large_enum_variant)]`. Users matching on it directly can cause unexpected stack copies.
- **Fix:** Make the enum `#[non_exhaustive]` and provide accessor methods instead of exposing large variants.

### API5. `dma.rs::flush_to_vram()` double-unmap panic risk  🟢 LOW
- **File:** `src/dma.rs`
- **Issue:** `flush_to_vram()` does not check `self.flushed` before calling `self.buffer.unmap()`. Calling it twice will likely panic inside wgpu.
- **Fix:** Add an early return or assert if already flushed.

### API6. Suppressed documentation lints  🟢 LOW
- **File:** `src/lib.rs`
- **Issue:** The crate enables `#![warn(missing_docs)]` but explicitly allows `clippy::missing_errors_doc`, `clippy::missing_panics_doc`, `clippy::doc_markdown`, and `clippy::must_use_candidate`.
- **Impact:** Public methods can return `Result` without documenting error conditions, and constructors are not forced to be `#[must_use]`.
- **Fix:** Remove the allow attributes and fix the resulting warnings.

---

## Actionable Fix Checklist

1. **Pattern compiler:** Change Aho-Corasick to `LeftmostLongest`.
2. **DFA builder:** Map truncated compact-DFA states to a dead state, not state 0.
3. **DFA scan:** Fix `scan_suffix_from_state` for variable-length regexes; remove or justify the 256-byte backward cap.
4. **GPU readback:** Replace `tokio::task::yield_now` with executor-agnostic yielding.
5. **GPU dispatch/shader:** Make literal shaders 2D-dispatch aware (`gid.y`).
6. **Compiled index builder:** Write little-endian explicitly in `push_u32_slice`.
7. **Compiled index loader:** Validate length-prefix bounds before `Vec::with_capacity`.
8. **Compiled index query:** Validate `bucket_ranges` against `entries.len()`.
9. **Router:** Remove thread-per-call in `scan_blocking`; use `spawn_blocking` or direct polling.
10. **Batch API:** Reject or properly handle items > 4 GB.
11. **Stream scanner:** Return truncation indicator when `MAX_CPU_MATCHES` is hit.
12. **API docs:** Correct `new_blocking` / `scan_blocking` documentation to reflect tokio panic hazard.
