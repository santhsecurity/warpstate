# ADVERSARIAL REVIEW: warpstate/core Streaming Components

**Scope:** src/stream.rs, src/pipeline.rs, src/batch.rs  
**Threat Model:** Streaming scan can LOSE matches at chunk boundaries, overflow buffers, or silently truncate results. A single false negative = malware undetected.

---

## CRITICAL FINDINGS

### CRITICAL | src/stream.rs:86-91 | Silent match truncation at MAX_CPU_MATCHES
**Description:** The `feed()` method silently truncates matches when count exceeds MAX_CPU_MATCHES (1,048,576). The visitor callback returns `false` to stop scanning, but `scan_with` returns `Ok(())` — no error is propagated. Matches beyond the limit are lost WITHOUT ANY INDICATION to the caller.

```rust
// In feed():
cpu::scan_with(self.patterns.ir(), &self.window, &mut |mat| {
    if matches.len() >= cpu::MAX_CPU_MATCHES {
        return false;  // SILENT TRUNCATION - no error!
    }
    matches.push(mat);
    true
})?;  // Ok(()) returned even if truncated
```

**Attack Vector:** Supply chunked data where a single window (overlap + chunk) contains >1M matches. This is trivial with repeating patterns like "aaaa..." and single-byte pattern "a".

**Impact:** FALSE NEGATIVE — malware signatures lost in high-match scenarios.

---

### CRITICAL | src/stream.rs:97 | Boundary match retention logic flaw
**Description:** `matches.retain(|mat| mat.end as usize > overlap_len)` filters matches based on END position. A match that STARTS in the overlap region but EXTENDS into the new chunk is incorrectly retained if its end is beyond overlap_len, BUT the offset adjustment is applied to ALL retained matches including those that started in old data.

```rust
matches.retain(|mat| mat.end as usize > overlap_len);  // Retains based on END
for mat in &mut matches {
    mat.start = add_global_offset(mat.start, base_offset)?;  // Adjusts ALL
```

**Attack Vector:** Pattern of length L where match starts at position overlap_len - 1 (in overlap) and ends at overlap_len + L - 1. Match is retained but start offset adjusted incorrectly (off by base_offset - 1).

**Impact:** INCORRECT OFFSETS — match reported at wrong position, potential false positive/negative in position-dependent detection.

---

### CRITICAL | src/stream.rs:129-135 | add_global_offset saturating_add overflow
**Description:** `base_offset.saturating_add(offset as usize)` can silently saturate at usize::MAX, then `u32::try_from` fails. But the error path doesn't distinguish between legitimate InputTooLarge and this overflow condition. More critically, the `saturating_add` prevents panic but produces WRONG offset values when overflow would have occurred.

```rust
fn add_global_offset(offset: u32, base_offset: usize) -> Result<u32> {
    let global_offset = base_offset.saturating_add(offset as usize);  // SILENT SATURATION
    u32::try_from(global_offset).map_err(|_| Error::InputTooLarge { ... })
}
```

**Attack Vector:** Stream >4GB with matches near the end. If base_offset is near u32::MAX and offset addition saturates, wrong offset calculated.

**Impact:** WRONG MATCH OFFSETS — position information corrupted.

---

### CRITICAL | src/pipeline.rs:101-147 | No regex DFA state carry across chunk boundaries
**Description:** The `StreamPipeline::scan()` implementation chunks data and scans independently. For literal patterns with `BlockMatcher`, this is safe due to overlap window. However, for regex patterns with DFA state, there is NO state preservation across chunks. A regex pattern that matches across chunk boundary will be LOST.

```rust
// In scan():
while offset < data.len() {
    let mut matches = self.backend.scan_block(chunk_data).await?;
    // Each scan_block starts fresh - no DFA state from previous chunk carried over
```

**Attack Vector:** Any regex pattern that could match across the configured chunk boundary will fail when the input is chunked.

**Impact:** FALSE NEGATIVE — regex-based malware signatures lost at chunk boundaries.

---

### CRITICAL | src/pipeline.rs:155-164 | dedup_by merge logic truncates matches incorrectly
**Description:** The deduplication logic keeps the SHORTER match when duplicates are found (`earlier.end = earlier.end.min(later.end)`). This is documented as handling "DFA EOI transition" effects but can SILENTLY TRUNCATE legitimate longer matches.

```rust
all_matches.dedup_by(|later, earlier| {
    if later.pattern_id == earlier.pattern_id && later.start == earlier.start {
        earlier.end = earlier.end.min(later.end);  // FORCES SHORTER MATCH
        true
    }
```

**Attack Vector:** A pattern that genuinely matches at the same start position with different lengths (e.g., regex alternations). The longer match is discarded.

**Impact:** FALSE NEGATIVE / TRUNCATED MATCH — full pattern match not reported.

---

### CRITICAL | src/batch.rs:117-119 | decoalesce silently drops remaining matches on cursor overrun
**Description:** When cursor advances past all entries, the remaining matches in `global_matches` are silently dropped via `break`, not returned as error or partial result.

```rust
if cursor >= map.entries.len() {
    break;  // SILENTLY DROPS all remaining matches!
}
```

**Attack Vector:** Malformed coalesce map or corrupted match offsets cause cursor to advance beyond entries. All subsequent matches lost.

**Impact:** SILENT DATA LOSS — matches disappear without error indication.

---

### CRITICAL | src/batch.rs:138-143 | usize→u32 truncation in decoalesce silently skips matches
**Description:** Large items (>4GB) have offsets that don't fit in u32. The code silently `continue`s past these matches instead of erroring.

```rust
let Ok(start) = u32::try_from(local_start) else {
    continue;  // SILENT SKIP!
};
let Ok(end) = u32::try_from(local_end) else {
    continue;  // SILENT SKIP!
};
```

**Attack Vector:** Batch scan items larger than 4GB. Matches in those items are silently dropped.

**Impact:** FALSE NEGATIVE — large file matches lost.

---

## HIGH SEVERITY FINDINGS

### HIGH | src/stream.rs:103-113 | Overlap rotation keeps wrong bytes when chunk < max_pattern_len
**Description:** `keep_len = self.max_pattern_len.min(self.window.len())` keeps max_pattern_len bytes from window tail. If a chunk is smaller than max_pattern_len, this keeps bytes from the OLD overlap that may have already been processed.

```rust
let keep_len = self.max_pattern_len.min(self.window.len());
let start = window_len - keep_len;
self.overlap.resize(keep_len, 0);
self.overlap.copy_from_slice(&self.window[start..]);  // May re-include old overlap data
```

**Attack Vector:** Single-byte chunks with large patterns. The overlap buffer cycles incorrectly, potentially causing duplicate match reports or missed matches.

**Impact:** DUPLICATE MATCHES or MISSED MATCHES — overlap handling becomes inconsistent.

---

### HIGH | src/stream.rs:68-74 | processed_bytes overflow check doesn't prevent offset wrap
**Description:** The check for `new_processed_bytes > u32::MAX` catches future overflow but doesn't account for the fact that `processed_bytes` could be near u32::MAX while `overlap_len` is also large, causing issues in base_offset calculation.

```rust
let new_processed_bytes = self.processed_bytes.saturating_add(chunk.len());
if new_processed_bytes > u32::MAX as usize {
    return Err(Error::InputTooLarge { ... });
}
// But processed_bytes itself might be > u32::MAX - max_pattern_len
// causing base_offset calculation issues
let base_offset = self.processed_bytes.saturating_sub(overlap_len);
```

**Impact:** WRONG OFFSETS near 4GB stream boundary.

---

### HIGH | src/pipeline.rs:127-136 | Offset addition uses saturating_add silently
**Description:** Match offsets adjusted for chunk position use `saturating_add`, which silently caps at u32::MAX instead of erroring.

```rust
for m in &mut matches {
    m.start = m.start.saturating_add(offset_u32);  // SILENT SATURATION
    m.end = m.end.saturating_add(offset_u32);      // SILENT SATURATION
}
```

**Attack Vector:** Large streams where chunk offset + match position exceeds u32::MAX.

**Impact:** WRONG OFFSETS — matches reported at incorrect (maxed-out) positions.

---

### HIGH | src/batch.rs:131-133 | Cross-boundary match rejection too aggressive
**Description:** Matches that span item boundaries in coalesced buffer are `continue`d (silently dropped). The check `if global_end > item_offset + item_len` rejects any match extending past item boundary.

```rust
if global_end > item_offset + item_len {
    continue;  // DROPS cross-boundary matches
}
```

**Attack Vector:** Pattern that happens to match across the concatenation point of two batch items. Valid pattern match in individual items that coincidentally spans the boundary in coalesced view is lost.

**Impact:** FALSE NEGATIVE — legitimate matches lost at coalesce boundaries.

---

### HIGH | src/pipeline.rs:145-146 | Zero advance scenario causes infinite loop
**Description:** `offset += advance` where `advance = chunk_size.saturating_sub(effective_overlap).max(1)` ensures at least 1 byte advance. But if chunk_size is 0 (empty backend), this still advances, causing issues with offset tracking.

Actually the `max(1)` prevents infinite loops, but when `chunk_size == 0`, it scans 0 bytes and advances by 1, potentially skipping data.

**Impact:** POTENTIAL DATA SKIP with misconfigured zero-block-size backend.

---

## MEDIUM SEVERITY FINDINGS

### MEDIUM | src/stream.rs:124-126 | finish() idempotency claim is misleading
**Description:** Documentation claims `finish()` is idempotent, but calling it after partial feed with matches pending in overlap could return different results. The comment says "returns empty vector" but doesn't guarantee this if `finish()` is called multiple times or if overlap contains complete matches.

**Impact:** API CONTRACT VIOLATION — caller expectations not met.

---

### MEDIUM | src/batch.rs:65-91 | coalesce unchecked capacity allocation
**Description:** `Vec::with_capacity(total_size)` where total_size can be up to u32::MAX (4GB). On 32-bit systems or memory-constrained environments, this could panic or OOM. No graceful handling.

**Impact:** CRASH / OOM on large batch operations.

---

### MEDIUM | src/pipeline.rs:108-110 | Empty chunk scan bypasses deduplication
**Description:** Early return for `data.len() <= chunk_size` bypasses the deduplication logic, meaning single-chunk scans may return different match characteristics than multi-chunk scans for the same data.

**Impact:** INCONSISTENT BEHAVIOR between small and large inputs.

---

## REVIEW SUMMARY

| Metric | Count |
|--------|-------|
| CRITICAL | 7 |
| HIGH | 5 |
| MEDIUM | 3 |

**Overall Quality Score:** 4/10

**Primary Issues:**
1. **Silent truncation** — Multiple paths where matches are silently dropped without error
2. **Offset corruption** — Saturating arithmetic and overflow handling produce wrong positions
3. **Boundary losses** — Chunk and coalesce boundaries can lose matches
4. **Regex DFA state loss** — StreamPipeline doesn't preserve regex state across chunks

**Recommendations:**
1. Replace all `continue`/`break` silent drops with proper error propagation
2. Add explicit overflow errors instead of saturating arithmetic for offsets
3. Implement regex DFA state checkpoint/resume for StreamPipeline
4. Add adversarial tests for >1M matches, >4GB streams, and exact boundary alignments
5. Verify overlap math with formal proof or exhaustive test for all chunk sizes < max_pattern_len
