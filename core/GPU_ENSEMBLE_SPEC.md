# GPU Regex Ensemble — Shared-Memory DFA Architecture

## Problem
Merging N regexes into one DFA via `build_many` produces a product automaton
with O(S₁ × S₂ × ... × Sₙ) states. 20 regexes with 10 states each = 10²⁰
potential states. This makes compilation O(2^N) — unusable beyond ~15 regexes.

## Solution: Independent DFAs in GPU Shared Memory

Each regex compiles to its own small DFA (~1-50KB). On GPU, each DFA runs in
its own workgroup with the transition table loaded into shared memory (48KB
per workgroup on NVIDIA, 64KB on AMD).

### Phase 1: Aho-Corasick Literal Pre-filter
- All literal patterns merged into one AC automaton (scales to millions)
- ONE pass through the file data
- Output: bitmap of which literal patterns matched
- 99% of files fail here → skip Phase 2 and 3

### Phase 2: Shared-Memory Regex Ensemble
- Only runs on files that passed Phase 1
- Each regex DFA in its own workgroup
- Shared memory holds transition table (fits in 48KB for most regexes)
- 256 threads per workgroup, each handles a chunk of byte positions
- Output: bitmap of which regex patterns matched

### Phase 3: vyre Condition Evaluation
- Only runs on files that matched patterns from Phase 1+2
- Evaluates temporal/proximity/scope conditions
- 63 opcodes including MatchOrder, MatchDistance, MatchBetween
- Output: which rules fired

## Performance Model

| Metric | Merged DFA | Ensemble |
|--------|-----------|----------|
| Compile time (20 regexes) | 30-60 seconds | <100ms |
| Compile time (100 regexes) | FAILS (DFA explosion) | <500ms |
| Compile time (10K regexes) | IMPOSSIBLE | ~5 seconds |
| Memory (20 regexes) | 50-500MB | 20 × 50KB = 1MB |
| Memory (10K regexes) | IMPOSSIBLE | 10K × 50KB = 500MB |
| Scan throughput | 1 pass through data | 1 pass per workgroup (GPU parallel) |
| GPU occupancy | Low (L2 thrashing on large table) | High (shared memory, no L2 pressure) |

## Shared Memory Layout (per workgroup)

```wgsl
var<workgroup> dfa_table: array<u32, 12288>;  // 48KB
var<workgroup> found_match: atomic<u32>;       // 4 bytes
```

## Workgroup Dispatch

```
dispatch_workgroups(dfa_count, 1, 1)
```

Each workgroup:
1. Load DFA metadata (offset, state_count, class_count, start_state)
2. Cooperative load: all 256 threads load DFA table into shared memory
3. workgroupBarrier()
4. Each thread scans its portion of the input (input_len / 256 bytes)
5. Walk DFA from start_state for each byte position
6. If match state reached: atomicStore(&found_match, 1)
7. workgroupBarrier()
8. Thread 0 writes found_match to global output buffer

## Fallback

If a regex DFA exceeds 48KB (complex regex with many states), it stays in
global memory. The workgroup still runs, just slower due to L2 cache misses.
This is rare — most security-relevant regexes are <100 states.
