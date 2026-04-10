//! CPU-side metadata for the GPU literal prefilter.

/// GPU lookup table for literal prefix hashes.
///
/// Patterns are partitioned by effective prefix length
/// `min(pattern_len, hash_window_len)`. Each partition owns a power-of-two
/// bucket table with contiguous collision chains, so the shader can hash a
/// prefix once and only inspect the matching bucket.
#[derive(Debug, Clone)]
pub struct LiteralPrefilterTable {
    /// Per-prefix metadata indexed by `(prefix_len - 1)`.
    ///
    /// Each entry is `[bucket_offset, bucket_mask, bucket_count, reserved]`.
    pub prefix_meta: Vec<[u32; 4]>,
    /// Per-bucket `[entry_start, entry_count]` ranges.
    pub bucket_ranges: Vec<[u32; 2]>,
    /// Collision-chain entries `[hash, literal_index]`.
    pub entries: Vec<[u32; 2]>,
}

impl LiteralPrefilterTable {
    /// Build the lookup table for all literal patterns.
    pub fn build(
        offsets: &[(u32, u32)],
        literal_hashes: &[u32],
        hash_window_len: u32,
    ) -> Result<Self, &'static str> {
        if offsets.len() != literal_hashes.len() {
            return Err("literal prefilter: offsets/hash count mismatch. Fix: ensure every literal has a corresponding hash.");
        }

        let prefix_slots = usize::try_from(hash_window_len).map_err(|_| {
            "literal prefilter: hash window too large. Fix: reduce hash_window_len to fit in usize."
        })?;
        let mut prefix_groups = vec![Vec::<[u32; 2]>::new(); prefix_slots];
        for (literal_index, (&(_, len), &hash)) in offsets.iter().zip(literal_hashes).enumerate() {
            let prefix_len = len.min(hash_window_len).max(1);
            let group = prefix_groups
                .get_mut(prefix_len as usize - 1)
                .ok_or("prefix length out of range")?;
            let literal_index =
                u32::try_from(literal_index).map_err(|_| "literal index exceeds u32")?;
            group.push([hash, literal_index]);
        }

        let mut prefix_meta = vec![[0u32; 4]; prefix_slots];
        let mut bucket_ranges = Vec::new();
        let mut entries = Vec::with_capacity(literal_hashes.len());

        for (prefix_idx, group) in prefix_groups.into_iter().enumerate() {
            if group.is_empty() {
                continue;
            }

            let bucket_count = group
                .len()
                .checked_mul(2)
                .and_then(usize::checked_next_power_of_two)
                .ok_or("literal prefilter bucket count overflow")?;
            let bucket_count_u32 =
                u32::try_from(bucket_count).map_err(|_| "bucket count exceeds u32")?;
            let bucket_offset =
                u32::try_from(bucket_ranges.len()).map_err(|_| "bucket offset exceeds u32")?;
            let mut buckets = vec![Vec::<[u32; 2]>::new(); bucket_count];

            for entry in group {
                let bucket_idx = (entry[0] as usize) & (bucket_count - 1);
                buckets[bucket_idx].push(entry);
            }

            for bucket in buckets {
                let entry_start =
                    u32::try_from(entries.len()).map_err(|_| "entry offset exceeds u32")?;
                let entry_count =
                    u32::try_from(bucket.len()).map_err(|_| "entry count exceeds u32")?;
                bucket_ranges.push([entry_start, entry_count]);
                entries.extend(bucket);
            }

            prefix_meta[prefix_idx] = [bucket_offset, bucket_count_u32 - 1, bucket_count_u32, 0];
        }

        Ok(Self {
            prefix_meta,
            bucket_ranges,
            entries,
        })
    }

    #[cfg(test)]
    pub(crate) fn probe(&self, prefix_len: u32, hash: u32) -> Box<dyn Iterator<Item = u32> + '_> {
        let Some(meta) = prefix_len
            .checked_sub(1)
            .and_then(|idx| self.prefix_meta.get(idx as usize))
        else {
            return Box::new(std::iter::empty());
        };
        if meta[2] == 0 {
            return Box::new(std::iter::empty());
        }

        let bucket_idx = meta[0] + (hash & meta[1]);
        let Some(range) = self.bucket_ranges.get(bucket_idx as usize) else {
            return Box::new(std::iter::empty());
        };
        let start = range[0] as usize;
        let Some(end) = start.checked_add(range[1] as usize) else {
            return Box::new(std::iter::empty());
        };
        let Some(slice) = self.entries.get(start..end) else {
            return Box::new(std::iter::empty());
        };
        Box::new(
            slice
                .iter()
                .filter(move |entry| entry[0] == hash)
                .map(|entry| entry[1]),
        )
    }
}
