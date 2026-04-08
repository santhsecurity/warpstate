use crate::compiled_index::{CompiledPatternIndex, MAGIC, VERSION};
use crate::error::{Error, Result};
use crate::literal_prefilter::LiteralPrefilterTable;
use crate::pattern::PatternSet;

impl CompiledPatternIndex {
    /// Compile a pattern set into a serialized index image.
    pub fn build(patterns: &PatternSet) -> Result<Vec<u8>> {
        let ir = patterns.ir();
        // Pre-estimate: header + packed_bytes + offsets + names + regex patterns + prefilter + DFA tables
        let estimate = 64
            + ir.packed_bytes.len()
            + ir.offsets.len() * 8
            + ir.regex_patterns
                .iter()
                .map(|(_, pattern)| pattern.len() + 8)
                .sum::<usize>()
            + ir.regex_dfas
                .iter()
                .map(|d| d.transition_table().len() * 4 + 2048)
                .sum::<usize>();
        let mut bytes = Vec::with_capacity(estimate);
        bytes.extend_from_slice(&MAGIC);
        push_u32(&mut bytes, VERSION);
        push_u32(&mut bytes, u32::from(ir.case_insensitive));
        push_u32(&mut bytes, usize_to_u32(ir.names.len(), "pattern count")?);
        push_u32(&mut bytes, usize_to_u32(ir.offsets.len(), "literal count")?);
        push_u32(
            &mut bytes,
            usize_to_u32(ir.regex_dfas.len(), "regex dfa count")?,
        );
        push_u32(&mut bytes, ir.hash_window_len);

        push_bytes(&mut bytes, &ir.packed_bytes)?;
        push_u32_pairs(&mut bytes, &ir.offsets)?;
        push_names(&mut bytes, &ir.names)?;
        push_regex_patterns(&mut bytes, &ir.regex_patterns)?;
        push_usize_slice(&mut bytes, &ir.literal_automaton_ids)?;
        push_prefilter_table(&mut bytes, &ir.literal_prefilter_table)?;

        for dfa in &ir.regex_dfas {
            push_u32_slice(&mut bytes, dfa.transition_table())?;
            push_u32_slice(&mut bytes, dfa.match_list_pointers())?;
            push_u32_slice(&mut bytes, dfa.match_lists())?;
            push_u32_slice(&mut bytes, dfa.pattern_lengths())?;
            push_u32(&mut bytes, dfa.start_state());
            push_u32(&mut bytes, dfa.class_count());
            push_u32(&mut bytes, dfa.eoi_class());
            push_u32_array_256(&mut bytes, dfa.byte_classes());
            push_bytes(&mut bytes, dfa.native_dfa_bytes())?;
            push_usize_slice(&mut bytes, dfa.native_original_ids())?;
        }

        // Append CRC32 for integrity verification on load.
        // A single bit flip in the DFA transition table could cause false negatives
        // or false positives — at internet scale that means missed malware.
        let crc = crc32_ieee(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());

        Ok(bytes)
    }
}

/// CRC32 (IEEE 802.3 polynomial) — same as flashsieve transport format.
pub(crate) fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

pub(crate) fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_bytes(bytes: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    push_u32(bytes, usize_to_u32(payload.len(), "payload length")?);
    bytes.extend_from_slice(payload);
    Ok(())
}

fn push_u32_slice(bytes: &mut Vec<u8>, values: &[u32]) -> Result<()> {
    push_u32(bytes, usize_to_u32(values.len(), "u32 slice length")?);
    // Zero-copy: cast the u32 slice to bytes and extend in one memcpy.
    // Previous version called push_u32 per element (100K calls for large DFAs).
    bytes.extend_from_slice(bytemuck::cast_slice(values));
    Ok(())
}

fn push_u32_pairs(bytes: &mut Vec<u8>, values: &[(u32, u32)]) -> Result<()> {
    push_u32(bytes, usize_to_u32(values.len(), "pair slice length")?);
    bytes.reserve(values.len() * 8);
    for &(left, right) in values {
        bytes.extend_from_slice(&left.to_le_bytes());
        bytes.extend_from_slice(&right.to_le_bytes());
    }
    Ok(())
}

fn push_usize_slice(bytes: &mut Vec<u8>, values: &[usize]) -> Result<()> {
    push_u32(bytes, usize_to_u32(values.len(), "usize slice length")?);
    bytes.reserve(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&usize_to_u32(value, "usize value")?.to_le_bytes());
    }
    Ok(())
}

fn push_names(bytes: &mut Vec<u8>, names: &[Option<String>]) -> Result<()> {
    push_u32(bytes, usize_to_u32(names.len(), "name count")?);
    for name in names {
        match name {
            Some(name) => {
                let raw = name.as_bytes();
                push_u32(bytes, usize_to_u32(raw.len(), "name length")?);
                bytes.extend_from_slice(raw);
            }
            None => push_u32(bytes, u32::MAX),
        }
    }
    Ok(())
}

fn push_regex_patterns(bytes: &mut Vec<u8>, regex_patterns: &[(usize, String)]) -> Result<()> {
    push_u32(
        bytes,
        usize_to_u32(regex_patterns.len(), "regex pattern count")?,
    );
    for (id, pattern) in regex_patterns {
        push_u32(bytes, usize_to_u32(*id, "regex pattern id")?);
        push_bytes(bytes, pattern.as_bytes())?;
    }
    Ok(())
}

fn push_prefilter_table(bytes: &mut Vec<u8>, table: &LiteralPrefilterTable) -> Result<()> {
    push_u32(
        bytes,
        usize_to_u32(table.prefix_meta.len(), "prefix meta length")?,
    );
    for entry in &table.prefix_meta {
        for &value in entry {
            push_u32(bytes, value);
        }
    }

    push_u32(
        bytes,
        usize_to_u32(table.bucket_ranges.len(), "bucket range length")?,
    );
    for entry in &table.bucket_ranges {
        push_u32(bytes, entry[0]);
        push_u32(bytes, entry[1]);
    }

    push_u32(
        bytes,
        usize_to_u32(table.entries.len(), "prefilter entry length")?,
    );
    for entry in &table.entries {
        push_u32(bytes, entry[0]);
        push_u32(bytes, entry[1]);
    }
    Ok(())
}

fn push_u32_array_256(bytes: &mut Vec<u8>, values: &[u32; 256]) {
    bytes.extend_from_slice(bytemuck::cast_slice(values));
}

pub(crate) fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::PatternCompilationFailed {
        reason: format!("{label} exceeds the on-disk 32-bit format. Fix: split the pattern set into smaller indexes."),
    })
}
