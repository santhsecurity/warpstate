pub(crate) use crate::algebraic::pipeline::AlgebraicState;
use crate::algebraic::shader::{ExtractUniforms, MapUniforms, ScanUniforms};
use crate::algebraic::{
    AlgebraicDfaMatcher, HYBRID_DEAD_STATE, HYBRID_ESCAPE_STATE, MAX_ALGEBRAIC_STATES, MAX_MATCHES,
    MAX_SCAN_ROUNDS,
};
use crate::dfa::RegexDFA;
use crate::error::{Error, Result};
use crate::Match;

impl AlgebraicDfaMatcher {
    pub(crate) fn upload_input(&self, state: &AlgebraicState, data: &[u8]) {
        let aligned_len = data.len() & !3;
        if aligned_len > 0 {
            self.queue
                .write_buffer(&state.input_buf, 0, &data[..aligned_len]);
        }
        let remainder = data.len() - aligned_len;
        if remainder > 0 {
            let mut stack_buf = [0u8; 4];
            stack_buf[..remainder].copy_from_slice(&data[aligned_len..]);
            self.queue
                .write_buffer(&state.input_buf, aligned_len as u64, &stack_buf);
        }
    }

    pub(crate) fn upload_uniforms(
        &self,
        state: &AlgebraicState,
        block_len: usize,
        carry_state: u32,
        block_offset: usize,
    ) -> Result<usize> {
        let input_len = u32::try_from(block_len).map_err(|_| Error::InputTooLarge {
            bytes: block_len,
            max_bytes: u32::MAX as usize,
        })?;
        let block_offset_u32 = u32::try_from(block_offset).map_err(|_| Error::InputTooLarge {
            bytes: block_offset,
            max_bytes: u32::MAX as usize,
        })?;

        // Validate round count won't exceed pre-allocated buffers
        let round_count = scan_round_count(block_len);
        if round_count > MAX_SCAN_ROUNDS {
            return Err(Error::InputTooLarge {
                bytes: block_len,
                max_bytes: (1usize << MAX_SCAN_ROUNDS).saturating_sub(1),
            });
        }

        self.queue
            .write_buffer(&state.count_buf, 0, bytemuck::cast_slice(&[0u32, 0u32]));

        let map_uniforms = MapUniforms {
            input_len,
            state_count: self.gpu_state_count,
            class_count: self.class_count,
            _padding0: self.eoi_class,
            byte_classes: self.packed_byte_classes,
        };
        self.queue
            .write_buffer(&state.map_uniform_buf, 0, bytemuck::bytes_of(&map_uniforms));

        let mut stride = 1u32;
        for scan_uniform_buf in state.scan_uniform_bufs.iter().take(round_count) {
            let scan_uniforms = ScanUniforms {
                input_len,
                state_count: self.gpu_state_count,
                stride,
                padding: 0,
            };
            self.queue
                .write_buffer(scan_uniform_buf, 0, bytemuck::bytes_of(&scan_uniforms));
            stride <<= 1;
        }

        let extract_uniforms = ExtractUniforms {
            input_len,
            state_count: self.gpu_state_count,
            carry_state,
            max_matches: MAX_MATCHES,
            block_offset: block_offset_u32,
            padding0: self.eoi_class,
            padding1: 0,
            padding2: 0,
        };
        self.queue.write_buffer(
            &state.extract_uniform_buf,
            0,
            bytemuck::bytes_of(&extract_uniforms),
        );

        Ok(round_count)
    }

    pub(crate) fn dispatch_block(
        &self,
        state: &AlgebraicState,
        block_len: usize,
        round_count: usize,
        capture_functions: bool,
    ) -> bool {
        // Early return for empty blocks to prevent underflow in tail_offset calculation
        if block_len == 0 {
            return true;
        }

        // SAFETY: block_len <= self.block_size (default 4096) which always fits u32.
        debug_assert!(u32::try_from(block_len).is_ok());
        let dispatch_len = u32::try_from(block_len).unwrap_or(u32::MAX);
        let workgroups = dispatch_len.div_ceil(crate::algebraic::shader::WORKGROUP_SIZE);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("algebraic block encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.map_pipeline);
            pass.set_bind_group(0, &state.map_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let mut output_is_a = true;
        for round_index in 0..round_count {
            let bind_group = if round_index % 2 == 0 {
                &state.scan_bind_groups_a_to_b[round_index]
            } else {
                &state.scan_bind_groups_b_to_a[round_index]
            };
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.scan_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            output_is_a = round_index % 2 == 1;
        }

        let extract_bind_group = if output_is_a {
            &state.extract_bind_group_a
        } else {
            &state.extract_bind_group_b
        };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.extract_pipeline);
            pass.set_bind_group(0, extract_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let match_buf_size = u64::from(MAX_MATCHES) * 16;
        encoder.copy_buffer_to_buffer(&state.count_buf, 0, &state.count_staging, 0, 8);
        encoder.copy_buffer_to_buffer(&state.match_buf, 0, &state.match_staging, 0, match_buf_size);

        if block_len > 0 {
            let final_buffer = if output_is_a {
                &state.func_a
            } else {
                &state.func_b
            };
            let tail_offset = ((block_len - 1)
                * self.gpu_state_count as usize
                * std::mem::size_of::<u32>()) as u64;
            let tail_size = u64::from(self.gpu_state_count) * std::mem::size_of::<u32>() as u64;
            encoder.copy_buffer_to_buffer(
                final_buffer,
                tail_offset,
                &state.tail_staging,
                0,
                tail_size,
            );
            if capture_functions {
                encoder.copy_buffer_to_buffer(
                    final_buffer,
                    0,
                    &state.func_staging,
                    0,
                    (block_len * self.gpu_state_count as usize * std::mem::size_of::<u32>()) as u64,
                );
            }
        }

        self.queue.submit(Some(encoder.finish()));
        output_is_a
    }

    pub(crate) async fn scan_hybrid(&self, data: &[u8]) -> Result<Vec<Match>> {
        let state = self.take_state().await?;
        let mut matches = Vec::new();
        let mut carry_state = self.start_state;
        let mut offset = 0usize;

        while offset < data.len() {
            let block_len = std::cmp::min(data.len() - offset, self.block_size);
            let chunk = &data[offset..offset + block_len];

            if carry_state >= MAX_ALGEBRAIC_STATES {
                carry_state = self.dfa.scan_suffix_from_state(
                    chunk,
                    carry_state,
                    offset,
                    &mut matches,
                    MAX_MATCHES as usize,
                )?;
                offset += block_len;
                continue;
            }

            self.upload_input(&state, chunk);
            let round_count = self.upload_uniforms(&state, block_len, carry_state, offset)?;
            self.dispatch_block(&state, block_len, round_count, true);

            let prefix_states = self
                .read_prefix_states(&state, block_len, carry_state)
                .await?;
            let escape_at = prefix_states
                .iter()
                .position(|&value| value == HYBRID_ESCAPE_STATE || value == HYBRID_DEAD_STATE);

            let mut block_matches = self.read_matches(&state).await?;
            if let Some(escape_pos) = escape_at {
                // Escape position is where we transitioned OUT of the algebraic region.
                // Matches ENDING at or before this position are still valid because:
                // 1. The transition to escape state happened AFTER processing byte at escape_pos
                // 2. We re-scan from escape_pos+1 onwards on CPU, handling the transition correctly
                let escape_end = offset as u32 + escape_pos as u32;
                block_matches.retain(|m| m.end <= escape_end);
            }
            matches.append(&mut block_matches);

            if let Some(escape_pos) = escape_at {
                let mut full_state = carry_state;
                for &byte in &chunk[..=escape_pos] {
                    full_state = self.dfa.transition_for_byte(full_state, byte);
                }
                let escape_end = offset as u32 + escape_pos as u32 + 1;
                if RegexDFA::is_match_state(full_state) {
                    self.dfa.collect_fixed_length_matches_at(
                        full_state,
                        escape_end as u64,
                        &mut block_matches,
                        MAX_MATCHES as usize,
                    )?;
                }
                carry_state = self.dfa.scan_suffix_from_state(
                    &chunk[escape_pos + 1..],
                    full_state,
                    offset + escape_pos + 1,
                    &mut matches,
                    MAX_MATCHES as usize,
                )?;
            } else {
                carry_state = self.read_tail_state(&state, carry_state).await?;
            }

            offset += block_len;
        }

        self.put_state(state);
        Ok(matches)
    }
}

pub(crate) fn scan_round_count(block_len: usize) -> usize {
    let mut rounds = 0usize;
    let mut stride = 1usize;
    while stride < block_len {
        rounds += 1;
        stride <<= 1;
    }
    rounds
}
