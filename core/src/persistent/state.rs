use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};

pub struct PersistentState {
    pub(crate) inflight: [bool; 2],
    pub(crate) pending: Option<crate::persistent::PendingWork>,
    pub(crate) use_counts: [usize; 2],
}

impl crate::persistent::PersistentMatcher {
    pub(crate) async fn wait_for_buffer_set(&self) -> usize {
        let preferred = self.active_set.fetch_add(1, Ordering::Relaxed) % self.buffer_sets.len();
        let mut backoff_us = 10_u64;
        loop {
            if let Some(set_idx) = self.try_claim_buffer_set(preferred) {
                return set_idx;
            }
            // Exponential backoff: 10µs → 20µs → 40µs → ... → 10ms cap.
            tokio::time::sleep(std::time::Duration::from_micros(backoff_us)).await;
            backoff_us = (backoff_us * 2).min(10_000);
        }
    }

    pub(crate) fn try_claim_buffer_set(&self, preferred: usize) -> Option<usize> {
        let alternate = (preferred + 1) % self.buffer_sets.len();
        let mut state = lock_state(&self.state);

        if !state.inflight[preferred] {
            state.inflight[preferred] = true;
            state.use_counts[preferred] += 1;
            return Some(preferred);
        }

        if !state.inflight[alternate] {
            state.inflight[alternate] = true;
            state.use_counts[alternate] += 1;
            return Some(alternate);
        }

        None
    }

    pub(crate) fn release_buffer_set(&self, set_idx: usize) {
        let mut state = lock_state(&self.state);
        state.inflight[set_idx] = false;
        if state
            .pending
            .as_ref()
            .is_some_and(|pending| pending.buffer_set_idx == set_idx)
        {
            state.pending = None;
        }
    }

    #[cfg(test)]
    pub(crate) fn buffer_use_counts(&self) -> [usize; 2] {
        lock_state(&self.state).use_counts
    }
}

pub(crate) fn lock_state(state: &Mutex<PersistentState>) -> MutexGuard<'_, PersistentState> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!(
                "persistent matcher state mutex was poisoned — resetting to safe defaults"
            );
            let mut guard = poisoned.into_inner();
            // Reset to known-good state: release all buffer sets, clear pending.
            // A panicking thread may have left inflight[i] = true permanently.
            guard.inflight = [false; 2];
            guard.pending = None;
            guard
        }
    }
}
