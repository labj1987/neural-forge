//! Per-submission GPU timestamps for the layer's own capture and compose work.
//!
//! One [`GpuTimer`] belongs to one fenced slot (a capture slot or an async compose slot): a
//! two-query `VK_QUERY_TYPE_TIMESTAMP` pool written in that slot's own command buffer, and read
//! back only once the code that already polls or waits on that slot's fence has seen it
//! signalled. Nothing here ever waits: the readback passes no `WAIT` flag, and a result that is
//! somehow still `VK_NOT_READY` is dropped rather than retried.
//!
//! The measured span, for every user of this module, is from the [`GpuTimer::record_start`]
//! timestamp (written at the `TRANSFER` stage just after the command buffer's opening
//! `ALL_COMMANDS -> TRANSFER` barrier, so it lands once the queue's earlier work -- usually the
//! game's own frame -- has drained and the layer's first transfer can start) to the
//! [`GpuTimer::record_end`] timestamp (`BOTTOM_OF_PIPE`, after the last command of the
//! buffer). It is therefore the layer's own GPU time for that submission, not the time it
//! spent queued behind the game.

use ash::vk;

/// A two-query timestamp pool plus what it takes to turn its ticks into milliseconds.
pub(crate) struct GpuTimer {
    pool: vk::QueryPool,
    /// `VkPhysicalDeviceLimits::timestampPeriod`: nanoseconds per tick.
    ns_per_tick: f64,
    /// The queue family's `timestampValidBits` as a mask over the raw 64-bit values.
    valid_mask: u64,
    /// A submission that wrote both timestamps went out and has not been read back yet. A
    /// slot that never submitted (or whose last submission was read) has nothing to read:
    /// its queries hold nothing or a reading that was already published.
    pending: bool,
    /// The last reading [`Self::read_if_pending`] produced and nobody took yet.
    reading: Option<f32>,
}

impl GpuTimer {
    /// `None` when `queue_family` cannot write timestamps at all (`timestampValidBits` 0: the
    /// slot then simply goes untimed), when the device reports no usable `timestampPeriod`, or
    /// when the pool cannot be created.
    pub(crate) fn new(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue_family: u32) -> Option<Self> {
        // SAFETY: `physical_device` belongs to `instance`; both are plain property queries.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let bits = families.get(queue_family as usize)?.timestamp_valid_bits;
        if bits == 0 {
            return None;
        }
        // SAFETY: as above.
        let period = unsafe { instance.get_physical_device_properties(physical_device) }.limits.timestamp_period;
        if !period.is_finite() || period <= 0.0 {
            return None;
        }
        let info = vk::QueryPoolCreateInfo::builder().query_type(vk::QueryType::TIMESTAMP).query_count(2);
        // SAFETY: `device` is live; `info` is valid.
        let pool = unsafe { device.create_query_pool(&info, None) }.ok()?;
        Some(Self { pool, ns_per_tick: f64::from(period), valid_mask: Self::mask(bits), pending: false, reading: None })
    }

    fn mask(bits: u32) -> u64 {
        if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
    }

    /// Resets both queries and writes the start timestamp at the `TRANSFER` stage. Call right
    /// after the command buffer's opening `ALL_COMMANDS -> TRANSFER` barrier (see the module
    /// doc comment for why there and not at `TOP_OF_PIPE` before it).
    ///
    /// # Safety
    /// `cmd` must be recording, outside a render pass, on a queue family of the one this timer
    /// was built for, and its previous submission (if any) must have completed.
    pub(crate) unsafe fn record_start(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        // SAFETY: forwarded from this function's own contract; a reset followed by a write in
        // the same command buffer is ordered by submission order.
        unsafe {
            device.cmd_reset_query_pool(cmd, self.pool, 0, 2);
            device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TRANSFER, self.pool, 0);
        }
    }

    /// Writes the end timestamp at `BOTTOM_OF_PIPE`: after every earlier command of `cmd`.
    ///
    /// # Safety
    /// Same as [`Self::record_start`], which must already have been recorded into `cmd`.
    pub(crate) unsafe fn record_end(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        // SAFETY: forwarded from this function's own contract.
        unsafe { device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.pool, 1) };
    }

    /// The command buffer holding both timestamps was submitted.
    pub(crate) fn mark_submitted(&mut self) {
        self.pending = true;
    }

    /// Reads the last submission's pair, if one is pending, into [`Self::take_reading`]. Only
    /// call once that submission's fence has been seen signalled. Never waits: no `WAIT` flag,
    /// and `VK_NOT_READY` (or any error) just means no reading this time.
    pub(crate) fn read_if_pending(&mut self, device: &ash::Device) {
        if !std::mem::take(&mut self.pending) {
            return;
        }
        let mut ticks = [0u64; 2];
        // SAFETY: `self.pool` holds exactly two queries, `ticks` two u64 slots, and `TYPE_64`
        // makes each result 8 bytes at an 8-byte stride (ash passes `size_of::<u64>()`).
        let result = unsafe { device.get_query_pool_results(self.pool, 0, 2, &mut ticks, vk::QueryResultFlags::TYPE_64) };
        if crate::note_vk(result).is_err() {
            return;
        }
        let elapsed = (ticks[1] & self.valid_mask).wrapping_sub(ticks[0] & self.valid_mask) & self.valid_mask;
        let ms = (elapsed as f64 * self.ns_per_tick / 1.0e6) as f32;
        if ms.is_finite() {
            self.reading = Some(ms);
        }
    }

    /// The reading [`Self::read_if_pending`] produced since the last call, if any.
    pub(crate) fn take_reading(&mut self) -> Option<f32> {
        self.reading.take()
    }

    /// # Safety
    /// No submission that writes this pool may still be in flight: call only where the owning
    /// slot's other resources are destroyed, after their fence (or the device) was waited on.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract.
        unsafe { device.destroy_query_pool(self.pool, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::GpuTimer;

    #[test]
    fn the_valid_bits_mask_covers_exactly_those_bits() {
        assert_eq!(GpuTimer::mask(64), u64::MAX);
        assert_eq!(GpuTimer::mask(36), (1u64 << 36) - 1);
        assert_eq!(GpuTimer::mask(1), 1);
    }
}
