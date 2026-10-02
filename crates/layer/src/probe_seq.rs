//! The `NEURAL_FORGE_PROBE_NGX` probe's per-command-buffer recorder: the ordered sequence of
//! commands recorded into one command buffer, from `vkBeginCommandBuffer` to
//! `vkEndCommandBuffer`, compressed so that a buffer with thousands of draws stays a few dozen
//! readable lines (see `docs/PRE_UPSCALER_PROBE.md`, "Second run").
//!
//! Commands that touch an *image of interest* (one whose view was registered with
//! `VK_NVX_image_view_handle`, i.e. handed to DLSS) are kept as individual entries, and so is
//! every `vkCmdCuLaunchKernelNVX`. Everything between two kept entries is folded into one
//! [`Gap`] that only counts what kind of commands it held. Pure data: the caller decides
//! what is interesting, and nothing here touches Vulkan.

use ash::vk;
use ash::vk::Handle;

/// A cap on kept entries per command buffer; past it, further commands are only counted.
const MAX_OPS: usize = 4096;

/// Access bits that mean a write (sync2 values; the sync1 bits are the same low bits).
const WRITE_ACCESS: u64 = vk::AccessFlags2::SHADER_WRITE.as_raw()
    | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
    | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
    | vk::AccessFlags2::HOST_WRITE.as_raw()
    | vk::AccessFlags2::MEMORY_WRITE.as_raw()
    | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw();

pub(crate) fn is_write(access: u64) -> bool {
    access & WRITE_ACCESS != 0
}

fn stages(raw: u64) -> String {
    format!("{:?}", vk::PipelineStageFlags2::from_raw(raw))
}

fn access(raw: u64) -> String {
    if raw == 0 { "NONE".into() } else { format!("{:?}", vk::AccessFlags2::from_raw(raw)) }
}

/// Uninteresting commands between two kept entries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Gap {
    pub draws: u32,
    pub dispatches: u32,
    /// Copies, blits, clears and resolves that touch no image of interest.
    pub transfers: u32,
    /// Render passes and dynamic renderings with no image of interest attached.
    pub renderings: u32,
    /// Image barriers on other images.
    pub image_barriers: u32,
    pub buffer_barriers: u32,
    /// Global memory barriers, with the union of their stages and accesses.
    pub memory_barriers: u32,
    pub mem_src_stage: u64,
    pub mem_src_access: u64,
    pub mem_dst_stage: u64,
    pub mem_dst_access: u64,
    /// `vkCmdExecuteCommands` calls whose secondaries carry no launches.
    pub executes: u32,
    /// `vkCmdExecuteGeneratedCommandsNV` (device-generated draws or dispatches).
    pub generated: u32,
}

impl std::fmt::Display for Gap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        for (n, what) in [
            (self.draws, "draws"),
            (self.dispatches, "dispatches"),
            (self.transfers, "transfers(other)"),
            (self.renderings, "renderings(other)"),
            (self.image_barriers, "img-barriers(other)"),
            (self.buffer_barriers, "buf-barriers"),
            (self.executes, "executes"),
            (self.generated, "generated-cmds"),
        ] {
            if n > 0 {
                parts.push(format!("{n} {what}"));
            }
        }
        if self.memory_barriers > 0 {
            parts.push(format!(
                "{} mem-barriers(src {}:{} -> dst {}:{})",
                self.memory_barriers,
                stages(self.mem_src_stage),
                access(self.mem_src_access),
                stages(self.mem_dst_stage),
                access(self.mem_dst_access)
            ));
        }
        if parts.is_empty() { f.write_str("(nothing)") } else { f.write_str(&parts.join(", ")) }
    }
}

/// One attachment of a render pass or dynamic rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Attachment {
    pub image: vk::Image,
    /// `None` for a render pass (its layouts and load ops live in the render pass object).
    pub layout: Option<vk::ImageLayout>,
    /// `None` for a render pass.
    pub load: Option<vk::AttachmentLoadOp>,
    pub depth: bool,
}

/// An image barrier, sync1 or sync2, widened to sync2's 64-bit masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageBarrier {
    pub image: vk::Image,
    pub old: vk::ImageLayout,
    pub new: vk::ImageLayout,
    pub src_stage: u64,
    pub src_access: u64,
    pub dst_stage: u64,
    pub dst_access: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Gap(Gap),
    Launch { ordinal: u32, kernel: String },
    /// A copy/blit/clear/resolve with an image of interest as source or destination.
    Transfer { kind: &'static str, src: Option<(vk::Image, vk::ImageLayout)>, dst: Option<(vk::Image, vk::ImageLayout)> },
    /// A render pass or rendering with an image of interest attached, and the draws and
    /// `vkCmdClearAttachments` recorded inside it.
    Render { kind: &'static str, attachments: Vec<Attachment>, draws: u32, clears: u32 },
    Barrier(ImageBarrier),
    /// `vkCmdExecuteCommands` whose secondaries carry launches.
    Execute { buffers: u32, launches: u32 },
}

/// The recorded sequence of one command buffer since its last `vkBeginCommandBuffer`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Recording {
    pub flags: vk::CommandBufferUsageFlags,
    /// A process-wide counter value taken at begin: equal epochs at two submits mean the
    /// buffer was resubmitted without being re-recorded.
    pub epoch: u64,
    pub ops: Vec<Op>,
    /// Launches recorded directly (secondaries' are counted in `Execute`).
    pub launches: u32,
    /// Inside a render pass/rendering: `Some(Some(i))` when it is the kept `ops[i]`.
    render: Option<Option<usize>>,
    /// Commands not kept because of `MAX_OPS` (still counted in the last gap).
    pub overflowed: bool,
    /// What the caller knew about the colour input when the first launch was recorded
    /// (a barrier recorded in some other command buffer), for the verdict line.
    pub colour_elsewhere_at_first_launch: Option<String>,
}

impl Recording {
    pub fn new(flags: vk::CommandBufferUsageFlags, epoch: u64) -> Self {
        Self { flags, epoch, ..Self::default() }
    }

    fn gap(&mut self) -> &mut Gap {
        if !matches!(self.ops.last(), Some(Op::Gap(_))) {
            // Past the cap a new gap is still opened (at most one per launch marker), so
            // counting never stops.
            if self.ops.len() >= MAX_OPS {
                self.overflowed = true;
            }
            self.ops.push(Op::Gap(Gap::default()));
        }
        match self.ops.last_mut() {
            Some(Op::Gap(gap)) => gap,
            _ => unreachable!("a gap was just ensured"),
        }
    }

    fn keep(&mut self, op: Op) -> Option<usize> {
        if self.ops.len() >= MAX_OPS {
            self.overflowed = true;
            return None;
        }
        self.ops.push(op);
        Some(self.ops.len() - 1)
    }

    pub fn draw(&mut self) {
        if let Some(Some(i)) = self.render {
            if let Some(Op::Render { draws, .. }) = self.ops.get_mut(i) {
                *draws += 1;
                return;
            }
        }
        self.gap().draws += 1;
    }

    pub fn dispatch(&mut self) {
        self.gap().dispatches += 1;
    }

    pub fn generated(&mut self) {
        self.gap().generated += 1;
    }

    pub fn clear_attachments(&mut self) {
        if let Some(Some(i)) = self.render {
            if let Some(Op::Render { clears, .. }) = self.ops.get_mut(i) {
                *clears += 1;
                return;
            }
        }
        self.gap().transfers += 1;
    }

    pub fn transfer(&mut self, kind: &'static str, src: Option<(vk::Image, vk::ImageLayout)>, dst: Option<(vk::Image, vk::ImageLayout)>, interesting: bool) {
        if interesting {
            if self.keep(Op::Transfer { kind, src, dst }).is_none() {
                self.gap().transfers += 1;
            }
        } else {
            self.gap().transfers += 1;
        }
    }

    pub fn begin_render(&mut self, kind: &'static str, attachments: Vec<Attachment>, interesting: bool) {
        let kept = if interesting { self.keep(Op::Render { kind, attachments, draws: 0, clears: 0 }) } else { None };
        if kept.is_none() {
            self.gap().renderings += 1;
        }
        self.render = Some(kept);
    }

    pub fn end_render(&mut self) {
        self.render = None;
    }

    pub fn image_barrier(&mut self, barrier: ImageBarrier, interesting: bool) {
        if !interesting || self.keep(Op::Barrier(barrier)).is_none() {
            self.gap().image_barriers += 1;
        }
    }

    pub fn memory_barrier(&mut self, src_stage: u64, src_access: u64, dst_stage: u64, dst_access: u64) {
        let gap = self.gap();
        gap.memory_barriers += 1;
        gap.mem_src_stage |= src_stage;
        gap.mem_src_access |= src_access;
        gap.mem_dst_stage |= dst_stage;
        gap.mem_dst_access |= dst_access;
    }

    pub fn buffer_barriers(&mut self, n: u32) {
        if n > 0 {
            self.gap().buffer_barriers += n;
        }
    }

    pub fn execute(&mut self, buffers: u32, launches: u32) {
        if launches == 0 || self.keep(Op::Execute { buffers, launches }).is_none() {
            self.gap().executes += 1;
        }
    }

    pub fn launch(&mut self, kernel: String) {
        self.launches += 1;
        // Launch markers are never dropped, even past the cap: there are only a few dozen.
        self.ops.push(Op::Launch { ordinal: self.launches, kernel });
    }

    /// Launches this buffer carries, its own plus its secondaries'.
    pub fn total_launches(&self) -> u32 {
        self.launches
            + self.ops.iter().map(|op| if let Op::Execute { launches, .. } = op { *launches } else { 0 }).sum::<u32>()
    }

    /// What the commands before the first launch did to `colour`.
    pub fn analyse(&self, colour: Option<vk::Image>) -> Analysis {
        let mut a = Analysis::default();
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                Op::Launch { kernel, .. } => {
                    a.first_launch = Some((i, kernel.clone()));
                    break;
                }
                Op::Execute { launches, .. } => {
                    a.first_launch = Some((i, format!("(inside a secondary, {launches} launches)")));
                    break;
                }
                Op::Gap(gap) => {
                    a.draws_before += gap.draws;
                    a.dispatches_before += gap.dispatches + gap.generated;
                    a.renderings_before += gap.renderings;
                    a.transfers_before += gap.transfers;
                    if is_write(gap.mem_src_access) {
                        a.write_memory_barriers_before += gap.memory_barriers;
                    }
                }
                Op::Transfer { kind, src, dst } => {
                    let Some(colour) = colour else { continue };
                    if let Some((image, layout)) = dst {
                        if *image == colour {
                            a.explicit_writes.push(format!("[{i}] {kind} dst"));
                            a.usage_layout = Some((*layout, format!("[{i}] {kind} dst")));
                        }
                    }
                    if let Some((image, layout)) = src {
                        if *image == colour {
                            a.usage_layout = Some((*layout, format!("[{i}] {kind} src")));
                        }
                    }
                }
                Op::Render { kind, attachments, draws, clears } => {
                    a.draws_before += draws;
                    let Some(colour) = colour else { continue };
                    if let Some(att) = attachments.iter().find(|att| att.image == colour) {
                        let cleared = att.load == Some(vk::AttachmentLoadOp::CLEAR);
                        a.explicit_writes.push(format!(
                            "[{i}] {kind} attachment, {draws} draws{}{}",
                            if *clears > 0 { format!(", {clears} clear-attachments") } else { String::new() },
                            if cleared { ", load CLEAR" } else { "" }
                        ));
                        if let Some(layout) = att.layout {
                            a.usage_layout = Some((layout, format!("[{i}] {kind} attachment")));
                        }
                    }
                }
                Op::Barrier(b) => {
                    if Some(b.image) != colour {
                        continue;
                    }
                    a.barrier_layout = Some((b.new, i));
                    if is_write(b.src_access) {
                        a.barrier_writes.push(format!("[{i}] src access {}", access(b.src_access)));
                    }
                    if b.old != b.new {
                        a.transitions.push(format!("[{i}] {:?}->{:?}", b.old, b.new));
                    }
                }
            }
        }
        a
    }

    /// The kept entries as log lines, one per entry. `label` names an image.
    pub fn lines(&self, label: &dyn Fn(vk::Image) -> String) -> Vec<String> {
        self.ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let body = match op {
                    Op::Gap(gap) => format!("... {gap}"),
                    Op::Launch { ordinal, kernel } => format!("LAUNCH #{ordinal} {kernel}"),
                    Op::Execute { buffers, launches } => format!("EXECUTE {buffers} secondaries carrying {launches} launches"),
                    Op::Transfer { kind, src, dst } => {
                        let side = |side: &Option<(vk::Image, vk::ImageLayout)>| {
                            side.map_or_else(|| "-".to_string(), |(image, layout)| format!("{} ({layout:?})", label(image)))
                        };
                        format!("{kind} src={} dst={}", side(src), side(dst))
                    }
                    Op::Render { kind, attachments, draws, clears } => {
                        let atts: Vec<String> = attachments
                            .iter()
                            .map(|a| {
                                format!(
                                    "{}{} {}{}",
                                    if a.depth { "depth " } else { "" },
                                    label(a.image),
                                    a.layout.map_or_else(|| "(layout in render pass)".to_string(), |l| format!("{l:?}")),
                                    a.load.map_or_else(String::new, |l| format!(" load={l:?}"))
                                )
                            })
                            .collect();
                        format!("{kind} [{}] draws={draws} clear-attachments={clears}", atts.join(", "))
                    }
                    Op::Barrier(b) => format!(
                        "barrier {} {:?}->{:?} src {}:{} dst {}:{}",
                        label(b.image),
                        b.old,
                        b.new,
                        stages(b.src_stage),
                        access(b.src_access),
                        stages(b.dst_stage),
                        access(b.dst_access)
                    ),
                };
                format!("[{i}] {body}")
            })
            .collect()
    }
}

/// See [`Recording::analyse`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Analysis {
    /// Index of the first launch entry and its kernel.
    pub first_launch: Option<(usize, String)>,
    pub draws_before: u32,
    pub dispatches_before: u32,
    pub renderings_before: u32,
    pub transfers_before: u32,
    /// Global memory barriers with a write in their source access, before the first launch.
    pub write_memory_barriers_before: u32,
    /// Copies/clears/resolves into the colour input, and renderings with it attached.
    pub explicit_writes: Vec<String>,
    /// Barriers on the colour input whose source access includes a write.
    pub barrier_writes: Vec<String>,
    /// Layout transitions on the colour input.
    pub transitions: Vec<String>,
    /// The colour input's layout from the last barrier on it before the first launch.
    pub barrier_layout: Option<(vk::ImageLayout, usize)>,
    /// The layout a copy or rendering used it in, before the first launch.
    pub usage_layout: Option<(vk::ImageLayout, String)>,
}

impl Analysis {
    /// A short class for aggregation across buffers.
    pub fn class(&self) -> &'static str {
        if !self.explicit_writes.is_empty() {
            "explicit-write"
        } else if !self.barrier_writes.is_empty() {
            "barrier-write"
        } else if !self.transitions.is_empty() {
            "transition-only"
        } else {
            "untouched"
        }
    }

    pub fn layout_text(&self) -> String {
        match self.barrier_layout {
            Some((layout, i)) => format!("{layout:?} (barrier [{i}])"),
            None => "unknown, no barrier in this cmdbuf".into(),
        }
    }
}

/// A short handle for log lines.
pub(crate) fn hex<H: Handle>(handle: H) -> String {
    format!("{:#x}", handle.as_raw())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(raw: u64) -> vk::Image { vk::Image::from_raw(raw) }

    fn barrier(image: u64, old: vk::ImageLayout, new: vk::ImageLayout, src_access: vk::AccessFlags2) -> ImageBarrier {
        ImageBarrier {
            image: img(image),
            old,
            new,
            src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
            src_access: src_access.as_raw(),
            dst_stage: vk::PipelineStageFlags2::ALL_COMMANDS.as_raw(),
            dst_access: vk::AccessFlags2::SHADER_READ.as_raw(),
        }
    }

    #[test]
    fn uninteresting_commands_fold_into_one_gap_between_kept_entries() {
        let mut r = Recording::new(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, 7);
        for _ in 0..300 {
            r.draw();
        }
        r.dispatch();
        r.dispatch();
        r.memory_barrier(0x800, vk::AccessFlags2::SHADER_WRITE.as_raw(), 0x800, vk::AccessFlags2::SHADER_READ.as_raw());
        r.memory_barrier(0x400, 0, 0x800, vk::AccessFlags2::SHADER_READ.as_raw());
        r.image_barrier(barrier(0x99, vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL, vk::AccessFlags2::SHADER_WRITE), false);
        r.transfer("copy", Some((img(0x98), vk::ImageLayout::GENERAL)), Some((img(0x97), vk::ImageLayout::GENERAL)), false);
        r.begin_render("rendering", vec![], false);
        r.draw();
        r.end_render();
        r.buffer_barriers(3);
        assert_eq!(r.ops.len(), 1, "{:?}", r.ops);
        let Op::Gap(gap) = &r.ops[0] else { panic!() };
        assert_eq!((gap.draws, gap.dispatches, gap.memory_barriers, gap.image_barriers, gap.transfers, gap.renderings, gap.buffer_barriers), (301, 2, 2, 1, 1, 1, 3));
        assert_eq!(gap.mem_src_access, vk::AccessFlags2::SHADER_WRITE.as_raw());
        r.launch("k1".into());
        r.launch("k2".into());
        r.dispatch();
        assert_eq!(r.ops.len(), 4);
        assert_eq!(r.total_launches(), 2);
        let lines = r.lines(&|i| hex(i));
        assert!(lines[0].starts_with("[0] ... 301 draws, 2 dispatches, 1 transfers(other), 1 renderings(other), 1 img-barriers(other), 3 buf-barriers, 2 mem-barriers(src "), "{}", lines[0]);
        assert!(lines[0].contains("SHADER_WRITE"), "{}", lines[0]);
        assert_eq!(lines[1], "[1] LAUNCH #1 k1");
        assert_eq!(lines[3], "[3] ... 1 dispatches");
        let a = r.analyse(Some(img(0x1)));
        assert_eq!(a.first_launch, Some((1, "k1".into())));
        assert_eq!((a.draws_before, a.dispatches_before, a.write_memory_barriers_before), (301, 2, 2));
        assert_eq!(a.class(), "untouched");
        assert_eq!(a.layout_text(), "unknown, no barrier in this cmdbuf");
    }

    #[test]
    fn writes_into_the_colour_input_before_the_first_launch_are_found() {
        let colour = img(0xc0);
        let mut r = Recording::default();
        r.draw();
        r.begin_render("rendering", vec![Attachment { image: colour, layout: Some(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL), load: Some(vk::AttachmentLoadOp::LOAD), depth: false }], true);
        r.draw();
        r.draw();
        r.clear_attachments();
        r.end_render();
        r.draw();
        r.image_barrier(barrier(0xc0, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE), true);
        r.transfer("clear-color", None, Some((colour, vk::ImageLayout::GENERAL)), true);
        r.launch("cuda_engine_input_kernel".into());
        // After the first launch nothing counts.
        r.transfer("copy", None, Some((colour, vk::ImageLayout::TRANSFER_DST_OPTIMAL)), true);
        let a = r.analyse(Some(colour));
        assert_eq!(a.first_launch, Some((5, "cuda_engine_input_kernel".into())));
        assert_eq!(a.draws_before, 4, "draws inside the kept rendering count too");
        assert_eq!(a.explicit_writes, vec!["[1] rendering attachment, 2 draws, 1 clear-attachments".to_string(), "[4] clear-color dst".to_string()]);
        assert_eq!(a.barrier_writes.len(), 1);
        assert_eq!(a.transitions, vec!["[3] COLOR_ATTACHMENT_OPTIMAL->GENERAL".to_string()]);
        assert_eq!(a.layout_text(), "GENERAL (barrier [3])");
        assert_eq!(a.usage_layout.as_ref().map(|u| u.0), Some(vk::ImageLayout::GENERAL));
        assert_eq!(a.class(), "explicit-write");
        // The same buffer seen without knowing the colour input: no writes attributed.
        assert_eq!(r.analyse(None).class(), "untouched");
        let lines = r.lines(&|i| if i == colour { "C".into() } else { hex(i) });
        assert_eq!(lines[1], "[1] rendering [C COLOR_ATTACHMENT_OPTIMAL load=LOAD] draws=2 clear-attachments=1");
        assert!(lines[3].starts_with("[3] barrier C COLOR_ATTACHMENT_OPTIMAL->GENERAL src COMPUTE_SHADER:COLOR_ATTACHMENT_WRITE dst ALL_COMMANDS:SHADER_READ"), "{}", lines[3]);
    }

    #[test]
    fn a_barrier_only_write_and_secondaries_are_classified() {
        let colour = img(0xc0);
        let mut r = Recording::default();
        r.dispatch();
        r.image_barrier(barrier(0xc0, vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL, vk::AccessFlags2::SHADER_WRITE), true);
        r.execute(1, 0);
        r.execute(1, 14);
        let a = r.analyse(Some(colour));
        assert_eq!(a.class(), "barrier-write");
        assert_eq!(a.first_launch.as_ref().map(|f| f.0), Some(3));
        assert_eq!(r.total_launches(), 14);
        let mut quiet = Recording::default();
        quiet.image_barrier(barrier(0xc0, vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags2::NONE), true);
        assert_eq!(quiet.analyse(Some(colour)).class(), "transition-only");
    }

    #[test]
    fn the_cap_keeps_counting_and_never_drops_a_launch() {
        let mut r = Recording::default();
        for i in 0..(MAX_OPS + 10) {
            r.image_barrier(barrier(i as u64, vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL, vk::AccessFlags2::NONE), true);
        }
        assert!(r.overflowed);
        assert_eq!(r.ops.len(), MAX_OPS + 1, "kept entries stop at the cap, then one gap counts the rest");
        r.launch("k".into());
        assert!(matches!(r.ops.last(), Some(Op::Launch { .. })));
        r.draw();
        assert!(matches!(r.ops.last(), Some(Op::Gap(_))));
    }
}
