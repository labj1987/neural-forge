//! The application's state around a hold recorded into its own command buffer ([`super::inline`]).
//!
//! The hold inside DLSS's command buffer records compute dispatches (the HDR encode, the network, the
//! decode, the edit cache's passes): they bind the layer's compute pipelines and descriptor sets and push
//! its constants. The application recording that buffer (vkd3d-proton) caches what it bound and does not
//! bind it again before its next dispatch, so after the hold the layer binds back what the application had
//! bound. Graphics state is never touched by the hold: it records no draw, binds nothing to the graphics bind
//! point and sets no dynamic state; descriptor sets and descriptor-buffer offsets are per bind point, and
//! the buffers bound by `vkCmdBindDescriptorBuffersEXT` are a binding point of their own that binding
//! descriptor sets leaves alone.
//!
//! What is tracked per command buffer, from its `vkBeginCommandBuffer` on (each only where the application
//! set it, so only that is restored):
//!
//! - the compute pipeline, or the compute shader object (`vkCmdBindShadersEXT`), whichever came last;
//! - the compute bind point's descriptor bindings, as the calls that made them, the last for each set
//!   number: `vkCmdBindDescriptorSets` (with their dynamic offsets), `vkCmdPushDescriptorSetKHR` (deep
//!   copied) and its template form (with the template's data), `vkCmdSetDescriptorBufferOffsetsEXT` and
//!   `vkCmdBindDescriptorBufferEmbeddedSamplersEXT`. Descriptor sets and descriptor-buffer offsets cannot be
//!   bound at the same time: a call of one kind drops the calls of the other, as it invalidates them;
//! - the descriptor buffers `vkCmdBindDescriptorBuffersEXT` bound last: the specification keeps them bound
//!   across `vkCmdBindDescriptorSets`, but the validation layer takes binding descriptor sets to unbind them,
//!   so they are bound again before the offsets that use them;
//! - the push constants of every push that includes the compute stage, with the layout it was made with
//!   (values are per stage, and the hold's pushes are compute ones; a push for graphics stages only is left
//!   alone by the hold). A later push covering an earlier one's stages and bytes replaces it.
//!
//! The `2` forms of the binding commands (VK_KHR_maintenance6) are taken as the same calls when they name a
//! layout and carry nothing else; `vkCmdExecuteCommands` leaves everything undefined, so tracking starts over.
//! Replayed in the order the application made the calls, they leave the bind point as the application
//! had it, including what a later call disturbed. A call this module cannot replay (an unknown structure in
//! a chain, a push without a layout) marks the buffer: no hold is recorded into it ([`State::unrestorable`]).
//!
//! The hot path is every compute bind and push of every command buffer. Nothing is noted until the device
//! first reaches a hold point inside DLSS's buffer ([`Tracker::activate`]: GTA V's buffers, which the split
//! takes, never do), and only buffers begun after that are held in (their state is complete); then the
//! state of the buffer last used on a thread is kept in a thread-local cache, so most calls take one
//! uncontended lock.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ash::vk;

use super::lock;

/// What is bound to the compute bind point's pipeline slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Compute {
    Pipeline(vk::Pipeline),
    /// `vkCmdBindShadersEXT` for the compute stage (null: unbound).
    Shader(u64),
}

/// One descriptor write of a push, deep copied.
#[derive(Clone, Debug)]
pub(crate) struct OwnedWrite {
    binding: u32,
    array_element: u32,
    count: u32,
    ty: vk::DescriptorType,
    images: Vec<vk::DescriptorImageInfo>,
    buffers: Vec<vk::DescriptorBufferInfo>,
    texel_views: Vec<vk::BufferView>,
    /// `VkWriteDescriptorSetInlineUniformBlock`'s bytes.
    inline: Option<Vec<u8>>,
    /// `VkWriteDescriptorSetAccelerationStructureKHR`'s handles.
    accel: Option<Vec<vk::AccelerationStructureKHR>>,
}

/// One binding call of the compute bind point.
#[derive(Clone, Debug)]
pub(crate) enum Binding {
    /// `vkCmdBindDescriptorSets`: sets `first..first + sets.len()`.
    Sets { layout: vk::PipelineLayout, first: u32, sets: Vec<vk::DescriptorSet>, dynamic: Vec<u32> },
    Push { layout: vk::PipelineLayout, set: u32, writes: Vec<OwnedWrite> },
    PushTemplate { template: vk::DescriptorUpdateTemplate, layout: vk::PipelineLayout, set: u32, data: Vec<u8> },
    /// `vkCmdSetDescriptorBufferOffsetsEXT`: sets `first..first + indices.len()`.
    Offsets { layout: vk::PipelineLayout, first: u32, indices: Vec<u32>, offsets: Vec<vk::DeviceSize> },
    Embedded { layout: vk::PipelineLayout, set: u32 },
}

impl Binding {
    /// The set numbers the call binds.
    fn sets(&self) -> std::ops::Range<u32> {
        match self {
            Binding::Sets { first, sets, .. } => *first..*first + sets.len() as u32,
            Binding::Offsets { first, indices, .. } => *first..*first + indices.len() as u32,
            Binding::Push { set, .. } | Binding::PushTemplate { set, .. } | Binding::Embedded { set, .. } => *set..*set + 1,
        }
    }

    /// Descriptor sets bound with `vkCmdBindDescriptorSets`: what descriptor-buffer offsets invalidate, and the
    /// other way round. Pushed sets are neither kind: a push-descriptor set is used beside either
    /// (VK_EXT_descriptor_buffer takes pushes with descriptor buffers), so neither drops it, and it drops
    /// neither; only a call binding its set number again replaces it.
    fn classic(&self) -> bool {
        matches!(self, Binding::Sets { .. })
    }

    fn buffer(&self) -> bool {
        matches!(self, Binding::Offsets { .. } | Binding::Embedded { .. })
    }
}

#[derive(Clone, Debug)]
struct Push {
    layout: vk::PipelineLayout,
    stages: vk::ShaderStageFlags,
    offset: u32,
    bytes: Vec<u8>,
}

/// One binding of `vkCmdBindDescriptorBuffersEXT`: its address, its usage, and the buffer of a push-descriptor
/// binding.
type BufferBinding = (vk::DeviceAddress, vk::BufferUsageFlags, Option<vk::Buffer>);

/// More pushes than this kept for one buffer (none covering another) and the buffer is not held in.
const MAX_PUSHES: usize = 64;
/// More binding calls than this kept for one buffer and the buffer is not held in.
const MAX_BINDINGS: usize = 64;

/// What the application bound in one command buffer, since it began recording.
#[derive(Debug, Default)]
pub(crate) struct State {
    seq: u64,
    compute: Option<(u64, Compute)>,
    /// `vkCmdBindDescriptorBuffersEXT`'s last call: each binding's address and usage, and the buffer of a
    /// push-descriptor binding (`VkDescriptorBufferBindingPushDescriptorBufferHandleEXT`).
    buffers: Option<(u64, Vec<BufferBinding>)>,
    bindings: Vec<(u64, Binding)>,
    pushes: Vec<(u64, Push)>,
    /// Why the state cannot be put back after a hold, if it cannot.
    unrestorable: Option<&'static str>,
    /// Conditional rendering is active: the hold's dispatches would be subject to it.
    conditional: bool,
    /// Noted since the buffer began recording (or since a `vkCmdExecuteCommands` in it).
    complete: bool,
}

impl State {
    fn next(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    fn clear(&mut self) {
        *self = State { complete: true, ..State::default() };
    }

    pub(crate) fn bind_compute(&mut self, c: Compute) {
        let s = self.next();
        self.compute = Some((s, c));
    }

    pub(crate) fn bind(&mut self, b: Binding) {
        let s = self.next();
        let range = b.sets();
        // The other kind is invalidated; a call whose every set number is bound again is gone.
        let (classic, buffer) = (b.classic(), b.buffer());
        self.bindings.retain(|(_, old)| {
            if (classic && old.buffer()) || (buffer && old.classic()) {
                return false;
            }
            let r = old.sets();
            !(range.start <= r.start && r.end <= range.end)
        });
        self.bindings.push((s, b));
        if self.bindings.len() > MAX_BINDINGS {
            self.unrestorable.get_or_insert("more descriptor binding calls than the layer keeps");
        }
    }

    /// `vkCmdBindDescriptorBuffersEXT`.
    ///
    /// # Safety
    /// `infos` are the application's, valid for this call.
    pub(crate) unsafe fn bind_buffers(&mut self, infos: &[vk::DescriptorBufferBindingInfoEXT]) {
        let mut out = Vec::with_capacity(infos.len());
        for i in infos {
            let mut push = None;
            let mut next = i.p_next.cast::<vk::BaseInStructure>().cast_const();
            // SAFETY: the application's chain, valid for this call (contract).
            while let Some(n) = unsafe { next.as_ref() } {
                if n.s_type == vk::StructureType::DESCRIPTOR_BUFFER_BINDING_PUSH_DESCRIPTOR_BUFFER_HANDLE_EXT {
                    // SAFETY: as above; the structure of that type.
                    push = Some(unsafe { (*next.cast::<vk::DescriptorBufferBindingPushDescriptorBufferHandleEXT>()).buffer });
                } else {
                    self.refuse("a descriptor buffer binding with a structure the layer does not copy");
                }
                next = n.p_next;
            }
            out.push((i.address, i.usage, push));
        }
        let s = self.next();
        self.buffers = Some((s, out));
    }

    pub(crate) fn push(&mut self, layout: vk::PipelineLayout, stages: vk::ShaderStageFlags, offset: u32, bytes: &[u8]) {
        if !stages.contains(vk::ShaderStageFlags::COMPUTE) {
            return;
        }
        let s = self.next();
        let end = offset + bytes.len() as u32;
        self.pushes.retain(|(_, p)| !(stages.contains(p.stages) && offset <= p.offset && p.offset + p.bytes.len() as u32 <= end));
        self.pushes.push((s, Push { layout, stages, offset, bytes: bytes.to_vec() }));
        if self.pushes.len() > MAX_PUSHES {
            self.unrestorable.get_or_insert("more push-constant ranges than the layer keeps");
        }
    }

    pub(crate) fn refuse(&mut self, why: &'static str) {
        self.unrestorable.get_or_insert(why);
    }

    pub(crate) fn conditional(&mut self, active: bool) {
        self.conditional = active;
    }

    /// Why no hold may be recorded into the buffer now, if none may.
    pub(crate) fn unrestorable(&self) -> Option<&'static str> {
        self.unrestorable.or(self.conditional.then_some("conditional rendering is active"))
    }

    /// Whether everything the application bound in the buffer was noted (it began after the tracking
    /// started).
    pub(crate) fn complete(&self) -> bool {
        self.complete
    }

    /// Binds back everything tracked, in the order the application bound it.
    ///
    /// # Safety
    /// `cb` is being recorded on this thread, outside a render pass instance; every handle tracked is still
    /// valid (the application's own, from this recording); `fns` are this device's.
    pub(crate) unsafe fn replay(&self, device: &ash::Device, fns: &Fns, cb: vk::CommandBuffer) {
        enum Item<'a> {
            Compute(Compute),
            Buffers(&'a [BufferBinding]),
            Binding(&'a Binding),
            Push(&'a Push),
        }
        let mut items: Vec<(u64, Item)> = Vec::with_capacity(2 + self.bindings.len() + self.pushes.len());
        items.extend(self.compute.map(|(s, c)| (s, Item::Compute(c))));
        items.extend(self.buffers.as_ref().map(|(s, b)| (*s, Item::Buffers(b))));
        items.extend(self.bindings.iter().map(|(s, b)| (*s, Item::Binding(b))));
        items.extend(self.pushes.iter().map(|(s, p)| (*s, Item::Push(p))));
        items.sort_by_key(|(s, _)| *s);
        let compute = vk::PipelineBindPoint::COMPUTE;
        for (_, item) in items {
            // SAFETY: the contract; each call repeats one the application made in this recording.
            unsafe {
                match item {
                    Item::Compute(Compute::Pipeline(p)) => device.cmd_bind_pipeline(cb, compute, p),
                    Item::Compute(Compute::Shader(shader)) => {
                        if let Some(bind) = fns.bind_shaders {
                            let stage = vk::ShaderStageFlags::COMPUTE;
                            bind(cb, 1, &stage, &shader);
                        }
                    }
                    Item::Buffers(buffers) => {
                        let Some(bind) = fns.bind_descriptor_buffers else { continue };
                        let handles: Vec<vk::DescriptorBufferBindingPushDescriptorBufferHandleEXT> =
                            buffers.iter().map(|(_, _, b)| vk::DescriptorBufferBindingPushDescriptorBufferHandleEXT { buffer: b.unwrap_or_default(), ..Default::default() }).collect();
                        let infos: Vec<vk::DescriptorBufferBindingInfoEXT> = buffers
                            .iter()
                            .zip(&handles)
                            .map(|((address, usage, push), handle)| vk::DescriptorBufferBindingInfoEXT {
                                p_next: if push.is_some() { std::ptr::from_ref(handle).cast_mut().cast() } else { std::ptr::null_mut() },
                                address: *address,
                                usage: *usage,
                                ..Default::default()
                            })
                            .collect();
                        bind(cb, infos.len() as u32, infos.as_ptr());
                    }
                    Item::Binding(Binding::Sets { layout, first, sets, dynamic }) => device.cmd_bind_descriptor_sets(cb, compute, *layout, *first, sets, dynamic),
                    Item::Binding(Binding::Push { layout, set, writes }) => {
                        let Some(push) = fns.push_descriptor_set else { continue };
                        let mut inline: Vec<vk::WriteDescriptorSetInlineUniformBlock> = Vec::with_capacity(writes.len());
                        let mut accel: Vec<vk::WriteDescriptorSetAccelerationStructureKHR> = Vec::with_capacity(writes.len());
                        let mut out: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(writes.len());
                        for w in writes {
                            let mut v = vk::WriteDescriptorSet {
                                dst_binding: w.binding,
                                dst_array_element: w.array_element,
                                descriptor_count: w.count,
                                descriptor_type: w.ty,
                                ..Default::default()
                            };
                            if !w.images.is_empty() {
                                v.p_image_info = w.images.as_ptr();
                            }
                            if !w.buffers.is_empty() {
                                v.p_buffer_info = w.buffers.as_ptr();
                            }
                            if !w.texel_views.is_empty() {
                                v.p_texel_buffer_view = w.texel_views.as_ptr();
                            }
                            if let Some(bytes) = &w.inline {
                                inline.push(vk::WriteDescriptorSetInlineUniformBlock { data_size: bytes.len() as u32, p_data: bytes.as_ptr().cast(), ..Default::default() });
                                v.p_next = std::ptr::from_ref(inline.last().expect("pushed")).cast();
                            } else if let Some(handles) = &w.accel {
                                accel.push(vk::WriteDescriptorSetAccelerationStructureKHR {
                                    acceleration_structure_count: handles.len() as u32,
                                    p_acceleration_structures: handles.as_ptr(),
                                    ..Default::default()
                                });
                                v.p_next = std::ptr::from_ref(accel.last().expect("pushed")).cast();
                            }
                            out.push(v);
                        }
                        push(cb, compute, *layout, *set, out.len() as u32, out.as_ptr());
                    }
                    Item::Binding(Binding::PushTemplate { template, layout, set, data }) => {
                        if let Some(push) = fns.push_descriptor_set_with_template {
                            push(cb, *template, *layout, *set, data.as_ptr().cast());
                        }
                    }
                    Item::Binding(Binding::Offsets { layout, first, indices, offsets }) => {
                        if let Some(set) = fns.set_descriptor_buffer_offsets {
                            set(cb, compute, *layout, *first, indices.len() as u32, indices.as_ptr(), offsets.as_ptr());
                        }
                    }
                    Item::Binding(Binding::Embedded { layout, set }) => {
                        if let Some(bind) = fns.bind_descriptor_buffer_embedded_samplers {
                            bind(cb, compute, *layout, *set);
                        }
                    }
                    Item::Push(p) => device.cmd_push_constants(cb, p.layout, p.stages, p.offset, &p.bytes),
                }
            }
        }
    }

    /// Whether replaying needs a function `fns` lacks (the application used it, so the device has it; a
    /// lookup that failed is a reason not to hold).
    pub(crate) fn needs_missing(&self, fns: &Fns) -> bool {
        let shader = matches!(self.compute, Some((_, Compute::Shader(_)))) && fns.bind_shaders.is_none();
        shader
            || (self.buffers.is_some() && fns.bind_descriptor_buffers.is_none())
            || self.bindings.iter().any(|(_, b)| match b {
                Binding::Sets { .. } => false,
                Binding::Push { .. } => fns.push_descriptor_set.is_none(),
                Binding::PushTemplate { .. } => fns.push_descriptor_set_with_template.is_none(),
                Binding::Offsets { .. } => fns.set_descriptor_buffer_offsets.is_none(),
                Binding::Embedded { .. } => fns.bind_descriptor_buffer_embedded_samplers.is_none(),
            })
    }
}

/// `vkCmdBindShadersEXT` (newer than the pinned ash).
pub(crate) type PfnCmdBindShadersEXT = unsafe extern "system" fn(vk::CommandBuffer, u32, *const vk::ShaderStageFlags, *const u64);

/// The next layer's binding commands that are not core 1.0, for the replay (`None` where the device has none).
#[derive(Clone, Copy, Default)]
pub(crate) struct Fns {
    pub push_descriptor_set: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
    pub push_descriptor_set_with_template: Option<vk::PFN_vkCmdPushDescriptorSetWithTemplateKHR>,
    pub set_descriptor_buffer_offsets: Option<vk::PFN_vkCmdSetDescriptorBufferOffsetsEXT>,
    pub bind_descriptor_buffer_embedded_samplers: Option<vk::PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT>,
    pub bind_descriptor_buffers: Option<vk::PFN_vkCmdBindDescriptorBuffersEXT>,
    pub bind_shaders: Option<PfnCmdBindShadersEXT>,
}

impl std::fmt::Debug for Fns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fns").finish_non_exhaustive()
    }
}

/// Deep copies a push's writes, or says why it cannot.
///
/// # Safety
/// `writes` are valid `VkWriteDescriptorSet`s (the application's, for this call).
pub(crate) unsafe fn own_writes(writes: &[vk::WriteDescriptorSet]) -> Result<Vec<OwnedWrite>, &'static str> {
    let mut out = Vec::with_capacity(writes.len());
    for w in writes {
        let n = w.descriptor_count as usize;
        let mut own = OwnedWrite {
            binding: w.dst_binding,
            array_element: w.dst_array_element,
            count: w.descriptor_count,
            ty: w.descriptor_type,
            images: Vec::new(),
            buffers: Vec::new(),
            texel_views: Vec::new(),
            inline: None,
            accel: None,
        };
        // SAFETY (each arm): the array the descriptor type uses holds `descriptorCount` elements.
        unsafe {
            match w.descriptor_type {
                vk::DescriptorType::SAMPLER
                | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                | vk::DescriptorType::SAMPLED_IMAGE
                | vk::DescriptorType::STORAGE_IMAGE
                | vk::DescriptorType::INPUT_ATTACHMENT => {
                    if n > 0 && w.p_image_info.is_null() {
                        return Err("a pushed image descriptor without its image infos");
                    }
                    own.images = std::slice::from_raw_parts(w.p_image_info, n).to_vec();
                }
                vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    if n > 0 && w.p_buffer_info.is_null() {
                        return Err("a pushed buffer descriptor without its buffer infos");
                    }
                    own.buffers = std::slice::from_raw_parts(w.p_buffer_info, n).to_vec();
                }
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER | vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                    if n > 0 && w.p_texel_buffer_view.is_null() {
                        return Err("a pushed texel buffer descriptor without its views");
                    }
                    own.texel_views = std::slice::from_raw_parts(w.p_texel_buffer_view, n).to_vec();
                }
                vk::DescriptorType::INLINE_UNIFORM_BLOCK | vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {}
                _ => return Err("a pushed descriptor of a type the layer does not copy"),
            }
            let mut next = w.p_next.cast::<vk::BaseInStructure>();
            while let Some(s) = next.as_ref() {
                match s.s_type {
                    vk::StructureType::WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK => {
                        let b = &*next.cast::<vk::WriteDescriptorSetInlineUniformBlock>();
                        own.inline = Some(std::slice::from_raw_parts(b.p_data.cast::<u8>(), b.data_size as usize).to_vec());
                    }
                    vk::StructureType::WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR => {
                        let a = &*next.cast::<vk::WriteDescriptorSetAccelerationStructureKHR>();
                        own.accel = Some(std::slice::from_raw_parts(a.p_acceleration_structures, a.acceleration_structure_count as usize).to_vec());
                    }
                    _ => return Err("a pushed descriptor write with a structure the layer does not copy"),
                }
                next = s.p_next;
            }
        }
        if (w.descriptor_type == vk::DescriptorType::INLINE_UNIFORM_BLOCK && own.inline.is_none())
            || (w.descriptor_type == vk::DescriptorType::ACCELERATION_STRUCTURE_KHR && own.accel.is_none())
        {
            return Err("a pushed descriptor without the structure its type needs");
        }
        out.push(own);
    }
    Ok(out)
}

/// The bytes a template's data spans (`None` for a descriptor type it cannot size).
pub(crate) fn template_bytes(entries: &[vk::DescriptorUpdateTemplateEntry]) -> Option<usize> {
    let mut end = 0usize;
    for e in entries {
        if e.descriptor_count == 0 {
            continue;
        }
        let size = match e.descriptor_type {
            vk::DescriptorType::SAMPLER | vk::DescriptorType::COMBINED_IMAGE_SAMPLER | vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::STORAGE_IMAGE | vk::DescriptorType::INPUT_ATTACHMENT => {
                std::mem::size_of::<vk::DescriptorImageInfo>()
            }
            vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                std::mem::size_of::<vk::DescriptorBufferInfo>()
            }
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER | vk::DescriptorType::STORAGE_TEXEL_BUFFER => std::mem::size_of::<vk::BufferView>(),
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => std::mem::size_of::<vk::AccelerationStructureKHR>(),
            // The count is in bytes, at the offset; no stride.
            vk::DescriptorType::INLINE_UNIFORM_BLOCK => {
                end = end.max(e.offset + e.descriptor_count as usize);
                continue;
            }
            _ => return None,
        };
        end = end.max(e.offset + (e.descriptor_count as usize - 1) * e.stride + size);
    }
    Some(end)
}

/// One device's tracking: each command buffer's [`State`], and the push templates' sizes.
pub(crate) struct Tracker {
    /// Unique in the process (a thread's cached state names its tracker by it).
    id: u64,
    /// Noting at all ([`Self::activate`]).
    active: std::sync::atomic::AtomicBool,
    /// Never removed (a buffer freed and a new one with the same handle start over at their
    /// `vkBeginCommandBuffer`): a thread's cached entry stays the buffer's.
    states: Mutex<HashMap<vk::CommandBuffer, Arc<Mutex<State>>>>,
    /// Each push-descriptor template's data size, compute bind point only (`None`: not sizable).
    templates: Mutex<HashMap<vk::DescriptorUpdateTemplate, Option<usize>>>,
    pub(crate) fns: Fns,
}

/// A thread's last state: its tracker ([`Tracker::id`]), command buffer and state.
type Last = (u64, vk::CommandBuffer, Arc<Mutex<State>>);

thread_local! {
    /// The state this thread used last: its tracker, command buffer and state.
    static LAST: std::cell::RefCell<Option<Last>> = const { std::cell::RefCell::new(None) };
}

/// The next [`Tracker::id`].
static NEXT_TRACKER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Tracker {
    pub(crate) fn new(fns: Fns) -> Self {
        Self {
            id: NEXT_TRACKER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            active: std::sync::atomic::AtomicBool::new(false),
            states: Mutex::default(),
            templates: Mutex::default(),
            fns,
        }
    }

    /// Whether the application's binds are noted (the hooks return at once otherwise).
    pub(crate) fn active(&self) -> bool {
        self.active.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Starts noting (the device reached a hold point inside DLSS's buffer). `true` when it already was.
    pub(crate) fn activate(&self) -> bool {
        self.active.swap(true, std::sync::atomic::Ordering::Relaxed)
    }

    fn state(&self, cb: vk::CommandBuffer) -> Arc<Mutex<State>> {
        let me = self.id;
        LAST.with(|last| {
            let mut last = last.borrow_mut();
            if let Some((t, c, s)) = last.as_ref() {
                if *t == me && *c == cb {
                    return s.clone();
                }
            }
            let s = lock(&self.states).entry(cb).or_default().clone();
            *last = Some((me, cb, s.clone()));
            s
        })
    }

    /// Runs `f` on `cb`'s state while noting is on ([`Self::active`]).
    pub(crate) fn note(&self, cb: vk::CommandBuffer, f: impl FnOnce(&mut State)) {
        if self.active() {
            self.with(cb, f);
        }
    }

    /// Runs `f` on `cb`'s state.
    pub(crate) fn with<R>(&self, cb: vk::CommandBuffer, f: impl FnOnce(&mut State) -> R) -> R {
        let s = self.state(cb);
        let mut g = lock(&s);
        f(&mut g)
    }

    /// `vkBeginCommandBuffer`, `vkResetCommandBuffer`, `vkFreeCommandBuffers`, a pool reset, or
    /// `vkCmdExecuteCommands` (after which every state of the primary is undefined): nothing is bound.
    pub(crate) fn clear(&self, cb: vk::CommandBuffer) {
        if self.active() {
            self.with(cb, State::clear);
        }
    }

    /// `vkFreeCommandBuffers` or a pool destroyed: nothing is bound in `cb` any more, and nothing is noted
    /// until it begins again.
    pub(crate) fn forget(&self, cb: vk::CommandBuffer) {
        if let Some(s) = lock(&self.states).get(&cb).cloned() {
            *lock(&s) = State::default();
        }
    }

    /// Templates are noted from the device's creation on, active or not: they outlive command buffers.
    pub(crate) fn template_created(&self, template: vk::DescriptorUpdateTemplate, info: &vk::DescriptorUpdateTemplateCreateInfo) {
        if info.template_type != vk::DescriptorUpdateTemplateType::PUSH_DESCRIPTORS_KHR {
            return;
        }
        let size = if info.pipeline_bind_point != vk::PipelineBindPoint::COMPUTE || info.descriptor_update_entry_count == 0 || info.p_descriptor_update_entries.is_null() {
            Some(0)
        } else {
            // SAFETY: the application's create info, valid for this call.
            template_bytes(unsafe { std::slice::from_raw_parts(info.p_descriptor_update_entries, info.descriptor_update_entry_count as usize) })
        };
        lock(&self.templates).insert(template, size);
    }

    pub(crate) fn template_destroyed(&self, template: vk::DescriptorUpdateTemplate) {
        lock(&self.templates).remove(&template);
    }

    /// `vkCmdPushDescriptorSetWithTemplateKHR` (compute templates only matter).
    ///
    /// # Safety
    /// `data` holds the template's data.
    pub(crate) unsafe fn push_template(&self, cb: vk::CommandBuffer, template: vk::DescriptorUpdateTemplate, layout: vk::PipelineLayout, set: u32, data: *const std::ffi::c_void) {
        if !self.active() {
            return;
        }
        let size = lock(&self.templates).get(&template).copied();
        self.with(cb, |s| match size {
            // Another bind point's template: nothing of the compute state.
            Some(Some(0)) => {}
            Some(Some(n)) if !data.is_null() => {
                // SAFETY: the template's data spans `n` bytes (contract).
                let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), n) }.to_vec();
                s.bind(Binding::PushTemplate { template, layout, set, data: bytes });
            }
            _ => s.refuse("a push-descriptor template the layer cannot size"),
        });
    }
}

// ---- The binding commands newer than the layer framework ----------------------------------------
//
// The pinned framework hands out the next layer's pointers for commands it does not know. Those that bind
// compute state (VK_KHR_maintenance6's `2` forms, their core 1.4 names, the core 1.4 names of the push
// descriptor commands, and `vkCmdBindShadersEXT`) are wrapped here, for devices whose tracking registered
// itself ([`register`]): each notes the call in the command buffer's [`State`] and forwards it unchanged.

/// The structures of the `2` binding commands (VK_KHR_maintenance6; newer than the pinned ash).
#[repr(C)]
struct BindDescriptorSetsInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    stage_flags: vk::ShaderStageFlags,
    layout: vk::PipelineLayout,
    first_set: u32,
    descriptor_set_count: u32,
    p_descriptor_sets: *const vk::DescriptorSet,
    dynamic_offset_count: u32,
    p_dynamic_offsets: *const u32,
}

#[repr(C)]
struct PushConstantsInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    layout: vk::PipelineLayout,
    stage_flags: vk::ShaderStageFlags,
    offset: u32,
    size: u32,
    p_values: *const std::ffi::c_void,
}

#[repr(C)]
struct PushDescriptorSetInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    stage_flags: vk::ShaderStageFlags,
    layout: vk::PipelineLayout,
    set: u32,
    descriptor_write_count: u32,
    p_descriptor_writes: *const vk::WriteDescriptorSet,
}

#[repr(C)]
struct PushDescriptorSetWithTemplateInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    descriptor_update_template: vk::DescriptorUpdateTemplate,
    layout: vk::PipelineLayout,
    set: u32,
    p_data: *const std::ffi::c_void,
}

#[repr(C)]
struct SetDescriptorBufferOffsetsInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    stage_flags: vk::ShaderStageFlags,
    layout: vk::PipelineLayout,
    first_set: u32,
    set_count: u32,
    p_buffer_indices: *const u32,
    p_offsets: *const vk::DeviceSize,
}

#[repr(C)]
struct BindDescriptorBufferEmbeddedSamplersInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    stage_flags: vk::ShaderStageFlags,
    layout: vk::PipelineLayout,
    set: u32,
}

type Pfn2<T> = unsafe extern "system" fn(vk::CommandBuffer, *const T);
type PfnPushDescriptorSet = vk::PFN_vkCmdPushDescriptorSetKHR;
type PfnPushDescriptorSetWithTemplate = vk::PFN_vkCmdPushDescriptorSetWithTemplateKHR;

/// The next layer's pointers for the wrapped commands of one device, as handed out.
#[derive(Default)]
struct Next {
    bind_descriptor_sets2: Option<Pfn2<BindDescriptorSetsInfo>>,
    push_constants2: Option<Pfn2<PushConstantsInfo>>,
    push_descriptor_set2: Option<Pfn2<PushDescriptorSetInfo>>,
    push_descriptor_set_with_template2: Option<Pfn2<PushDescriptorSetWithTemplateInfo>>,
    set_descriptor_buffer_offsets2: Option<Pfn2<SetDescriptorBufferOffsetsInfo>>,
    bind_descriptor_buffer_embedded_samplers2: Option<Pfn2<BindDescriptorBufferEmbeddedSamplersInfo>>,
    push_descriptor_set: Option<PfnPushDescriptorSet>,
    push_descriptor_set_with_template: Option<PfnPushDescriptorSetWithTemplate>,
    bind_shaders: Option<PfnCmdBindShadersEXT>,
}

struct Wrapped {
    key: usize,
    tracker: Arc<Tracker>,
    next: Mutex<Next>,
}

/// The devices whose tracking is on, by the loader's dispatch key (a command buffer's is its device's).
static WRAPPED: Mutex<Vec<Arc<Wrapped>>> = Mutex::new(Vec::new());

/// The loader's dispatch key of a dispatchable handle: its first pointer-sized word.
///
/// # Safety
/// `raw` is a live dispatchable handle.
unsafe fn key(raw: u64) -> usize {
    // SAFETY: every dispatchable object starts with the loader's dispatch pointer (contract).
    unsafe { *(raw as *const usize) }
}

/// Starts wrapping the newer binding commands for `device`, noting them in `tracker`.
///
/// # Safety
/// `device` is live.
pub(crate) unsafe fn register(device: vk::Device, tracker: Arc<Tracker>) {
    use ash::vk::Handle;
    // SAFETY: forwarded.
    let key = unsafe { key(device.as_raw()) };
    let mut w = lock(&WRAPPED);
    w.retain(|e| e.key != key);
    w.push(Arc::new(Wrapped { key, tracker, next: Mutex::new(Next::default()) }));
}

/// Stops wrapping for `device` (its destruction).
///
/// # Safety
/// `device` is live.
pub(crate) unsafe fn unregister(device: vk::Device) {
    use ash::vk::Handle;
    // SAFETY: forwarded.
    let key = unsafe { key(device.as_raw()) };
    lock(&WRAPPED).retain(|e| e.key != key);
}

fn wrapped_for(key: usize) -> Option<Arc<Wrapped>> {
    lock(&WRAPPED).iter().find(|e| e.key == key).cloned()
}

/// The device's entry for command buffer `cb`.
///
/// # Safety
/// `cb` is live.
unsafe fn of(cb: vk::CommandBuffer) -> Option<Arc<Wrapped>> {
    use ash::vk::Handle;
    // SAFETY: forwarded.
    wrapped_for(unsafe { key(cb.as_raw()) })
}

/// `vkGetDeviceProcAddr`'s answer for `name` on `device`: one of the wrapped commands' wrappers when the
/// device's tracking is on and the next layer has the command, else `original` unchanged.
///
/// # Safety
/// `device` is live; `original` is the next layer's pointer for `name`.
pub(crate) unsafe fn intercept(device: vk::Device, name: &[u8], original: vk::PFN_vkVoidFunction) -> vk::PFN_vkVoidFunction {
    use ash::vk::Handle;
    let next = original?;
    // SAFETY: forwarded.
    let Some(w) = wrapped_for(unsafe { key(device.as_raw()) }) else { return original };
    let mut n = lock(&w.next);
    // SAFETY (each arm): the next layer's pointer for that command name, which has that signature.
    unsafe {
        let wrap = |f: *const ()| Some(std::mem::transmute::<*const (), unsafe extern "system" fn()>(f));
        match name {
            b"vkCmdBindDescriptorSets2" | b"vkCmdBindDescriptorSets2KHR" => {
                n.bind_descriptor_sets2 = Some(std::mem::transmute::<unsafe extern "system" fn(), Pfn2<BindDescriptorSetsInfo>>(next));
                wrap(bind_descriptor_sets2 as *const ())
            }
            b"vkCmdPushConstants2" | b"vkCmdPushConstants2KHR" => {
                n.push_constants2 = Some(std::mem::transmute::<unsafe extern "system" fn(), Pfn2<PushConstantsInfo>>(next));
                wrap(push_constants2 as *const ())
            }
            b"vkCmdPushDescriptorSet2" | b"vkCmdPushDescriptorSet2KHR" => {
                n.push_descriptor_set2 = Some(std::mem::transmute::<unsafe extern "system" fn(), Pfn2<PushDescriptorSetInfo>>(next));
                wrap(push_descriptor_set2 as *const ())
            }
            b"vkCmdPushDescriptorSetWithTemplate2" | b"vkCmdPushDescriptorSetWithTemplate2KHR" => {
                n.push_descriptor_set_with_template2 = Some(std::mem::transmute::<unsafe extern "system" fn(), Pfn2<PushDescriptorSetWithTemplateInfo>>(next));
                wrap(push_descriptor_set_with_template2 as *const ())
            }
            b"vkCmdSetDescriptorBufferOffsets2EXT" => {
                n.set_descriptor_buffer_offsets2 = Some(std::mem::transmute::<unsafe extern "system" fn(), Pfn2<SetDescriptorBufferOffsetsInfo>>(next));
                wrap(set_descriptor_buffer_offsets2 as *const ())
            }
            b"vkCmdBindDescriptorBufferEmbeddedSamplers2EXT" => {
                n.bind_descriptor_buffer_embedded_samplers2 = Some(std::mem::transmute::<unsafe extern "system" fn(), Pfn2<BindDescriptorBufferEmbeddedSamplersInfo>>(next));
                wrap(bind_descriptor_buffer_embedded_samplers2 as *const ())
            }
            // The core 1.4 names of the push-descriptor commands (the framework hooks only the KHR ones).
            b"vkCmdPushDescriptorSet" => {
                n.push_descriptor_set = Some(std::mem::transmute::<unsafe extern "system" fn(), PfnPushDescriptorSet>(next));
                wrap(push_descriptor_set as *const ())
            }
            b"vkCmdPushDescriptorSetWithTemplate" => {
                n.push_descriptor_set_with_template = Some(std::mem::transmute::<unsafe extern "system" fn(), PfnPushDescriptorSetWithTemplate>(next));
                wrap(push_descriptor_set_with_template as *const ())
            }
            b"vkCmdBindShadersEXT" => {
                n.bind_shaders = Some(std::mem::transmute::<unsafe extern "system" fn(), PfnCmdBindShadersEXT>(next));
                wrap(bind_shaders as *const ())
            }
            _ => original,
        }
    }
}

/// The wrapper's device entry and its next pointer for one command; `None` (nothing called) only when the
/// device is gone, which the application may not do while recording.
unsafe fn next_of<T: Copy>(cb: vk::CommandBuffer, pick: impl Fn(&Next) -> Option<T>) -> Option<(Arc<Wrapped>, T)> {
    // SAFETY: forwarded.
    let w = unsafe { of(cb) }?;
    let f = pick(&lock(&w.next))?;
    Some((w, f))
}

unsafe extern "system" fn bind_descriptor_sets2(cb: vk::CommandBuffer, info: *const BindDescriptorSetsInfo) {
    // SAFETY: the application's call: `cb` is recording and `info` valid.
    unsafe {
        let Some((w, next)) = next_of(cb, |n| n.bind_descriptor_sets2) else { return };
        if let Some(i) = info.as_ref().filter(|i| i.stage_flags.contains(vk::ShaderStageFlags::COMPUTE)) {
            w.tracker.note(cb, |s| {
                if !i.p_next.is_null() || i.layout == vk::PipelineLayout::null() {
                    s.refuse("vkCmdBindDescriptorSets2 with a chained structure");
                } else {
                    let sets = slice(i.p_descriptor_sets, i.descriptor_set_count).to_vec();
                    let dynamic = slice(i.p_dynamic_offsets, i.dynamic_offset_count).to_vec();
                    s.bind(Binding::Sets { layout: i.layout, first: i.first_set, sets, dynamic });
                }
            });
        }
        next(cb, info);
    }
}

unsafe extern "system" fn push_constants2(cb: vk::CommandBuffer, info: *const PushConstantsInfo) {
    // SAFETY: as above.
    unsafe {
        let Some((w, next)) = next_of(cb, |n| n.push_constants2) else { return };
        if let Some(i) = info.as_ref().filter(|i| i.stage_flags.contains(vk::ShaderStageFlags::COMPUTE)) {
            w.tracker.note(cb, |s| {
                if !i.p_next.is_null() || i.layout == vk::PipelineLayout::null() {
                    s.refuse("vkCmdPushConstants2 with a chained structure");
                } else {
                    s.push(i.layout, i.stage_flags, i.offset, slice(i.p_values.cast::<u8>(), i.size));
                }
            });
        }
        next(cb, info);
    }
}

unsafe extern "system" fn push_descriptor_set2(cb: vk::CommandBuffer, info: *const PushDescriptorSetInfo) {
    // SAFETY: as above.
    unsafe {
        let Some((w, next)) = next_of(cb, |n| n.push_descriptor_set2) else { return };
        if let Some(i) = info.as_ref().filter(|i| i.stage_flags.contains(vk::ShaderStageFlags::COMPUTE)) {
            w.tracker.note(cb, |s| {
                if !i.p_next.is_null() || i.layout == vk::PipelineLayout::null() {
                    s.refuse("vkCmdPushDescriptorSet2 with a chained structure");
                    return;
                }
                match own_writes(slice(i.p_descriptor_writes, i.descriptor_write_count)) {
                    Ok(writes) => s.bind(Binding::Push { layout: i.layout, set: i.set, writes }),
                    Err(why) => s.refuse(why),
                }
            });
        }
        next(cb, info);
    }
}

unsafe extern "system" fn push_descriptor_set_with_template2(cb: vk::CommandBuffer, info: *const PushDescriptorSetWithTemplateInfo) {
    // SAFETY: as above.
    unsafe {
        let Some((w, next)) = next_of(cb, |n| n.push_descriptor_set_with_template2) else { return };
        if let Some(i) = info.as_ref() {
            if !i.p_next.is_null() || i.layout == vk::PipelineLayout::null() {
                w.tracker.note(cb, |s| s.refuse("vkCmdPushDescriptorSetWithTemplate2 with a chained structure"));
            } else {
                w.tracker.push_template(cb, i.descriptor_update_template, i.layout, i.set, i.p_data);
            }
        }
        next(cb, info);
    }
}

unsafe extern "system" fn set_descriptor_buffer_offsets2(cb: vk::CommandBuffer, info: *const SetDescriptorBufferOffsetsInfo) {
    // SAFETY: as above.
    unsafe {
        let Some((w, next)) = next_of(cb, |n| n.set_descriptor_buffer_offsets2) else { return };
        if let Some(i) = info.as_ref().filter(|i| i.stage_flags.contains(vk::ShaderStageFlags::COMPUTE)) {
            w.tracker.note(cb, |s| {
                if !i.p_next.is_null() || i.layout == vk::PipelineLayout::null() {
                    s.refuse("vkCmdSetDescriptorBufferOffsets2EXT with a chained structure");
                } else {
                    let indices = slice(i.p_buffer_indices, i.set_count).to_vec();
                    let offsets = slice(i.p_offsets, i.set_count).to_vec();
                    s.bind(Binding::Offsets { layout: i.layout, first: i.first_set, indices, offsets });
                }
            });
        }
        next(cb, info);
    }
}

unsafe extern "system" fn bind_descriptor_buffer_embedded_samplers2(cb: vk::CommandBuffer, info: *const BindDescriptorBufferEmbeddedSamplersInfo) {
    // SAFETY: as above.
    unsafe {
        let Some((w, next)) = next_of(cb, |n| n.bind_descriptor_buffer_embedded_samplers2) else { return };
        if let Some(i) = info.as_ref().filter(|i| i.stage_flags.contains(vk::ShaderStageFlags::COMPUTE)) {
            w.tracker.note(cb, |s| {
                if !i.p_next.is_null() || i.layout == vk::PipelineLayout::null() {
                    s.refuse("vkCmdBindDescriptorBufferEmbeddedSamplers2EXT with a chained structure");
                } else {
                    s.bind(Binding::Embedded { layout: i.layout, set: i.set });
                }
            });
        }
        next(cb, info);
    }
}

unsafe extern "system" fn push_descriptor_set(cb: vk::CommandBuffer, bind_point: vk::PipelineBindPoint, layout: vk::PipelineLayout, set: u32, count: u32, writes: *const vk::WriteDescriptorSet) {
    // SAFETY: as above.
    unsafe {
        let Some((w, next)) = next_of(cb, |n| n.push_descriptor_set) else { return };
        if bind_point == vk::PipelineBindPoint::COMPUTE {
            w.tracker.note_push_descriptor_set(cb, layout, set, slice(writes, count));
        }
        next(cb, bind_point, layout, set, count, writes);
    }
}

unsafe extern "system" fn push_descriptor_set_with_template(cb: vk::CommandBuffer, template: vk::DescriptorUpdateTemplate, layout: vk::PipelineLayout, set: u32, data: *const std::ffi::c_void) {
    // SAFETY: as above.
    unsafe {
        let Some((w, next)) = next_of(cb, |n| n.push_descriptor_set_with_template) else { return };
        w.tracker.push_template(cb, template, layout, set, data);
        next(cb, template, layout, set, data);
    }
}

unsafe extern "system" fn bind_shaders(cb: vk::CommandBuffer, count: u32, stages: *const vk::ShaderStageFlags, shaders: *const u64) {
    // SAFETY: as above.
    unsafe {
        let Some((w, next)) = next_of(cb, |n| n.bind_shaders) else { return };
        let st = slice(stages, count);
        if let Some(k) = st.iter().position(|s| *s == vk::ShaderStageFlags::COMPUTE) {
            let shader = if shaders.is_null() { 0 } else { *shaders.add(k) };
            w.tracker.note(cb, |s| s.bind_compute(Compute::Shader(shader)));
        }
        next(cb, count, stages, shaders);
    }
}

/// `count` elements at `ptr` (empty for a null pointer or a count of 0).
///
/// # Safety
/// A non-null `ptr` points to `count` elements.
unsafe fn slice<'a, T>(ptr: *const T, count: u32) -> &'a [T] {
    if ptr.is_null() || count == 0 {
        &[]
    } else {
        // SAFETY: the contract.
        unsafe { std::slice::from_raw_parts(ptr, count as usize) }
    }
}

impl Tracker {
    /// `vkCmdPushDescriptorSetKHR` on the compute bind point.
    ///
    /// # Safety
    /// `writes` are the application's, valid for this call.
    pub(crate) unsafe fn note_push_descriptor_set(&self, cb: vk::CommandBuffer, layout: vk::PipelineLayout, set: u32, writes: &[vk::WriteDescriptorSet]) {
        if !self.active() {
            return;
        }
        // SAFETY: forwarded.
        let owned = unsafe { own_writes(writes) };
        self.with(cb, |s| match owned {
            Ok(writes) => s.bind(Binding::Push { layout, set, writes }),
            Err(why) => s.refuse(why),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn set(raw: u64) -> vk::DescriptorSet {
        vk::DescriptorSet::from_raw(raw)
    }

    fn layout(raw: u64) -> vk::PipelineLayout {
        vk::PipelineLayout::from_raw(raw)
    }

    #[test]
    fn a_call_rebinding_every_set_of_an_earlier_one_replaces_it_and_the_other_kind_is_dropped() {
        let mut s = State::default();
        s.bind(Binding::Sets { layout: layout(1), first: 0, sets: vec![set(10), set(11)], dynamic: vec![] });
        s.bind(Binding::Sets { layout: layout(1), first: 1, sets: vec![set(12)], dynamic: vec![4] });
        assert_eq!(s.bindings.len(), 2, "set 0 of the first call is still bound");
        s.bind(Binding::Sets { layout: layout(1), first: 0, sets: vec![set(13), set(14)], dynamic: vec![] });
        assert_eq!(s.bindings.len(), 1);
        s.bind(Binding::Offsets { layout: layout(2), first: 0, indices: vec![0], offsets: vec![256] });
        assert_eq!(s.bindings.len(), 1, "descriptor-buffer offsets invalidate the descriptor sets");
        assert!(matches!(s.bindings[0].1, Binding::Offsets { .. }));
        s.bind(Binding::Embedded { layout: layout(2), set: 1 });
        assert_eq!(s.bindings.len(), 2);
        s.bind(Binding::Push { layout: layout(3), set: 2, writes: vec![] });
        assert_eq!(s.bindings.len(), 3, "a push leaves descriptor-buffer offsets alone");
        s.bind(Binding::Offsets { layout: layout(2), first: 0, indices: vec![0], offsets: vec![512] });
        assert_eq!(s.bindings.len(), 3, "offsets for set 0 replace set 0's and leave the push of set 2");
        s.bind(Binding::Sets { layout: layout(1), first: 0, sets: vec![set(15)], dynamic: vec![] });
        assert_eq!(s.bindings.iter().filter(|(_, b)| b.buffer()).count(), 0, "descriptor sets invalidate the offsets");
        assert!(s.bindings.iter().any(|(_, b)| matches!(b, Binding::Push { set: 2, .. })), "and leave the push");
        s.bind(Binding::PushTemplate { template: vk::DescriptorUpdateTemplate::from_raw(4), layout: layout(2), set: 3, data: vec![] });
        s.bind(Binding::Offsets { layout: layout(2), first: 0, indices: vec![0], offsets: vec![0] });
        assert_eq!(s.bindings.iter().filter(|(_, b)| matches!(b, Binding::Push { .. } | Binding::PushTemplate { .. })).count(), 2, "offsets leave both pushes");
        s.bind(Binding::Sets { layout: layout(1), first: 2, sets: vec![set(16)], dynamic: vec![] });
        assert!(!s.bindings.iter().any(|(_, b)| matches!(b, Binding::Push { .. })), "binding set 2 again replaces its push");
    }

    #[test]
    fn only_pushes_reaching_the_compute_stage_are_kept_and_a_covering_push_replaces_the_earlier() {
        let mut s = State::default();
        s.push(layout(1), vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, &[1; 16]);
        assert!(s.pushes.is_empty(), "graphics stages only: the hold never touches them");
        s.push(layout(1), vk::ShaderStageFlags::ALL, 0, &[2; 16]);
        s.push(layout(1), vk::ShaderStageFlags::ALL, 16, &[3; 8]);
        assert_eq!(s.pushes.len(), 2);
        s.push(layout(1), vk::ShaderStageFlags::ALL, 0, &[4; 32]);
        assert_eq!(s.pushes.len(), 1);
        s.push(layout(1), vk::ShaderStageFlags::COMPUTE, 0, &[5; 32]);
        assert_eq!(s.pushes.len(), 2, "a compute-only push does not cover the other stages of an ALL push");
        for k in 0..MAX_PUSHES as u32 {
            s.push(layout(1), vk::ShaderStageFlags::COMPUTE, 64 + 4 * k, &[6; 4]);
        }
        assert!(s.unrestorable().is_some());
    }

    /// Nothing is noted before the device's first hold point, and a buffer begun before it is never taken
    /// for complete; one begun after is.
    #[test]
    fn only_buffers_begun_after_the_first_hold_point_are_complete() {
        let t = Tracker::new(Fns::default());
        let (early, late) = (vk::CommandBuffer::from_raw(1), vk::CommandBuffer::from_raw(2));
        t.clear(early);
        t.note(early, |s| s.bind_compute(Compute::Pipeline(vk::Pipeline::from_raw(5))));
        assert!(lock(&t.states).is_empty(), "nothing noted before the first hold point");
        assert!(!t.activate());
        t.clear(late);
        t.note(late, |s| s.bind_compute(Compute::Pipeline(vk::Pipeline::from_raw(5))));
        assert!(!t.with(early, |s| s.complete()), "began before: its earlier binds were not noted");
        assert!(t.with(late, |s| s.complete() && s.compute.is_some()));
        t.forget(late);
        assert!(!t.with(late, |s| s.complete()), "freed: nothing noted until it begins again");
    }

    #[test]
    fn a_template_spans_its_furthest_element() {
        let e = |ty, offset, count, stride| vk::DescriptorUpdateTemplateEntry { dst_binding: 0, dst_array_element: 0, descriptor_count: count, descriptor_type: ty, offset, stride };
        let b = std::mem::size_of::<vk::DescriptorBufferInfo>();
        assert_eq!(template_bytes(&[e(vk::DescriptorType::STORAGE_BUFFER, 8, 3, 32)]), Some(8 + 2 * 32 + b));
        assert_eq!(template_bytes(&[e(vk::DescriptorType::INLINE_UNIFORM_BLOCK, 100, 12, 0), e(vk::DescriptorType::UNIFORM_TEXEL_BUFFER, 0, 1, 0)]), Some(112));
        assert_eq!(template_bytes(&[e(vk::DescriptorType::MUTABLE_EXT, 0, 1, 0)]), None);
    }
}
