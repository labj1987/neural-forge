//! This process's writer thread for the channel: the writes the game's threads could not make at once
//! because another writer held a sequence lock's write section (`ShmHeader`'s `*_without_waiting`
//! writes). A game thread never waits for another writer: not for a live one, and not the
//! 50 ms (`STUCK_FOR`) a writer that died inside its section is given before it is taken over. This
//! thread may wait, and makes the writes in the order they were handed over; while any is waiting
//! here, later ones from the game's threads queue behind it, so an older write never lands after a
//! newer one.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Mutex, OnceLock};

/// A write handed over.
pub(crate) type Job = Box<dyn FnOnce() + Send>;

/// Writes handed over and not finished yet.
static PENDING: AtomicUsize = AtomicUsize::new(0);

/// The writer thread's queue, started with the first write handed over; `None` when it could not be
/// started.
static QUEUE: OnceLock<Option<Mutex<Sender<Job>>>> = OnceLock::new();

/// Whether a write handed over is not finished yet: a later one goes behind it.
pub(crate) fn pending() -> bool {
    PENDING.load(Ordering::Acquire) > 0
}

/// Makes `job` on the writer thread, after the ones handed over before. Without a thread (it could not be
/// started) it is made here, the one case a game thread may wait for another writer.
pub(crate) fn hand_over(job: Job) {
    let queue = QUEUE.get_or_init(|| {
        let (send, receive) = channel::<Job>();
        std::thread::Builder::new()
            .name("nf-shm-writer".into())
            .spawn(move || {
                for job in receive {
                    job();
                    PENDING.fetch_sub(1, Ordering::Release);
                }
            })
            .ok()
            .map(|_| Mutex::new(send))
    });
    PENDING.fetch_add(1, Ordering::AcqRel);
    let job = match queue {
        Some(send) => match send.lock().unwrap_or_else(std::sync::PoisonError::into_inner).send(job) {
            Ok(()) => return,
            Err(failed) => failed.0,
        },
        None => job,
    };
    job();
    PENDING.fetch_sub(1, Ordering::Release);
}

/// Tests: waits up to `limit` for every write handed over to be made. Whether they were.
#[cfg(test)]
pub(crate) fn settle(limit: std::time::Duration) -> bool {
    let start = std::time::Instant::now();
    while pending() {
        if start.elapsed() >= limit {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    true
}
