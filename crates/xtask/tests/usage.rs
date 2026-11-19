//! The dispatcher's own exit code: an unknown or missing task is EX_USAGE (64), which no task uses for its own outcome
//! (they use 0, 1, 2, 3 and 10).

use std::process::Command;

#[test]
fn xtask_unknown_task_and_no_task_exit_ex_usage() {
    for args in [vec!["no-such-task"], vec![]] {
        let out = Command::new(env!("CARGO_BIN_EXE_xtask")).args(&args).output().unwrap();
        assert_eq!(out.status.code(), Some(64), "{args:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("usage: cargo xtask"), "{args:?}");
    }
}
