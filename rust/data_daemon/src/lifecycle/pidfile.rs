//! Single-instance enforcement via an `flock`'d PID file.
//!
//! The launching process opens the PID file with `O_CREAT|O_RDWR`, takes a
//! non-blocking exclusive `flock`, writes its own PID, and keeps the file
//! descriptor open for the rest of the daemon's life. When the [`PidFile`]
//! value is dropped — either explicitly on graceful shutdown or implicitly on
//! process exit — the kernel releases the lock. The file itself is left in
//! place (unlinking an `flock`'d file races with a concurrent launcher); the
//! next launcher reuses it under the lock.
//!
//! The `flock` gives atomic single-instance semantics across crash, SIGKILL,
//! and parallel launches: a stale PID file from a SIGKILL'd daemon has no
//! active flock holder, so the next launcher's `flock` immediately succeeds
//! and the launcher overwrites the contents with its own PID.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use nix::fcntl::{Flock, FlockArg};
use nix::unistd::Pid;
use thiserror::Error;

/// Errors raised while acquiring or reading a PID file.
#[derive(Debug, Error)]
pub enum PidFileError {
    /// Another daemon already holds the PID file's flock. Carries the PID we
    /// found on disk (when readable) for the user-facing error message.
    #[error("Daemon already running (pid={0})")]
    AlreadyRunning(i32),
    /// An I/O or `flock` failure prevented us from acquiring the file.
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// An exclusively `flock`'d PID file that releases the lock on drop. The file
/// is left in place (see [`PidFile::release`]); the next launcher reuses it.
pub struct PidFile {
    lock: Option<Flock<File>>,
}

impl PidFile {
    /// Acquire the PID file at `path`, writing the current process's PID into
    /// it. Returns an [`PidFileError::AlreadyRunning`] if another daemon
    /// already holds the lock.
    ///
    /// Parent directories are created if missing.
    pub fn acquire(path: impl Into<PathBuf>) -> Result<Self, PidFileError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let mut lock = match Flock::lock(file, FlockArg::LockExclusiveNonblock) {
            Ok(lock) => lock,
            Err((file, nix::errno::Errno::EWOULDBLOCK)) => {
                let pid = read_pid_from_open_file(&file).unwrap_or(-1);
                return Err(PidFileError::AlreadyRunning(pid));
            }
            Err((_, err)) => return Err(io::Error::from(err).into()),
        };

        // The previous holder may have left a PID written from before its
        // exit; truncate before writing ours so a partial read sees a clean
        // value. `set_len(0)` does not move the file offset, so the explicit
        // seek to byte 0 is what guarantees we overwrite from the start.
        lock.set_len(0)?;
        lock.seek(SeekFrom::Start(0))?;
        writeln!(lock, "{}", Pid::this().as_raw())?;
        lock.flush()?;

        Ok(PidFile { lock: Some(lock) })
    }

    /// Release the flock, relinquishing single-instance ownership. Idempotent.
    ///
    /// Deliberately does **not** unlink the file. Unlinking an `flock`'d PID
    /// file races in *both* orderings: unlink-then-drop lets the exiting process
    /// briefly co-hold with a starting one, and drop-then-unlink is worse — a
    /// new launcher can grab the lock on the existing inode in the gap, after
    /// which our `remove_file` deletes *its* live PID file, letting a third
    /// launcher create a fresh inode and run a second daemon. The flock alone is
    /// the single-instance authority, so we just release it and leave the file
    /// for the next launcher to reuse in place — exactly the path `acquire`
    /// already takes after a SIGKILL (lock, truncate, rewrite).
    pub fn release(&mut self) {
        if let Some(lock) = self.lock.take() {
            drop(lock);
        }
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        self.release();
    }
}

/// Read an integer PID from a file at `path`, returning `None` when the file
/// is missing, empty, or contains a non-integer or non-positive value.
///
/// Mirrors `daemon_os_control.read_pid_from_file`.
pub fn read_pid_from_file(path: &Path) -> Option<i32> {
    let text = std::fs::read_to_string(path).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let pid = trimmed.parse::<i32>().ok()?;
    if pid > 0 {
        Some(pid)
    } else {
        None
    }
}

fn read_pid_from_open_file(file: &File) -> Option<i32> {
    let mut clone = file.try_clone().ok()?;
    clone.seek(SeekFrom::Start(0)).ok()?;
    let mut buffer = String::new();
    clone.read_to_string(&mut buffer).ok()?;
    let trimmed = buffer.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<i32>().ok().filter(|pid| *pid > 0)
}

/// Return `true` when `pid` is a live, non-zombie process the current user can
/// signal.
///
/// The zombie probe is per-platform because POSIX exposes none: Linux reads
/// `/proc/<pid>/stat`, macOS the Darwin-only `libproc` call, and any other
/// target skips the filter.
pub fn pid_is_running(pid: i32) -> bool {
    match nix::sys::signal::kill(Pid::from_raw(pid), None) {
        Ok(()) => !is_zombie(pid),
        Err(nix::errno::Errno::EPERM) => true,
        Err(_) => false,
    }
}

#[cfg(target_os = "linux")]
fn is_zombie(pid: i32) -> bool {
    let stat_path = PathBuf::from(format!("/proc/{pid}/stat"));
    let Ok(contents) = std::fs::read_to_string(stat_path) else {
        return false;
    };
    // Field 3 of /proc/<pid>/stat is the process state, immediately after the
    // closing paren of the comm field.
    let Some(after_comm) = contents.rsplit(')').next() else {
        return false;
    };
    after_comm.split_whitespace().next() == Some("Z")
}

#[cfg(target_os = "macos")]
fn is_zombie(pid: i32) -> bool {
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    // SAFETY: `proc_pidinfo` writes at most `size` bytes into `info`, which is
    // a live, correctly sized `proc_bsdinfo`.
    let written = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTBSDINFO,
            0,
            std::ptr::addr_of_mut!(info).cast(),
            size,
        )
    };
    // A short or failed read leaves the status unknown; treat it as not a
    // zombie so the caller falls back to the `kill` probe's answer.
    written == size && info.pbi_status == libc::SZOMB
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn is_zombie(_pid: i32) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use std::process::Command;
    use std::time::{Duration, Instant};

    use super::*;
    use tempfile::tempdir;

    #[test]
    fn acquire_writes_current_pid_and_release_frees_the_lock() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("daemon.pid");

        let mut pid_file = PidFile::acquire(&path).expect("acquire");
        assert!(path.exists());
        assert_eq!(read_pid_from_file(&path), Some(std::process::id() as i32));

        pid_file.release();
        // The file is intentionally left in place (unlinking an flock'd file
        // races a concurrent launcher); releasing only frees the lock, so a
        // fresh acquire on the same path must succeed.
        assert!(
            path.exists(),
            "release leaves the pid file for in-place reuse"
        );
        let reacquired = PidFile::acquire(&path).expect("re-acquire after release");
        drop(reacquired);
    }

    #[test]
    fn second_acquire_in_same_process_fails_with_already_running() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("daemon.pid");

        let _guard = PidFile::acquire(&path).expect("first acquire");
        match PidFile::acquire(&path) {
            Err(PidFileError::AlreadyRunning(pid)) => {
                assert_eq!(pid, std::process::id() as i32);
            }
            Err(other) => panic!("expected AlreadyRunning, got error: {other}"),
            Ok(_) => panic!("expected AlreadyRunning, got Ok"),
        }
    }

    #[test]
    fn acquire_after_release_succeeds_and_overwrites_pid() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("daemon.pid");

        {
            let _guard = PidFile::acquire(&path).expect("first acquire");
        }
        // release() leaves the file in place (only the flock is dropped). Seed a
        // stale PID to prove acquire re-locks and overwrites an existing file.
        std::fs::write(&path, "99999\n").expect("seed stale pid");
        let pid_file = PidFile::acquire(&path).expect("acquire after stale");
        assert_eq!(read_pid_from_file(&path), Some(std::process::id() as i32));
        drop(pid_file);
    }

    #[test]
    fn read_pid_from_file_handles_missing_and_garbage() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("daemon.pid");

        assert_eq!(read_pid_from_file(&path), None);
        std::fs::write(&path, "").unwrap();
        assert_eq!(read_pid_from_file(&path), None);
        std::fs::write(&path, "abc\n").unwrap();
        assert_eq!(read_pid_from_file(&path), None);
        std::fs::write(&path, "0\n").unwrap();
        assert_eq!(read_pid_from_file(&path), None);
        std::fs::write(&path, "  4321  ").unwrap();
        assert_eq!(read_pid_from_file(&path), Some(4321));
    }

    #[test]
    fn pid_is_running_true_for_self_and_false_for_unused_pid() {
        let our_pid = std::process::id() as i32;
        assert!(pid_is_running(our_pid));
        // Use a PID guaranteed to be above the kernel's `pid_max` (Linux's
        // hard cap is 2^22 ≈ 4M, well below `i32::MAX` ≈ 2.1B) so the
        // probe is guaranteed to refer to no process. `read_pid_from_file`
        // already rejects 0/negative values, so we do not need to guard
        // against `kill(0, 0)`'s broadcast-to-process-group semantics here.
        assert!(!pid_is_running(i32::MAX));
    }

    #[test]
    fn pid_is_running_false_for_an_unreaped_zombie_child() {
        // `stop` signals a daemon it did not spawn, so it can never reap the
        // corpse; the real parent may stay blocked for seconds. A zombie has to
        // read as not-running or `stop` waits out its graceful and SIGKILL
        // timeouts on a process that already exited.
        let mut child = Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .spawn()
            .expect("spawn");
        let pid = child.id() as i32;

        let deadline = Instant::now() + Duration::from_secs(5);
        while pid_is_running(pid) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        let still_running = pid_is_running(pid);

        child.wait().expect("reap");
        assert!(
            !still_running,
            "exited-but-unreaped child reported as running"
        );
    }
}
