//! Double-fork + `setsid` detachment for `launch --background`.
//!
//! The standard Unix recipe:
//!
//! 1. Parent forks. Parent blocks on a readiness pipe waiting for the
//!    grandchild's startup status (see below); the intermediate child
//!    continues.
//! 2. Intermediate child calls `setsid` so it becomes the session leader of a
//!    new session with no controlling terminal.
//! 3. Intermediate child forks again, then exits — orphaning the grandchild
//!    to init so it can never reacquire a controlling terminal even on
//!    accidental `open(...)` of a TTY.
//! 4. Grandchild closes stdin/stdout/stderr (redirecting to `/dev/null`) and
//!    returns to the caller as the long-lived daemon process.
//!
//! ### Startup readiness pipe
//!
//! A pipe is wired between the original caller and the grandchild so that
//! early-startup failures (PID file already held, IO error binding the lock,
//! tracing init failure) reach the user's terminal instead of being lost to
//! the `/dev/null`-redirected stderr. The grandchild must explicitly call
//! [`ReadinessReporter::ready`] or [`ReadinessReporter::fail`]; otherwise the
//! reporter's `Drop` reports an "exited before reporting readiness" message
//! so the launcher always observes some terminal status rather than blocking
//! forever.
//!
//! Foreground mode (no `--background`) is a no-op: the caller stays in the
//! current process group and keeps its terminal.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::io::{AsRawFd, OwnedFd};

use nix::sys::stat::{umask, Mode};
use nix::sys::wait::waitpid;
use nix::unistd::{chdir, dup2, fork, pipe, setsid, ForkResult};

/// Outcome of [`daemonize`].
pub enum DaemonizeOutcome {
    /// We are the original caller process. Call [`ReadinessReader::read`] to
    /// block until the grandchild reports its startup status, then propagate
    /// the result to the user's shell.
    Parent(ReadinessReader),
    /// We are the long-lived daemon process. Continue with daemon startup
    /// and hand off to the reporter once the PID file is acquired (or any
    /// earlier step fails) so the launcher unblocks.
    Child(ReadinessReporter),
}

/// Startup status the grandchild reports back to the original caller.
#[derive(Debug)]
pub enum Readiness {
    /// Grandchild started successfully; payload is the daemon's PID as a
    /// string so the launcher can echo it on stdout for the user.
    Ready(String),
    /// Grandchild reported a startup failure; payload is the human-readable
    /// reason to surface on stderr.
    Failed(String),
    /// The pipe closed without any message — typically the grandchild was
    /// signalled or aborted before it could report. Treated as failure.
    Disconnected,
}

/// Read end of the readiness pipe, held by the original caller.
pub struct ReadinessReader {
    pipe: OwnedFd,
}

impl ReadinessReader {
    /// Block until the grandchild closes its end of the pipe, then classify
    /// the message it sent.
    pub fn read(self) -> io::Result<Readiness> {
        let mut file = File::from(self.pipe);
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        let trimmed = buffer.trim_end_matches('\n');
        if trimmed.is_empty() {
            return Ok(Readiness::Disconnected);
        }
        if let Some(pid) = trimmed.strip_prefix("OK ") {
            return Ok(Readiness::Ready(pid.to_string()));
        }
        if let Some(message) = trimmed.strip_prefix("ERR ") {
            return Ok(Readiness::Failed(message.to_string()));
        }
        // Unrecognised payload — surface the raw text so the user has
        // something to act on.
        Ok(Readiness::Failed(trimmed.to_string()))
    }
}

/// Write end of the readiness pipe, held by the grandchild.
///
/// `ready` and `fail` are consuming methods: sending a status closes the
/// underlying FD, which is what unblocks the original caller's
/// [`ReadinessReader::read`].
pub struct ReadinessReporter {
    pipe: Option<OwnedFd>,
}

impl ReadinessReporter {
    /// Report a successful startup, passing the daemon's PID. Tolerates a
    /// closed pipe (caller may have died) so the daemon does not crash on
    /// `EPIPE` after the launcher has been killed.
    pub fn ready(mut self, pid: u32) -> io::Result<()> {
        write_status(self.pipe.take(), &format!("OK {pid}"))
    }

    /// Report a startup failure. Same broken-pipe tolerance as
    /// [`ready`](Self::ready).
    pub fn fail(mut self, message: impl AsRef<str>) -> io::Result<()> {
        write_status(self.pipe.take(), &format!("ERR {}", message.as_ref()))
    }
}

impl Drop for ReadinessReporter {
    fn drop(&mut self) {
        // Best-effort fallback so the launcher gets a deterministic failure
        // message instead of a blank `Disconnected` when the grandchild
        // aborts between fork and the explicit ready/fail call.
        let _ = write_status(
            self.pipe.take(),
            "ERR daemon exited before reporting readiness",
        );
    }
}

fn write_status(pipe: Option<OwnedFd>, line: &str) -> io::Result<()> {
    let Some(pipe) = pipe else { return Ok(()) };
    let mut file = File::from(pipe);
    match writeln!(file, "{line}") {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(error) => Err(error),
    }
}

/// Detach the current process into a background daemon using the standard
/// double-fork + `setsid` recipe.
///
/// # Safety
///
/// `fork` after threads have been spawned is undefined behaviour. The caller
/// must invoke this *before* creating the Tokio runtime or spawning any
/// threads — `cli::launch::run` enforces that by daemonising at the top of the
/// command handler.
pub fn daemonize() -> io::Result<DaemonizeOutcome> {
    let (read_fd, write_fd) = pipe().map_err(io::Error::from)?;

    // First fork: parent waits on the pipe for the grandchild's status.
    // SAFETY: caller must not have spawned any threads yet; see fn docs.
    match unsafe { fork() }.map_err(io::Error::from)? {
        ForkResult::Parent { child } => {
            // The parent never writes to the pipe; drop its write end so the
            // only remaining writers are the intermediate child and the
            // grandchild. Once both close, `read_to_string` returns.
            drop(write_fd);
            // Reap the intermediate child so it does not become a zombie. It
            // exits immediately after the second fork below.
            let _ = waitpid(child, None);
            return Ok(DaemonizeOutcome::Parent(ReadinessReader { pipe: read_fd }));
        }
        ForkResult::Child => {
            // Intermediate child does not need to read from the pipe.
            drop(read_fd);
        }
    }

    // From here we are inside the intermediate child. Its stderr is still
    // attached to the user's TTY, so we must not return Err — anyhow would
    // print a stray error in addition to whatever the original launcher
    // prints. Instead, push any failure down the readiness pipe (which the
    // launcher already reads) and `exit(1)`.
    match run_intermediate_child(write_fd) {
        Ok(outcome) => Ok(outcome),
        Err(IntermediateExit) => std::process::exit(1),
    }
}

/// Marker returned when the intermediate child failed and has already
/// reported the failure via the readiness pipe.
struct IntermediateExit;

/// Runs the post-first-fork steps in the intermediate child. On any error,
/// writes an `ERR ...` line down the readiness pipe and returns
/// [`IntermediateExit`] so the caller can `process::exit(1)` — keeping all
/// failure output going through the launcher's already-attached pipe rather
/// than the intermediate child's still-live stderr.
fn run_intermediate_child(write_fd: OwnedFd) -> Result<DaemonizeOutcome, IntermediateExit> {
    // `write_status` consumes the OwnedFd, so keep the original around for
    // the success path and only build a fresh File handle on error via
    // `try_clone`.
    let report_failure = |error: &dyn std::fmt::Display, stage: &str| -> IntermediateExit {
        // Best-effort: if `try_clone` fails we still exit — Drop on
        // `write_fd` will close the pipe and the launcher will surface a
        // `Disconnected` status.
        if let Ok(clone) = write_fd.try_clone() {
            let _ = write_status(Some(clone), &format!("ERR {stage}: {error}"));
        }
        IntermediateExit
    };

    // Become a session leader so we have no controlling terminal.
    if let Err(error) = setsid() {
        return Err(report_failure(&error, "setsid failed"));
    }

    // Second fork: ensures the daemon is not a session leader, so it can never
    // reacquire a controlling terminal.
    // SAFETY: only this child is running at this point — still single-threaded.
    let fork_result = match unsafe { fork() } {
        Ok(result) => result,
        Err(error) => return Err(report_failure(&error, "second fork failed")),
    };
    match fork_result {
        ForkResult::Parent { .. } => {
            // The grandchild owns its own inherited copy of `write_fd`. We
            // drop ours here so the original parent's `read_to_string` only
            // waits for the grandchild's eventual write — otherwise our
            // copy would keep the pipe open until this scope's `exit`.
            drop(write_fd);
            // Intermediate child exits immediately so the grandchild is
            // re-parented to PID 1 (init).
            std::process::exit(0);
        }
        ForkResult::Child => {}
    }

    // Standard double-fork hygiene: detach from the launcher's CWD (so it
    // can be unmounted) and reset umask to a predictable value so files we
    // create later have stable permissions regardless of the launching
    // shell's environment.
    if let Err(error) = chdir("/") {
        return Err(report_failure(&error, "chdir(\"/\") failed"));
    }
    umask(Mode::empty());

    if let Err(error) = redirect_standard_streams_to_devnull() {
        return Err(report_failure(&error, "redirect std streams failed"));
    }

    Ok(DaemonizeOutcome::Child(ReadinessReporter {
        pipe: Some(write_fd),
    }))
}

fn redirect_standard_streams_to_devnull() -> io::Result<()> {
    let devnull = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/null")?;
    let fd = devnull.as_raw_fd();
    // stdin
    dup2(fd, 0).map_err(io::Error::from)?;
    // stdout
    dup2(fd, 1).map_err(io::Error::from)?;
    // stderr — direct stderr writes (panics, libc messages) are lost after
    // this point. `cli::launch::run_daemon` initialises tracing against a
    // log file *after* this redirect for background mode, so structured
    // logs survive.
    dup2(fd, 2).map_err(io::Error::from)?;
    Ok(())
}
