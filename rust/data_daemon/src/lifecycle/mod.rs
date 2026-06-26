//! Daemon lifecycle: PID file management, daemonization, signal handling, and
//! startup recovery from a previous unclean exit.

pub mod daemonize;
pub mod pidfile;
pub mod recovery;
pub mod shutdown;
