//! Logging: a stderr layer for the person at the terminal, an optional
//! JSON-lines file for the record.

use std::fs::File;
use std::io::IsTerminal;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

/// The stderr layer's filter: `RUST_LOG` when set, else `default`.
fn stderr_filter(default: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default))
}

/// Install the subscriber. The stderr layer shows `info` (`warn` with
/// `quiet`) unless `RUST_LOG` says otherwise; the file, when asked for, always
/// gets `info` and ignores `RUST_LOG`, so a record is the same on every run.
pub fn init(log_file: Option<&Path>, quiet: bool) -> Result<()> {
    let stderr = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_target(false)
        .with_filter(stderr_filter(if quiet { "warn" } else { "info" }));
    let file = match log_file {
        Some(path) => {
            let f = File::create(path).with_context(|| format!("create log file {}", path.display()))?;
            Some(fmt::layer().json().with_writer(Mutex::new(f)).with_ansi(false).with_filter(EnvFilter::new("info")))
        }
        None => None,
    };
    tracing_subscriber::registry()
        .with(stderr)
        .with(file)
        .try_init()
        .context("install the log subscriber")?;
    Ok(())
}
