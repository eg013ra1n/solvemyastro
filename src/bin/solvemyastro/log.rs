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

/// The filter of one layer: `RUST_LOG` when set, else `default`.
fn filter(default: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default))
}

/// Install the subscriber. The stderr layer shows `info` (`warn` with
/// `quiet`); the file, when asked for, always gets `info`.
pub fn init(log_file: Option<&Path>, quiet: bool) -> Result<()> {
    let stderr = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_target(false)
        .with_filter(filter(if quiet { "warn" } else { "info" }));
    let file = match log_file {
        Some(path) => {
            let f = File::create(path).with_context(|| format!("create log file {}", path.display()))?;
            Some(fmt::layer().json().with_writer(Mutex::new(f)).with_ansi(false).with_filter(filter("info")))
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
