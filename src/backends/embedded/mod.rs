// Embedded model backends for offline command generation

// Platform-specific MLX backend (Apple Silicon only)
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod mlx;

// Cross-platform CPU backend
pub mod cpu;

// Common types and traits
mod common;

// Re-export common types
pub use common::{EmbeddedConfig, InferenceBackend, ModelVariant};

// Re-export platform-specific backend
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use mlx::MlxBackend;

pub use cpu::CpuBackend;
