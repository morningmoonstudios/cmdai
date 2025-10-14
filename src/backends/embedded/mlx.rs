// MLX GPU-accelerated inference for Apple Silicon
// Only compiles on macOS aarch64

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;

use crate::backends::embedded::common::{EmbeddedConfig, InferenceBackend, ModelVariant};
use crate::backends::GeneratorError;

/// MLX backend for Apple Silicon GPU acceleration
pub struct MlxBackend {
    model_path: PathBuf,
    // Model will be loaded lazily
    // model: Option<Arc<MlxModel>>,
    // tokenizer: Option<Arc<Tokenizer>>,
}

impl MlxBackend {
    /// Create a new MLX backend with the given model path
    pub fn new(model_path: PathBuf) -> Result<Self, GeneratorError> {
        if model_path.to_str().unwrap_or("").is_empty() {
            return Err(GeneratorError::ConfigError {
                message: "Model path cannot be empty".to_string(),
            });
        }

        Ok(Self { model_path })
    }
}

#[async_trait]
impl InferenceBackend for MlxBackend {
    async fn infer(&self, _prompt: &str, _config: &EmbeddedConfig) -> Result<String, GeneratorError> {
        // Placeholder - will be implemented in Phase 4.3
        Err(GeneratorError::GenerationFailed {
            details: "MLX backend not yet implemented".to_string(),
        })
    }

    fn variant(&self) -> ModelVariant {
        ModelVariant::MLX
    }

    async fn load(&mut self) -> Result<(), GeneratorError> {
        // Placeholder - will be implemented in Phase 4.3
        Ok(())
    }

    async fn unload(&mut self) -> Result<(), GeneratorError> {
        // Placeholder - will be implemented in Phase 4.3
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlx_backend_new() {
        let backend = MlxBackend::new(PathBuf::from("/tmp/model.gguf"));
        assert!(backend.is_ok());
    }

    #[test]
    fn test_mlx_backend_empty_path() {
        let backend = MlxBackend::new(PathBuf::from(""));
        assert!(backend.is_err());
    }

    #[test]
    fn test_mlx_variant() {
        let backend = MlxBackend::new(PathBuf::from("/tmp/model.gguf")).unwrap();
        assert_eq!(backend.variant(), ModelVariant::MLX);
    }
}
