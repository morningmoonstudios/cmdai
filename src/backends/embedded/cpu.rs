// Candle CPU inference for cross-platform support

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;

use crate::backends::embedded::common::{EmbeddedConfig, InferenceBackend, ModelVariant};
use crate::backends::GeneratorError;

/// CPU backend using Candle for cross-platform inference
pub struct CpuBackend {
    model_path: PathBuf,
    // Model will be loaded lazily
    // model: Option<Arc<CandleModel>>,
    // tokenizer: Option<Arc<Tokenizer>>,
}

impl CpuBackend {
    /// Create a new CPU backend with the given model path
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
impl InferenceBackend for CpuBackend {
    async fn infer(&self, _prompt: &str, _config: &EmbeddedConfig) -> Result<String, GeneratorError> {
        // Placeholder - will be implemented in Phase 4.3
        Err(GeneratorError::GenerationFailed {
            details: "CPU backend not yet implemented".to_string(),
        })
    }

    fn variant(&self) -> ModelVariant {
        ModelVariant::CPU
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
    fn test_cpu_backend_new() {
        let backend = CpuBackend::new(PathBuf::from("/tmp/model.gguf"));
        assert!(backend.is_ok());
    }

    #[test]
    fn test_cpu_backend_empty_path() {
        let backend = CpuBackend::new(PathBuf::from(""));
        assert!(backend.is_err());
    }

    #[test]
    fn test_cpu_variant() {
        let backend = CpuBackend::new(PathBuf::from("/tmp/model.gguf")).unwrap();
        assert_eq!(backend.variant(), ModelVariant::CPU);
    }
}
