# Contract: MlxBackend

**Entity**: E1 - MlxBackend
**Location**: `src/backends/embedded/mlx.rs`
**Purpose**: MLX GPU-accelerated inference for Apple Silicon using unified memory architecture

## Behavioral Contract

### Must Implement

**Trait**: `InferenceBackend` (internal trait for embedded model variants)

```rust
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn infer(&self, prompt: &str, config: &EmbeddedConfig) -> Result<String, GeneratorError>;
    fn variant(&self) -> ModelVariant;
    async fn load(&mut self) -> Result<(), GeneratorError>;
    async fn unload(&mut self) -> Result<(), GeneratorError>;
}
```

### Contract Requirements

#### CR-MLX-001: Platform Restriction
**MUST** only compile and run on macOS with Apple Silicon (aarch64).

**Test**:
```rust
#[test]
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn test_mlx_backend_unavailable() {
    // This should not compile on non-Apple Silicon platforms
    // Use conditional compilation to prevent instantiation
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn test_mlx_backend_available() {
    let result = MlxBackend::new(test_model_path());
    assert!(result.is_ok(), "MLX must be available on Apple Silicon");
}
```

#### CR-MLX-002: Unified Memory Efficiency
**MUST** use Metal unified memory architecture for zero-copy tensor operations.

**Test**:
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
async fn test_unified_memory_usage() {
    let mlx = MlxBackend::new(test_model_path()).unwrap();

    // Load model
    mlx.load().await.unwrap();

    // Check that model tensors are in unified memory
    let device = mlx.device();
    assert_eq!(device.memory_type(), MemoryType::Unified,
        "Must use unified memory, not separate GPU memory");
}
```

#### CR-MLX-003: Fast Initialization
**MUST** initialize within 100ms (FR-027 startup budget).

**Test**:
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
async fn test_mlx_fast_initialization() {
    let start = Instant::now();

    let mut mlx = MlxBackend::new(test_model_path()).unwrap();
    mlx.load().await.unwrap();

    let load_time = start.elapsed();

    assert!(load_time < Duration::from_millis(100),
        "MLX initialization must complete within 100ms, got {:?}", load_time);
}
```

#### CR-MLX-004: Inference Performance
**MUST** generate commands within 2s total (FR-025).

**Test**:
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
async fn test_mlx_inference_performance() {
    let mut mlx = MlxBackend::new(test_model_path()).unwrap();
    mlx.load().await.unwrap();

    let config = EmbeddedConfig::default();
    let start = Instant::now();

    let output = mlx.infer("Generate bash command to list all files", &config).await.unwrap();

    let inference_time = start.elapsed();

    assert!(inference_time < Duration::from_secs(2),
        "MLX inference must complete within 2s, got {:?}", inference_time);
    assert!(!output.is_empty(), "Must return generated text");
}
```

#### CR-MLX-005: First Token Latency
**MUST** achieve <200ms first token latency for responsive UX.

**Test**:
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
async fn test_mlx_first_token_latency() {
    let mut mlx = MlxBackend::new(test_model_path()).unwrap();
    mlx.load().await.unwrap();

    let config = EmbeddedConfig::default();
    let start = Instant::now();

    // Use streaming to measure first token (if supported)
    let first_token_time = measure_first_token_time(&mlx, "list files", &config).await;

    assert!(first_token_time < Duration::from_millis(200),
        "First token must arrive within 200ms, got {:?}", first_token_time);
}
```

#### CR-MLX-006: Metal Framework Compatibility
**MUST** handle Metal framework errors gracefully.

**Test**:
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
async fn test_mlx_metal_error_handling() {
    let mlx = MlxBackend::new(test_model_path()).unwrap();

    // Simulate Metal framework error (e.g., device loss)
    simulate_metal_error();

    let result = mlx.infer("test", &EmbeddedConfig::default()).await;

    assert!(result.is_err(), "Must handle Metal errors");
    let error = result.unwrap_err();
    assert!(error.to_string().contains("Metal") ||
            error.to_string().contains("GPU"),
        "Error must indicate Metal/GPU issue");
}
```

#### CR-MLX-007: Model Format Support
**MUST** support GGUF quantized models (Q4_K_M minimum).

**Test**:
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn test_mlx_gguf_support() {
    // Test with Q4_K_M quantized model
    let q4_model_path = get_qwen_q4_model_path();
    let result = MlxBackend::new(&q4_model_path);

    assert!(result.is_ok(), "Must support GGUF Q4_K_M quantization");

    // Test with Q8_0 quantized model (higher quality)
    let q8_model_path = get_qwen_q8_model_path();
    let result = MlxBackend::new(&q8_model_path);

    assert!(result.is_ok(), "Must support GGUF Q8_0 quantization");
}
```

#### CR-MLX-008: Concurrent Request Handling
**MUST** safely handle concurrent inference requests (mutex-protected model).

**Test**:
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
async fn test_mlx_concurrent_requests() {
    let mlx = Arc::new(MlxBackend::new(test_model_path()).unwrap());
    let config = EmbeddedConfig::default();

    // Spawn multiple concurrent requests
    let mut handles = vec![];
    for i in 0..5 {
        let mlx_clone = Arc::clone(&mlx);
        let config_clone = config.clone();

        handles.push(tokio::spawn(async move {
            mlx_clone.infer(&format!("request {}", i), &config_clone).await
        }));
    }

    // All requests should complete successfully
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Concurrent requests must all succeed");
    }
}
```

#### CR-MLX-009: Resource Cleanup
**MUST** release GPU resources on unload.

**Test**:
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
async fn test_mlx_resource_cleanup() {
    let mut mlx = MlxBackend::new(test_model_path()).unwrap();

    // Load model to GPU
    mlx.load().await.unwrap();

    // Check GPU memory usage
    let gpu_mem_before = get_metal_memory_usage();

    // Unload
    mlx.unload().await.unwrap();

    // Wait for cleanup
    tokio::time::sleep(Duration::from_millis(100)).await;

    // GPU memory should be released
    let gpu_mem_after = get_metal_memory_usage();
    assert!(gpu_mem_after < gpu_mem_before,
        "GPU memory must be released after unload");
}
```

#### CR-MLX-010: Temperature Control
**MUST** respect temperature setting for sampling control.

**Test**:
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
async fn test_mlx_temperature_control() {
    let mut mlx = MlxBackend::new(test_model_path()).unwrap();
    mlx.load().await.unwrap();

    // Test with low temperature (deterministic)
    let config_low = EmbeddedConfig {
        temperature: 0.1,
        ..Default::default()
    };

    let output1 = mlx.infer("list files", &config_low).await.unwrap();
    let output2 = mlx.infer("list files", &config_low).await.unwrap();

    // Low temperature should produce similar outputs
    assert_eq!(output1, output2,
        "Low temperature should be deterministic");

    // Test with high temperature (creative)
    let config_high = EmbeddedConfig {
        temperature: 1.5,
        ..Default::default()
    };

    let output3 = mlx.infer("list files", &config_high).await.unwrap();
    let output4 = mlx.infer("list files", &config_high).await.unwrap();

    // High temperature may produce different outputs
    // (Not guaranteed, but statistically likely)
}
```

## Integration Points

### With EmbeddedModelBackend
- MlxBackend instantiated when `ModelVariant::MLX` selected
- Wrapped in `Box<dyn InferenceBackend>` for polymorphism
- Called via `InferenceBackend::infer()` interface

### With mlx-rs Crate
- Use `mlx-rs` 0.11+ for native Rust bindings
- Conditional compilation: `#[cfg(all(target_os = "macos", target_arch = "aarch64"))]`
- Access unified memory via `mlx::Device::metal()`

### With Tokenizer
- Use `tokenizers` crate for Qwen tokenizer
- Load from `tokenizer.json` in model directory
- Encode prompt → tokens → MLX forward pass → tokens → decode output

## Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| Initialization | <100ms | Model load to GPU (FR-027) |
| First token | <200ms | Prompt processing + first generation |
| Throughput | ~8 tokens/sec | M1 Mac baseline |
| Total inference | <2s | 20-token command generation (FR-025) |
| Memory usage | ~1.2GB | Model weights + runtime |
| GPU utilization | >80% | Efficient Metal usage |

## Error Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Non-Apple Silicon | Compile-time error (cfg gate) |
| Metal unavailable | `GeneratorError::PlatformUnsupported` |
| Model load failure | `GeneratorError::ModelLoadFailed` with Metal error details |
| OOM (model too large) | `GeneratorError::ResourceExhausted` with memory info |
| Inference timeout | `GeneratorError::Timeout` after 30s |
| Concurrent access | Mutex ensures serial access, no data race |

---

**Test Coverage Target**: 95% for MLX-specific logic

**Platform Testing**: Requires physical M1/M2/M3/M4 Mac for integration tests

**Conditional Compilation**: All MLX code must be behind `#[cfg(all(target_os = "macos", target_arch = "aarch64"))]`
