# CPU-only mode configuration for testnet environment
export CUDA_VISIBLE_DEVICES=""                    # Disable GPU access
export TS_METRICS_COLLECTOR_DISABLE=true          # Disable GPU metrics collection
export TORCH_DEVICE=cpu                           # Force CPU device

torchserve --start \
    --ncs \
    --model-store model_store \
    --models ple.mar \
    --ts-config config.properties
