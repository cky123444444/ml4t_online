# Copilot Instructions for ML Server Demo

## Architecture Overview

This is a **financial ML trading system** with 3 main services:
1. **Main Server** (Flask on port 8000) - Feature generation, order management, and API endpoints
2. **TorchServe** (ports 8080/8081/8082) - ML model inference using the `dragonnet` model
3. **Background Aggregator** - Hourly OHLCV data aggregation with 60-minute sliding windows

### Pipeline Architecture

All components inherit from `BaseOp` in [src/ops/base_op.py](src/ops/base_op.py) with a consistent `execute()` interface:

```
Retriever → Calculator → Adaptor → Strategy → Executors
```

**Key flow**: 
- Retriever fetches market data (live API or cached HDF)
- Calculator computes Alpha102 features + rolling z-scores
- Adaptor transforms to Binance-compatible format
- Strategy determines trading decisions
- Executors handle order routing/placement/closure/finalization

## Critical Conventions

### Dual-Mode Operation

The system operates in two modes via `OFFLINE_DEBUG_MODE`:
- **Offline Mode** (`OFFLINE_DEBUG_MODE=1`): Uses cached HDF data, saves debug files to `debug_output/`
- **Online Mode** (`OFFLINE_DEBUG_MODE=0`): Live Binance API integration

When coding new features, always respect this mode flag. Check [src/serving/server.py](src/serving/server.py#L56) for usage pattern.

### Logging Convention

All modules use the centralized logger from `src.utils.logger.setup_logger()`:
```python
from src.utils.logger import setup_logger
logger = setup_logger('module_name')  # Use descriptive module name
```

Logs go to `/app/logs/server/server.log` (rotated at 10MB) with format:
```
%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s
```

Control verbosity via `LOG_LEVEL` env var (defaults to INFO).

### Testing Patterns

Tests live in `src/test/*_test.py` and use **unittest** framework:

```bash
# Unit tests only (no API calls, mocked dependencies)
./run_unittest.sh

# Integration tests (requires BINANCE_API_KEY/SECRET)
./run_unittest.sh --integration
```

**Integration tests** are gated by `RUN_INTEGRATION_TESTS=true` env var and only run:
- On main branch pushes
- Daily at 2 AM UTC (via GitHub Actions)
- Manual workflow dispatch

Test outputs go to `unittest_output/`. See [run_unittest.sh](run_unittest.sh) for test configuration.

### Data Storage

- **HDF5 Files**: Market data in `data/hdf_dumper/*.h5` (format: `features_YYYYMMDD_HHMMSS_mmm_XXXX.h5`)
- **SQLite**: Order tracking (`order_repo.py`) and score tracking (`score_repo.py`)
- **Debug Output**: Intermediate DataFrames/JSON saved to `debug_output/` when `OFFLINE_DEBUG_MODE=1`

## Common Workflows

### Local Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Start TorchServe first
bash torchserve/start_torchserve.sh

# Start main server
python -m src.serving.server

# Run specific test
python -m pytest src/test/calculator_test.py::TestSampleCalculator::test_sample_calculation -v
```

### Docker Deployment

```bash
# Production: both services
./deploy.sh

# With debug mode and file watching
./deploy.sh --offline-debug-mode --watch

# Test deployment health
./test_deployment.sh

# Stop all services
./stop.sh
```

Docker networking: Main server connects to TorchServe via `TS_BASE_URL=http://torchserve:8080` (docker network name).

### Adding New Operations

When creating a new Retriever/Calculator/Strategy:

1. Inherit from the base class in `src/ops/base_{type}.py`
2. Implement the `execute()` method
3. Use consistent logger naming: `setup_logger('your_module_name')`
4. Add unit tests in `src/test/your_module_test.py`
5. For API-dependent features, add integration tests gated by `RUN_INTEGRATION_TESTS`

Example skeleton:
```python
from src.ops.calculator.base_calculator import BaseCalculator
from src.utils.logger import setup_logger

logger = setup_logger('my_calculator')

class MyCalculator(BaseCalculator):
    def __init__(self, df, **kwargs):
        super().__init__(df)
        self.df = df
        
    def execute(self):
        # Your logic here
        logger.info("Processing data...")
        return result
```

## Key Configuration

Environment variables (see [docker-compose.yml](docker-compose.yml)):
- `TS_BASE_URL`: TorchServe endpoint (default: `http://127.0.0.1:8080`)
- `OFFLINE_DEBUG_MODE`: Enable debug output and offline mode (`0` or `1`)
- `NUM_FEATURES`: Feature count to keep (default: 120)
- `HISTORY_WINDOW`: Rolling window size (default: 240)
- `BINANCE_API_KEY` / `BINANCE_API_SECRET`: For live trading (integration tests only)

Binance client config in [src/ops/clients/binance_futures_client.py](src/ops/clients/binance_futures_client.py):
- Defaults to **testnet mode** (`testnet=True`)
- Hedge mode enabled automatically
- Default taker fee: 0.05%

## TorchServe Model

**Model**: `dragonnet`
**Input shape**: `[batch_size, 120 features, 240 timesteps]`
**Outputs**: 5 predictions (y0_trmt, y0_ctrl, y1_trmt, y1_ctrl, t)

Deploy/redeploy model:
```bash
cd torchserve
./init_env.sh          # First time setup
./re_deploy_model.sh   # Update model
./do_inference.sh      # Test inference
```

## Aggregator Service

Run hourly aggregation job:
```bash
# One-off for yesterday
python run_aggregator.py --once

# Specific date
python run_aggregator.py --once --date 20250115

# Scheduler (runs at xx:05 every hour)
python run_aggregator.py
```

**Output**: Symbol-specific aggregated HDF files with 10 new features (open/high/low/close/volume mean/std).

**Critical Implementation Details**:
- **Window Semantics**: `[T-60, T)` using `shift(1)` to prevent data leakage
- **Filename Parsing**: Regex-based to support both `features_YYYYMMDD.h5` and `features_YYYYMMDD_HHMMSS_mmm_XXXX.h5`
- **Alert Triggers**: 
  - Data completeness < 90%: WARNING
  - 3 retries exhausted: CRITICAL
- **Performance**: Vectorized OHLCV parsing (10-100x faster than iterrows)
- **Retry Strategy**: Exponential backoff (5s, 10s, 20s)

## CI/CD

GitHub Actions workflow in [.github/workflows/test.yml](.github/workflows/test.yml):
- **Unit tests**: Run on every push to `main`, `develop`, `zy_dve_2`, and PRs to `main`
- **Integration tests**: Only on `main` pushes, scheduled runs, or manual dispatch
- **Artifacts**: Test results retained for 30 days

Branches: `main` (production), `develop` (staging), `zy_dve_2` (dev feature branch)
