# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Machine Learning Server Demo** that implements a financial trading system with the following components:

1. **Main Server** - A Python FastAPI service (`src/serving/server.py`) that handles market data retrieval, feature calculation, ML model inference via TorchServe, and order execution on Binance Futures.

2. **TorchServe** - A PyTorch model serving container for the `dragonnet` model.

3. **Operations Modules** - Modular components for data retrieval, calculation, adaptation, and order execution.

## Common Commands

### Testing

```bash
# Run all unit tests
./run_unittest.sh

# Run only integration tests
./run_unittest.sh --integration

# Run a single test file
python -m pytest src/test/pipeline_test.py -v

# Run a specific test class
python -m pytest src/test/binance_adaptor_test.py::BinanceAdaptorTest -v

# Run a specific test method
python -m pytest src/test/calculator_test.py::TestSampleCalculator::test_sample_calculation -v
```

Tests output to `unittest_output/` directory.

### Linting

```bash
# Run flake8 on the source code
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Run with complexity check (warnings only)
flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Install linting tools if needed
pip install flake8 pylint
```

### Docker Deployment

```bash
# Start all services (server + torchserve)
docker compose up --build

# Run in detached mode
docker compose up -d --build

# Start only torchserve
docker compose up torchserve --build

# Execute commands in running server container
docker exec -it ml-server /bin/bash

# View logs
docker compose logs -f server
docker compose logs -f torchserve

# Stop services
docker compose down
```

Environment variables for Docker:
- `OFFLINE_DEBUG_MODE` - Set to 1 for offline testing
- `LOG_LEVEL` - Set to DEBUG, INFO, WARNING, etc.

### Querying Features

```bash
# Query features for multiple pairs
python query_features.py BTCUSDT ETHUSDT

# Query with custom timestamp
python query_features.py BTCUSDT --timestamp 1704067200
```

## Architecture

### Operation Pipeline

The codebase uses a modular pipeline architecture where operations are chained together to process data:

1. **Retriever** (`src/ops/retriever/`) - Fetches market data
   - `BinanceRetriever` - Live data from Binance API
   - `BinanceCachedRetriever` - Cached data from HDF files
   - `HDFFileRetriever` - Direct HDF file access

2. **Calculator** (`src/ops/calculator/`) - Computes features/alphas
   - `SampleCalculator` - Implements Alpha formulas (e.g., Alpha102)

3. **Adaptor** (`src/ops/adaptor/`) - Transforms data formats
   - `BinanceAdaptor` - Converts calculator output to Binance-compatible format

4. **Strategy** (`src/ops/strategy/`) - Executes trading logic
   - `FirstStrategy` - Main strategy implementation using the pipeline

All components inherit from base classes in `src/ops/base_op.py` and follow a consistent `process()` interface.

### Order Execution Flow

Orders flow through multiple executors in `src/ops/executor/`:

1. **AccountRouter** - Routes orders to appropriate accounts
2. **OrderPlacer** - Places orders via Binance API
3. **OrderCloser** - Handles order closure/cancellation
4. **OrderFinalizer** - Finalizes completed orders

### Dual-Mode Architecture

The system supports two operational modes controlled by `OFFLINE_DEBUG_MODE`:

- **Online Mode** (`OFFLINE_DEBUG_MODE=0`): Full Binance API integration
- **Offline Mode** (`OFFLINE_DEBUG_MODE=1`): Uses cached HDF data, useful for development

### TorchServe Integration

The main server communicates with TorchServe for ML inference:
- TorchServe runs on port 8080 (inference), 8081 (management), 8082 (metrics)
- Model used: `dragonnet`
- Input: 120 features × 240 timesteps
- Connection configured via `TS_BASE_URL` environment variable

### Data Storage

- **HDF Files** - Historical market data stored in `/app/data/*.hdf`
- **SQLite Database** - Order and score tracking (`src/storage/order_repo.py`, `score_repo.py`)
- **Debug Output** - Intermediate data saved to `debug_output/` for troubleshooting

## Key Components

### Server Entry Point

`src/serving/server.py` is the main FastAPI application that:
- Initializes the operation pipeline
- Handles `/features` endpoint for feature queries
- Manages health checks at `/health`
- Integrates with TorchServe for predictions

### Configuration

- `src/config/constants.py` - Environment-aware configuration loader
- Uses `python-dotenv` for loading `.env` files
- Supports both localhost and Docker network configurations

### Testing Strategy

Tests follow naming convention: `*_test.py` in `src/test/`:
- Unit tests mock external dependencies
- Integration tests require valid Binance API keys (`BINANCE_API_KEY`, `BINANCE_API_SECRET`)
- CI runs on: push to main/develop/zy_dve_2 branches, PRs to main, and daily at 2 AM UTC

## Important Files

- `docker-compose.yml` - Service orchestration with health checks
- `deploy.sh` / `test_deployment.sh` - Deployment scripts
- `requirements.txt` - Python dependencies
- `.github/workflows/tests.yml` - CI/CD pipeline configuration