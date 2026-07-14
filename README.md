# ml4t_online

A minimal service that:
- Loads local HDF data
- Generates features (Alpha102 + rolling z-score)
- Calls a running TorchServe model
- Returns predictions via a Flask API

## Requirements
- Python 3.12+
- Install Python deps:
  ```bash
  python3 -m pip install -r requirements.txt
  ```
- If tables fails to install (Ubuntu):
  ```bash
  sudo apt update && sudo apt install -y libhdf5-dev
  ```

## Run
1) Start TorchServe (in torchserve/):
   ```bash
   bash torchserve/start_torchserve.sh
   ```
   Default: http://127.0.0.1:8080

2) Start the Flask server:
   ```bash
   python -m src.serving.server
   ```

## API
- Health:
  ```bash
  curl http://127.0.0.1:8000/health
  ```
- Predict:
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"model": "dragonnet"}'
  ```

## Configuration (env)
- TS_BASE_URL: TorchServe URL (default: http://127.0.0.1:8080)
- TS_MODEL: Default model name (default: dragonnet)
- HDF_PATH: HDF data path (default: torchserve/data/last_max_window_10000_ochlv.hdf)
- HDF_KEY: HDF key (default: my_data)
- NUM_FEATURES: Number of output features to keep (default: 111)

Example:
```bash
TS_BASE_URL=http://127.0.0.1:8080 \
HDF_PATH=./torchserve/data/last_max_window_10000_ochlv.hdf \
HDF_KEY=my_data \
NUM_FEATURES=111 \
python -m src.serving.server
```

## Pipeline Routing (JSON)

Serving uses JSON-driven pipeline orchestration from:
- `src/serving/pipelines.json`

Runtime executes pipeline `steps` in order by looping config and dispatching node `type` via whitelist registry:
- `src/serving/pipeline_runtime.py`

`POST /predict` routing rules:
- if `pipeline` provided: run that pipeline directly
- else if `method` provided: map by `method_mapping` (e.g. `process_3 -> p_binance_hdf_strategy_dump`)
- else: use `default_pipeline`

## Feature Dimension Contract

- DragonNet serving input expects `111` features.
- `SampleCalculator` keeps backward-compatible params (`batch_size`/`num_features`) and enforces fixed output width.
- If generated features are fewer than target width, calculator will zero-pad as a fallback and log warning.
- Important: rank feature columns are part of model input and should not be excluded from final feature set.

Typical warning when dimensions are short:
```text
Generated feature dim 107 is smaller than target 111, zero padding 4 columns
```
This usually means feature selection/exclusion changed unexpectedly and should be investigated.

## Deployment Troubleshooting

If `/predict` returns `502` with TorchServe `503`, check TorchServe logs first:
```bash
docker logs ml-torchserve --since 5m 2>&1 | tail -n 200
```

A common root cause is feature/model shape mismatch:
```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x107 and 111x128)
```

Use these checks:
1. Verify `NUM_FEATURES=111` in deployment env.
2. Verify calculator logs show `Generated feature dim: 111, target dim: 111`.
3. Run deployment smoke test:
```bash
./test_deployment.sh
```

## Project Structure (src)
- feature/retriever
  - base_retriever.py: retriever interface
  - hdf_file_retriever.py: read DataFrame from HDF
- feature/calculator
  - base_calculator.py: calculator interface
  - sample_calculator.py: Alpha102 + rolling z-score + selection
- serving/server.py: Flask API
- utils/helper.py: rolling z-score utilities

## Data Query

### Feature Dumper Query Tool

The system supports two feature storage methods:
- **HDF5**: `data/hdf_dumper/features_*.h5` - for fast querying and data analysis
- **SQLite**: `data/feature_dumper/feature_dump.db` - for persistent storage and convenient querying

Use the unified query tool `query_dumper.py` to query both data sources at once.

#### Basic Usage

**1. Show statistics (both data sources)**
```bash
python query_dumper.py --stats
```

Output includes:
- HDF5: file count, total records, data size, trading pairs, models, time range
- SQLite: total records, model distribution, trading pair distribution, time range, database size

**2. Query the latest N records**
```bash
python query_dumper.py --latest 5
```

Merges both data sources, sorts by time, and shows the latest records.

**3. Full information display**
```bash
python query_dumper.py --latest 5 --full
```

Shows full content, including OHLCV, feature matrix, and model output.

#### Advanced Queries

**Query HDF5 for a specific date**
```bash
python query_dumper.py --hdf --date 20260206
```

Shows the file statistics for that date and the latest 5 records.

**Query by request_id**
```bash
python query_dumper.py --request-id <request_id>
```

Queries the specified request record from both data sources.

**Query HDF5 only**
```bash
python query_dumper.py --hdf --stats
python query_dumper.py --hdf --latest 10
```

**Query SQLite only**
```bash
python query_dumper.py --sql --stats
python query_dumper.py --sql --latest 10
```

#### Custom Data Source Paths

```bash
# Specify HDF5 directory
python query_dumper.py --hdf-dir /path/to/hdf_dumper --stats

# Specify SQLite database
python query_dumper.py --db /path/to/feature_dump.db --stats

# Specify both
python query_dumper.py --hdf-dir /path/to/hdf --db /path/to/db --latest 5
```

#### Environment Variable Support

```bash
# Set environment variables to avoid specifying paths every time
export HDF_DUMP_DIR=./data/hdf_dumper
export FEATURE_DUMP_DB_PATH=./data/feature_dumper/feature_dump.db

python query_dumper.py --stats
```

#### Complete Usage Examples

```bash
# View overall system statistics
python query_dumper.py --stats

# Inspect the latest 10 prediction records (with full information)
python query_dumper.py --latest 10 --full

# Query HDF5 data for a specific date
python query_dumper.py --hdf --date 20250115

# Query detailed information for a specific request
python query_dumper.py --request-id "abc123xyz" --full

# View only the model-distribution statistics in SQLite
python query_dumper.py --sql --stats
```

#### Output Format

**Statistics example:**
```
================================================================================
📊 HDF5 Statistics
================================================================================
📁 Directory: ./data/hdf_dumper
📄 File count: 3
📈 Total records: 150
💾 Total size: 2.34 MB
📍 Trading pairs: BTCUSDT, ETHUSDT
🤖 Models: dragonnet
📅 Time range: 2025-01-15T10:00:00 ~ 2025-01-15T18:30:00

================================================================================
🗄️  SQLite Statistics
================================================================================
📊 Database: ./data/feature_dumper/feature_dump.db
📈 Total records: 150
💾 Database size: 5.67 MB
🤖 Model distribution:
     - dragonnet: 150
📍 Trading pair distribution:
     - BTCUSDT: 75
     - ETHUSDT: 75
📅 Time range: 2025-01-15 10:00:00 ~ 2025-01-15 18:30:00
```

## Testing

### Run unit tests (no real API access)

```bash
./run_unittest.sh
```

### Run integration tests (accesses real API)

```bash
./run_unittest.sh --integration
```

### CI/CD Configuration

In GitHub Actions or other CI tools:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: ./run_unittest.sh
  
  integration-tests:
    runs-on: ubuntu-latest
    # Only run integration tests on the main branch or on schedule
    if: github.ref == 'refs/heads/main' || github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v2
      - name: Run integration tests
        run: ./run_unittest.sh --integration
        env:
          BINANCE_API_KEY: ${{ secrets.BINANCE_API_KEY }}
          BINANCE_API_SECRET: ${{ secrets.BINANCE_API_SECRET }}
```

## Prediction Scheduler Service

The Prediction Scheduler is a background service that automatically calls the Main Server's `/predict` endpoint at regular intervals. It's useful for continuous model inference, monitoring, and automated data collection.

### Quick Start

#### Docker (Recommended)

The prediction scheduler runs automatically as part of the docker-compose stack:

```bash
docker-compose up prediction-scheduler
```

Or with all services:

```bash
docker-compose up
```

#### Local Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Make sure the main server is running
python -m src.serving.server

# In another terminal, start the scheduler
python prediction_scheduler.py
```

### Configuration

Configure the scheduler via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_URL` | `http://127.0.0.1:8000` | Main server URL |
| `SCHEDULER_INTERVAL_MINUTES` | `1` | Prediction interval in minutes |
| `SCHEDULER_METHOD` | `process_3` | Processing method: `process_1`, `process_2`, or `process_3` |
| `SCHEDULER_RETRY_COUNT` | `1` | Retry attempts on failure |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `TS_TIMEOUT` | `30` | Request timeout in seconds |

#### Docker Example

```yaml
prediction-scheduler:
  environment:
    - SERVER_URL=http://server:8000
    - SCHEDULER_INTERVAL_MINUTES=5
    - SCHEDULER_METHOD=process_2
    - SCHEDULER_RETRY_COUNT=2
    - LOG_LEVEL=DEBUG
```

#### Command Line Example

```bash
# Run scheduler with custom interval (every 5 minutes)
SERVER_URL=http://127.0.0.1:8000 \
SCHEDULER_INTERVAL_MINUTES=5 \
SCHEDULER_METHOD=process_3 \
LOG_LEVEL=DEBUG \
python prediction_scheduler.py
```

### Features

- **Automatic Retry**: Failed requests are automatically retried once after 1 second
- **Health Check**: Validates server connectivity on startup
- **Request Tracking**: Each request gets a unique ID for debugging
- **Detailed Logging**: Shows request IDs, full model output (keys and values), response times
- **Graceful Shutdown**: Properly shuts down via SIGINT/SIGTERM signals
- **Statistics Tracking**: Monitors total calls, success rate, and retry counts

### Output Examples

**Successful Prediction**:
```
[Job #1] ✅ Predict successful request_id=a1b2c3d4
Output:
{
  "y0_treatment": 0.523,
  "y0_control": 0.487,
  "y1_treatment": 0.612,
  "y1_control": 0.498,
  "treatment_effect": 0.125
},msg=Success (took 0.45s)
```

**Failed Request with Retry**:
```
[Job #2] First attempt failed: Timeout after 30s. Retrying...
[Job #2] ✅ Predict successful on retry request_id=e5f6g7h8, msg=Success (took 0.52s)
```

**Persistent Failure**:
```
[Job #3] ❌ Predict failed after retry. request_id=i9j0k1l2, error=Connection error: [Errno 111] Connection refused
```

### Statistics

The scheduler tracks and logs statistics:

```
📊 Scheduler Stats: total_calls=100, successful=98, failed=2, total_retries=5, success_rate=98.0%
```

Print stats at runtime or graceful shutdown (Ctrl+C).

### Logs

Logs are written to:
- **Docker**: `/app/logs/prediction_scheduler/prediction_scheduler.log`
- **Local**: Configured by `src.utils.logger.setup_logger()`

Logs rotate at 10MB with 3 backup files retained.

### Integration with Docker Compose

The scheduler service in docker-compose.yml:
- Depends on the main server being healthy
- Automatically restarts on failure
- Logs to `./logs/prediction_scheduler/`
- Connects via docker network (`ml-network`)

### Development & Debugging

Enable debug logging:

```bash
LOG_LEVEL=DEBUG python prediction_scheduler.py
```

This shows:
- Detailed request/response information
- Full JSON payloads
- Request timings
- Retry logic

### Common Use Cases

1. **Continuous Model Monitoring**: Run every minute to track model predictions
2. **Data Collection**: Gather predictions hourly for analysis
3. **Performance Testing**: Validate inference latency and success rates
4. **System Integration**: Feed predictions to downstream systems

### Troubleshooting

**Connection Refused**:
```
First attempt failed: Connection error: [Errno 111] Connection refused
```
→ Ensure main server is running on the configured `SERVER_URL`

**Timeout Errors**:
```
First attempt failed: Timeout after 30s
```
→ Increase `TS_TIMEOUT` or check server performance

**Invalid JSON Response**:
```
Invalid JSON response
```
→ Check server logs for errors; server may be returning non-JSON content

### Advanced Examples

#### Run with Docker Compose Override

```yaml
# docker-compose.override.yml
services:
  prediction-scheduler:
    environment:
      SCHEDULER_INTERVAL_MINUTES: 2
      SCHEDULER_METHOD: process_1
      LOG_LEVEL: DEBUG
```

Then start normally:
```bash
docker-compose up
```

#### Manual Scheduling for Testing

```bash
# One prediction every 30 seconds
SCHEDULER_INTERVAL_MINUTES=0.5 python prediction_scheduler.py

# One prediction every 2 minutes with verbose output
SCHEDULER_INTERVAL_MINUTES=2 LOG_LEVEL=DEBUG python prediction_scheduler.py
```

#### Monitoring from Another Service

The scheduler can be monitored by checking its container health:

```bash
# Docker
docker ps | grep prediction-scheduler
docker logs ml-prediction-scheduler -f

# Kubernetes equivalent would use liveness probes
```
