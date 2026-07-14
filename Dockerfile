# Main Server Dockerfile
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Ensure debug_output directory exists
RUN mkdir -p /app/debug_output && chmod 777 /app/debug_output

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy utility scripts (scheduler, aggregator, etc.)
COPY prediction_scheduler.py .
COPY run_aggregator.py .
COPY query_dumper.py .

# Copy data file (needed for the server to work)
COPY torchserve/data/last_max_window_10000_ochlv.hdf ./torchserve/data/

# Set environment variables
ENV TS_BASE_URL=http://torchserve:8080
ENV TS_MODEL=ple
ENV HDF_PATH=/app/torchserve/data/last_max_window_10000_ochlv.hdf
ENV HDF_KEY=my_data
ENV NUM_FEATURES=111
ENV HISTORY_WINDOW=14000
ENV TS_TIMEOUT=30

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["python", "-m", "src.serving.server"]