#!/bin/bash

# Docker Compose deployment script for ML Server Demo
# This script starts both TorchServe and the main server

set -e

# -----------------------------
# 解析命令行参数
# -----------------------------
OFFLINE_DEBUG_MODE=0  # 默认禁用
WATCH_MODE=0  # 默认禁用
LOG_LEVEL="INFO"  # 默认日志级别

# 解析参数
for arg in "$@"; do
    case $arg in
        -offline_debug_mode|--offline-debug-mode)
            OFFLINE_DEBUG_MODE=1
            LOG_LEVEL="DEBUG"
            shift
            ;;
        -w|--watch)
            WATCH_MODE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -offline_debug_mode, --offline-debug-mode    Enable offline debug mode (default: disabled)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                     # Start with offline_debug_mode disabled"
            echo "  $0 -offline_debug_mode # Start with offline_debug_mode enabled"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# 导出环境变量供 docker-compose 使用
export OFFLINE_DEBUG_MODE
export LOG_LEVEL

echo "🚀 Starting ML Server Demo with Docker Compose..."
echo "⚙️  OFFLINE_DEBUG_MODE=${OFFLINE_DEBUG_MODE}"
echo "⚙️  LOG_LEVEL=${LOG_LEVEL}"

# Create debug_output directory if it doesn't exist
echo "📁 Ensuring debug_output directory exists..."
mkdir -p debug_output
chmod 777 debug_output
echo "✅ debug_output directory ready"

# Build and start services
echo "📦 Building Docker images..."
docker-compose build

echo "🔄 Starting services..."
if [ "$WATCH_MODE" -eq 1 ]; then
    docker-compose up -w
else
    docker-compose up -d
fi

echo "⏳ Waiting for services to be healthy..."

# Wait for TorchServe to be ready
echo "🔍 Checking TorchServe health..."
MAX_WAIT=120
ELAPSED=0
until curl -f http://localhost:8080/ping >/dev/null 2>&1; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "❌ TorchServe failed to start within ${MAX_WAIT}s"
        exit 1
    fi
done
echo "✅ TorchServe is ready!"

# Wait for main server to be ready
echo "🔍 Checking main server health..."
ELAPSED=0
until curl -f http://localhost:8000/health >/dev/null 2>&1; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "❌ Main server failed to start within ${MAX_WAIT}s"
        exit 1
    fi
done
echo "✅ Main server is ready!"

echo ""
echo "🎉 All services are up and running!"
echo ""
echo "📊 Service URLs:"
echo "  • Main API:       http://localhost:8000"
echo "  • Health check:   http://localhost:8000/health"
echo "  • Config info:    http://localhost:8000/config"
echo "  • TorchServe:     http://localhost:8080"
echo "  • TS Management:  http://localhost:8081"
echo "  • TS Metrics:     http://localhost:8082"
echo ""
echo "📂 Debug files will be saved to: $(pwd)/debug_output"
echo ""
echo "🧪 To test the service, run: ./test_deployment.sh"
echo "📋 To view logs, run: docker-compose logs -f"
echo "🛑 To stop services, run: docker-compose down or ./stop.sh"
