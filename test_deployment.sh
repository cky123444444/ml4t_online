#!/bin/bash

# Test script for the ML Server Demo deployment
# Tests both health endpoints and the main prediction API

set -e

SKIP_DEPLOY=${SKIP_DEPLOY:-false}

# parse args
for arg in "$@"; do
    case $arg in
        -s|--skip_deploy)
        SKIP_DEPLOY=true
        shift
        ;;
        *)
        ;;
        -h|--help)
        echo "Usage: $0 [-skip_deploy] [-h|--help]"
        exit 0
        ;;
    esac
done

echo "Removing old logs..."
rm -rf ./logs/*

if [ "$SKIP_DEPLOY" = false ]; then
    echo "🚀 Deploying ML Server Demo..."
    bash stop.sh
    bash deploy.sh -offline_debug_mode
else
    echo "⚠️  Skipping deployment cleanup as per user request."
fi

echo "🧪 Testing ML Server Demo deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local url=$1
    local description=$2
    local expected_status=${3:-200}
    
    echo -n "Testing $description... "

    # clear last response
    rm -f /tmp/response
    
    response=$(curl -s -w "%{http_code}" -o /tmp/response "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        if [ -f /tmp/response ]; then
            echo "Response: $(cat /tmp/response | jq . 2>/dev/null || cat /tmp/response)"
        fi
    else
        echo -e "${RED}❌ FAIL (HTTP $response)${NC}"
        if [ -f /tmp/response ]; then
            echo "Response: $(cat /tmp/response)"
        fi
        return 1
    fi
    echo ""
}

# Function to test POST endpoint
test_post_endpoint() {
    local url=$1
    local data=$2
    local description=$3
    local expected_status=${4:-200}
    
    echo -n "Testing $description... "

    # clear last response
    rm -f /tmp/response
    
    response=$(curl -s -w "%{http_code}" -o /tmp/response \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        if [ -f /tmp/response ]; then
            echo "Response: $(cat /tmp/response | jq . 2>/dev/null || cat /tmp/response)"
        fi
    else
        echo -e "${RED}❌ FAIL (HTTP $response)${NC}"
        if [ -f /tmp/response ]; then
            echo "Response: $(cat /tmp/response)"
        fi
        return 1
    fi
    echo ""
}

echo "🔍 Testing service endpoints..."
echo ""

# Test TorchServe ping
test_endpoint "http://localhost:8080/ping" "TorchServe ping"

# Test main server health
test_endpoint "http://localhost:8000/health" "Main server health"

# Test main server config
test_endpoint "http://localhost:8000/config" "Main server config"

# Test main prediction endpoint
# echo -e "${YELLOW}🎯 Testing main prediction API. Process 2...${NC}"
# test_post_endpoint "http://localhost:8000/predict" '{"model": "dragonnet", "req_debug_mode": true, "method": "process_2"}' "DragonNet prediction"

# test process 3
echo -e "${YELLOW}🔢 Testing main prediction API. Process 3...${NC}"
test_post_endpoint "http://localhost:8000/predict" '{"model": "ple", "req_debug_mode": true, "method": "process_3"}' "PLE prediction"

# Test feature dump status
echo -e "${YELLOW}📊 Testing feature dump status...${NC}"
test_endpoint "http://localhost:8000/feature_dump/status" "Feature dump status"

# Test prediction scheduler
echo -e "${YELLOW}🔄 Waiting for prediction scheduler to run...${NC}"
sleep 5  # Wait 5 seconds for at least one prediction call

# Check prediction scheduler logs
SCHEDULER_LOG="./logs/prediction_scheduler/prediction_scheduler.log"
if [ -f "$SCHEDULER_LOG" ]; then
    echo -e "${BLUE}✅ Prediction scheduler log exists: $SCHEDULER_LOG${NC}"
    echo -e "${BLUE}📝 Recent scheduler logs:${NC}"
    tail -10 "$SCHEDULER_LOG"
else
    echo -e "${YELLOW}⚠️  Prediction scheduler log not found: $SCHEDULER_LOG${NC}"
fi

echo ""
echo -e "${GREEN}🎉 All tests completed!${NC}"

# Query dumped features (both HDF5 and SQLite)
if command -v python3.10 &> /dev/null; then
    echo ""
    echo -e "${CYAN}🔍 Checking dumped features (HDF5 + SQLite)...${NC}"
    echo ""
    
    # Paths based on docker-compose.yml volume mapping: ./data:/app/data
    HDF_DIR="./data/hdf_dumper"
    DB_PATH="./data/feature_dumper/feature_dump.db"
    
    # Check HDF5 directory
    if [ -d "$HDF_DIR" ] && [ "$(ls -A $HDF_DIR 2>/dev/null)" ]; then
        echo -e "${BLUE}✅ HDF5 目录存在: $HDF_DIR${NC}"
    else
        echo -e "${YELLOW}⚠️  HDF5 目录为空或不存在: $HDF_DIR${NC}"
    fi
    
    # Check SQLite database
    if [ -f "$DB_PATH" ]; then
        echo -e "${BLUE}✅ SQLite 数据库存在: $DB_PATH${NC}"
    else
        echo -e "${YELLOW}⚠️  SQLite 数据库不存在: $DB_PATH${NC}"
    fi
    
    echo ""
    
    # Run unified query tool
    if [ -f "query_dumper.py" ]; then
        echo -e "${BLUE}📊 统计信息 (HDF5 + SQLite):${NC}"
        python3.10 query_dumper.py \
            --hdf-dir "$HDF_DIR" \
            --db "$DB_PATH" \
            --stats 2>&1 || true
        
        echo ""
        echo -e "${BLUE}📝 最新 5 条记录:${NC}"
        python3.10 query_dumper.py \
            --hdf-dir "$HDF_DIR" \
            --db "$DB_PATH" \
            --latest 5 2>&1 || true
    else
        echo -e "${YELLOW}⚠️  query_dumper.py 不存在，使用 query_features.py${NC}"
        if [ -f "query_features.py" ]; then
            python3.10 query_features.py --db "$DB_PATH" --stats 2>&1 || true
        fi
    fi
else
    echo -e "${YELLOW}⚠️  python3.10 not found, skipping feature dump query${NC}"
fi

echo ""

if [ "$SKIP_DEPLOY" = false ]; then
    # stop
    echo "🛑 Stopping ML Server Demo..."
    bash stop.sh
else
    echo "⚠️  Skipping stopping deployment as per user request."
fi
