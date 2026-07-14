#!/bin/bash

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  运行单元测试${NC}"
echo -e "${GREEN}======================================${NC}"

# 默认配置
RUN_INTEGRATION_TESTS=false
OFFLINE_DEBUG_MODE=false
TEST_OUTPUT_DIR="./unittest_output"
TEST_LOG_DIR="./logs/unittest"  # ✅ 测试专用日志目录
CHECK_LOG_ERRORS=true

# 已知的”预期错误日志”关键字（由错误路径测试触发）
LOG_ERROR_ALLOWLIST=(
    "TestDumper: Failed to write batch"
    "Source file not found:"
    "Invalid pipeline request: Unknown pipeline"
    "分钟对齐校验重试耗尽: expected"
)

check_unexpected_log_errors() {
    local log_file="$1"
    local all_errors_tmp filtered_tmp

    if [ ! -f "$log_file" ]; then
        return 0
    fi

    all_errors_tmp=$(mktemp)
    filtered_tmp=$(mktemp)

    grep -E "ERROR|CRITICAL" "$log_file" > "$all_errors_tmp" || true

    # 无错误日志
    if [ ! -s "$all_errors_tmp" ]; then
        rm -f "$all_errors_tmp" "$filtered_tmp"
        return 0
    fi

    cp "$all_errors_tmp" "$filtered_tmp"
    for pattern in "${LOG_ERROR_ALLOWLIST[@]}"; do
        grep -v -F "$pattern" "$filtered_tmp" > "${filtered_tmp}.next" || true
        mv "${filtered_tmp}.next" "$filtered_tmp"
    done

    # 仍有未白名单错误，视为风险
    if [ -s "$filtered_tmp" ]; then
        echo ""
        echo -e "${RED}检测到未白名单的 ERROR/CRITICAL 日志:${NC}"
        tail -20 "$filtered_tmp"
        rm -f "$all_errors_tmp" "$filtered_tmp"
        return 1
    fi

    echo ""
    echo -e "${GREEN}日志检查通过：仅检测到白名单内的预期 ERROR。${NC}"
    rm -f "$all_errors_tmp" "$filtered_tmp"
    return 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --integration|-i)
            RUN_INTEGRATION_TESTS=true
            shift
            ;;
        --debug|-d)
            OFFLINE_DEBUG_MODE=true
            shift
            ;;
        --output|-o)
            TEST_OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            TEST_LOG_DIR="$2"
            shift 2
            ;;
        --skip-log-check)
            CHECK_LOG_ERRORS=false
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --integration, -i       运行包括集成测试在内的所有测试（需要网络连接）"
            echo "  --debug, -d             开启 DEBUG 模式，保存中间调试数据"
            echo "  --output DIR, -o DIR    指定输出目录（默认: ./unittest_output）"
            echo "  --log-dir DIR           指定日志目录（默认: ./logs/unittest）"
            echo "  --skip-log-check        跳过测试结束后的 ERROR/CRITICAL 日志检查"
            echo "  --help, -h              显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                              # 仅运行单元测试"
            echo "  $0 --integration                # 运行所有测试（包括集成测试）"
            echo "  $0 -i -d                        # 运行集成测试并保存调试数据"
            echo "  $0 -i -d -o ./debug_output      # 指定输出目录"
            echo ""
            echo "环境变量:"
            echo "  RUN_INTEGRATION_TESTS    是否运行集成测试 (true/false)"
            echo "  OFFLINE_DEBUG_MODE       是否保存调试数据 (true/false)"
            echo "  TEST_OUTPUT_DIR          输出目录路径"
            echo "  LOG_DIR                  日志目录路径"
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            echo "使用 '$0 --help' 查看帮助信息"
            exit 1
            ;;
    esac
done

# ✅ 设置测试环境的日志配置（在导出其他变量之前）
export LOG_DIR="$TEST_LOG_DIR"
export LOG_TO_FILE="true"
export LOG_LEVEL="INFO"
export LOG_FILE_NAME="test.log"

# 导出环境变量
export RUN_INTEGRATION_TESTS
export OFFLINE_DEBUG_MODE
export TEST_OUTPUT_DIR

# 显示配置
echo ""
echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}  测试配置${NC}"
echo -e "${CYAN}======================================${NC}"

if [ "$RUN_INTEGRATION_TESTS" = "true" ]; then
    echo -e "${YELLOW}运行模式: ${GREEN}包含集成测试${NC} ${YELLOW}(需要网络连接)${NC}"
else
    echo -e "${YELLOW}运行模式: ${BLUE}仅单元测试${NC} ${YELLOW}(使用 mock)${NC}"
    echo -e "${YELLOW}提示: 使用 './run_unittest.sh --integration' 运行集成测试${NC}"
fi

if [ "$OFFLINE_DEBUG_MODE" = "true" ]; then
    echo -e "${YELLOW}调试模式: ${GREEN}已启用${NC} ${YELLOW}(将保存中间数据)${NC}"
else
    echo -e "${YELLOW}调试模式: ${BLUE}已关闭${NC}"
    echo -e "${YELLOW}提示: 使用 './run_unittest.sh --debug' 启用调试模式${NC}"
fi

echo -e "${YELLOW}输出目录: ${BLUE}${TEST_OUTPUT_DIR}${NC}"
echo -e "${YELLOW}日志目录: ${BLUE}${LOG_DIR}${NC}"
echo -e "${YELLOW}日志文件: ${BLUE}${LOG_DIR}/${LOG_FILE_NAME}${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""

# ✅ 创建必要的目录
echo -e "${BLUE}准备测试环境...${NC}"

# 创建输出目录
if [ "$OFFLINE_DEBUG_MODE" = "true" ] || [ "$RUN_INTEGRATION_TESTS" = "true" ]; then
    echo -e "  ${CYAN}创建输出目录: ${TEST_OUTPUT_DIR}${NC}"
    mkdir -p "$TEST_OUTPUT_DIR"
fi

# ✅ 创建日志目录（测试必需）
echo -e "  ${CYAN}创建日志目录: ${LOG_DIR}${NC}"
mkdir -p "$LOG_DIR"

# 清理旧的测试日志（可选）
if [ -f "${LOG_DIR}/${LOG_FILE_NAME}" ]; then
    echo -e "  ${YELLOW}清理旧日志: ${LOG_DIR}/${LOG_FILE_NAME}${NC}"
    rm -f "${LOG_DIR}/${LOG_FILE_NAME}"
fi

echo ""

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo -e "${BLUE}激活虚拟环境 (.venv)...${NC}"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo -e "${BLUE}激活虚拟环境 (venv)...${NC}"
    source venv/bin/activate
fi

# ✅ 验证环境变量
echo ""
echo -e "${CYAN}环境变量检查:${NC}"
echo -e "  LOG_DIR=${LOG_DIR}"
echo -e "  LOG_TO_FILE=${LOG_TO_FILE}"
echo -e "  LOG_LEVEL=${LOG_LEVEL}"
echo -e "  LOG_FILE_NAME=${LOG_FILE_NAME}"

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  开始运行测试...${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 定义测试文件（自动扫描）
TEST_FILES=()
while IFS= read -r test_file; do
    if [ "$RUN_INTEGRATION_TESTS" != "true" ] && [ "$(basename "$test_file")" = "integration_test.py" ]; then
        continue
    fi
    TEST_FILES+=("$test_file")
done < <(find src/test -maxdepth 1 -type f -name "*_test.py" | sort)

# 提示是否包含集成测试
if [ "$RUN_INTEGRATION_TESTS" = "true" ]; then
    echo -e "${YELLOW}已包含集成测试文件${NC}"
else
    echo -e "${BLUE}已排除集成测试文件（integration_test.py）${NC}"
fi

echo ""

# 运行测试
python3 -m unittest "${TEST_FILES[@]}" -v 2>&1 | tee /tmp/test_output.log

# 捕获管道中第一个命令的退出码（不是 tee 的退出码）
EXIT_CODE=${PIPESTATUS[0]}

# 记录结束时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo -e "${CYAN}======================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ 所有测试通过！${NC}"
    
    # 显示测试统计信息
    echo ""
    echo -e "${BLUE}测试统计:${NC}"
    
    # 提取测试数量
    if [ -f /tmp/test_output.log ]; then
        # 统计测试结果
        TOTAL_TESTS=$(grep -E "^test_" /tmp/test_output.log | wc -l | tr -d ' ')
        PASSED=$(grep -c " \.\.\. ok$" /tmp/test_output.log || echo "0")
        FAILED=$(grep -c " \.\.\. FAIL$" /tmp/test_output.log || echo "0")
        ERRORS=$(grep -c " \.\.\. ERROR$" /tmp/test_output.log || echo "0")
        SKIPPED=$(grep -c " \.\.\. skipped" /tmp/test_output.log || echo "0")
        
        echo -e "${CYAN}  运行测试: ${GREEN}${TOTAL_TESTS}${NC}"
        echo -e "${CYAN}  通过: ${GREEN}${PASSED}${NC}"
        if [ "$FAILED" != "0" ]; then
            echo -e "${CYAN}  失败: ${RED}${FAILED}${NC}"
        fi
        if [ "$ERRORS" != "0" ]; then
            echo -e "${CYAN}  错误: ${RED}${ERRORS}${NC}"
        fi
        if [ "$SKIPPED" != "0" ]; then
            echo -e "${CYAN}  跳过: ${YELLOW}${SKIPPED}${NC}"
        fi
        
        # 提取运行时间统计
        echo ""
        echo -e "${CYAN}  测试模块耗时:${NC}"
        grep "Ran .* test.*in" /tmp/test_output.log | while IFS= read -r line; do
            echo "    $line"
        done
        
        # 清理临时文件
        rm -f /tmp/test_output.log
    fi
    
    echo ""
    echo -e "${YELLOW}总运行时间: ${GREEN}${TOTAL_TIME}秒${NC}"
    
    # 显示输出文件信息
    if [ "$RUN_INTEGRATION_TESTS" = "true" ] || [ "$OFFLINE_DEBUG_MODE" = "true" ]; then
        if [ -d "$TEST_OUTPUT_DIR" ]; then
            FILE_COUNT=$(find "$TEST_OUTPUT_DIR" -type f 2>/dev/null | wc -l)
            if [ $FILE_COUNT -gt 0 ]; then
                echo ""
                echo -e "${BLUE}输出文件已保存至: ${CYAN}${TEST_OUTPUT_DIR}${NC}"
                echo -e "${BLUE}文件列表:${NC}"
                ls -lh "$TEST_OUTPUT_DIR" 2>/dev/null | tail -n +2 | awk '{printf "  %s %s %s\n", $9, $5, $6" "$7" "$8}'
            fi
        fi
    fi
    
    # ✅ 显示日志文件信息
    if [ -d "$LOG_DIR" ]; then
        LOG_FILE_COUNT=$(find "$LOG_DIR" -type f 2>/dev/null | wc -l)
        if [ $LOG_FILE_COUNT -gt 0 ]; then
            echo ""
            echo -e "${BLUE}测试日志已保存至: ${CYAN}${LOG_DIR}${NC}"
            echo -e "${BLUE}日志文件列表:${NC}"
            ls -lh "$LOG_DIR" 2>/dev/null | tail -n +2 | awk '{printf "  %s %s %s\n", $9, $5, $6" "$7" "$8}'
            
            # 显示日志文件的最后几行
            if [ -f "${LOG_DIR}/${LOG_FILE_NAME}" ]; then
                echo ""
                echo -e "${BLUE}最后 10 条日志:${NC}"
                tail -10 "${LOG_DIR}/${LOG_FILE_NAME}" | while IFS= read -r line; do
                    echo "  $line"
                done
            fi
        fi
    fi

    # ✅ 单测通过后，额外检查日志中的 ERROR/CRITICAL（支持白名单）
    if [ "$CHECK_LOG_ERRORS" = "true" ] && [ -f "${LOG_DIR}/${LOG_FILE_NAME}" ]; then
        echo ""
        echo -e "${BLUE}执行日志风险检查...${NC}"
        if ! check_unexpected_log_errors "${LOG_DIR}/${LOG_FILE_NAME}"; then
            EXIT_CODE=1
            echo -e "${RED}日志风险检查失败：存在未白名单错误日志。${NC}"
        fi
    fi
else
    echo -e "${RED}✗ 测试失败 (退出码: $EXIT_CODE)${NC}"
    echo -e "${YELLOW}总运行时间: ${RED}${TOTAL_TIME}秒${NC}"
    
    # 显示失败测试的简要信息
    if [ -f /tmp/test_output.log ]; then
        echo ""
        echo -e "${RED}失败/错误的测试:${NC}"
        grep -E "FAIL|ERROR" /tmp/test_output.log | head -10
        rm -f /tmp/test_output.log
    fi
    
    # ✅ 检查日志文件是否有错误信息
    if [ -f "${LOG_DIR}/${LOG_FILE_NAME}" ]; then
        echo ""
        echo -e "${YELLOW}检查日志文件中的错误...${NC}"
        if grep -q "ERROR\|CRITICAL" "${LOG_DIR}/${LOG_FILE_NAME}"; then
            echo -e "${RED}日志中的错误:${NC}"
            grep "ERROR\|CRITICAL" "${LOG_DIR}/${LOG_FILE_NAME}" | tail -10
        fi
    fi
fi
echo -e "${CYAN}======================================${NC}"

exit $EXIT_CODE
