#!/bin/bash

# run_all_benchmarks.sh - Automated HiCache benchmark runner
# Usage: ./run_all_benchmarks.sh --server-host <host> --server-port <port> [options]

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [ "$DEBUG" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Default configuration
SERVER_HOST="127.0.0.1"
SERVER_PORT="8192"
NUM_REQUESTS="100"
OUTPUT_DIR="./benchmark_results/$(date +%Y%m%d_%H%M%S)"
QUICK_TEST=false
TIMEOUT=3600  # 1 hour timeout per test
DATASET_PATH=""
MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen3-4B-Instruct-2507/main"
DEBUG=false
FAIL_ON_ERROR=true
MAX_RETRIES=3
BASELINE_DIR=""
COMPARE_BASELINE=false
TEST_SCALES=""
CI_MODE=false

# Test configurations for each benchmark
SERVING_CONFIG="--backend sglang --dataset-name sharegpt --data-scenario serving --use-fixed-data"
MULTITURN_CONFIG="--data-scenario multiturn --use-fixed-data"
MIX_CONFIG="--data-scenario mix --use-fixed-data --num-rounds 10 --num-clients 60 --round-ratios 50,25,15,15,10,10,9,8,7,6 --mean-new-tokens-per-round 1000,400,350,300,280,260,240,220,210,200 --mean-return-tokens-per-round 300,300,300,300,300,300,300,300,300,300 --mean-inter-round-interval 1,1,1,1,1,1,1,1,1,1"
LONGCONTEXT_CONFIG="--data-scenario long_context --use-fixed-data"

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --server-host)
                SERVER_HOST="$2"
                shift 2
                ;;
            --server-port)
                SERVER_PORT="$2"
                shift 2
                ;;
            --num-requests)
                NUM_REQUESTS="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --quick-test)
                QUICK_TEST=true
                NUM_REQUESTS="100"
                TIMEOUT=600  # 10 minutes for quick test
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --dataset-path)
                DATASET_PATH="$2"
                shift 2
                ;;
            --model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            --continue-on-error)
                FAIL_ON_ERROR=false
                shift
                ;;
            --max-retries)
                MAX_RETRIES="$2"
                shift 2
                ;;
            --baseline-dir)
                BASELINE_DIR="$2"
                COMPARE_BASELINE=true
                shift 2
                ;;
            --compare-baseline)
                COMPARE_BASELINE=true
                shift
                ;;
            --test-scales)
                TEST_SCALES="$2"
                shift 2
                ;;
            --ci-mode)
                CI_MODE=true
                FAIL_ON_ERROR=true
                DEBUG=false
                shift
                ;;
            --flush-cache)
                FLUSH_CACHE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help message
show_help() {
    cat << EOF
HiCache Benchmark Runner

Usage: $0 --server-host <host> --server-port <port> [options]

Required Arguments:
  --server-host HOST        Server hostname or IP address
  --server-port PORT        Server port number

Optional Arguments:
  --num-requests N          Number of requests per test (default: 100)
  --output-dir DIR          Output directory for results (default: ./benchmark_results/TIMESTAMP)
  --quick-test              Run quick test with 100 requests and 10min timeout
  --timeout SECONDS         Timeout per test in seconds (default: 3600)
  --dataset-path PATH       Path to ShareGPT dataset (auto-detected if not specified)
  --model-path PATH         Model path for tokenizer (default: meta-llama/Llama-3.1-8B-Instruct)
  --debug                   Enable debug output
  --continue-on-error       Continue running other tests if one fails
  --max-retries N           Maximum retries per test (default: 3)
  --baseline-dir DIR        Directory containing baseline results for comparison
  --compare-baseline        Compare results with baseline
  --test-scales "N,M,..."   Run tests with multiple request scales
  --flush-cache              Flush server cache before running benchmarks
  --ci-mode                 Continuous integration mode (strict error handling)
  --help                    Show this help message

Examples:
  # Basic usage
  $0 --server-host 127.0.0.1 --server-port 30000

  # Quick test mode
  $0 --server-host localhost --server-port 30000 --quick-test

  # Custom configuration
  $0 --server-host 192.168.1.100 --server-port 8000 --num-requests 5000 --output-dir ./results

  # Multiple test scales
  $0 --server-host localhost --server-port 30000 --test-scales "100,1000,5000"

  # Regression testing
  $0 --server-host localhost --server-port 30000 --baseline-dir ./baseline_results --compare-baseline

  # CI mode
  $0 --server-host localhost --server-port 30000 --ci-mode
EOF
}

# Validate server connectivity
validate_server() {
    log_info "Validating server connectivity..."
    local health_url="http://${SERVER_HOST}:${SERVER_PORT}/health"
    local models_url="http://${SERVER_HOST}:${SERVER_PORT}/v1/models"
    local chat_url="http://${SERVER_HOST}:${SERVER_PORT}/v1/chat/completions"

    # Try health endpoint first
    local health_response=$(curl -s --max-time 10 "$health_url" 2>&1)
    if [ $? -eq 0 ]; then
        log_info "✓ Server health check passed"
        log_debug "Health response: $health_response"
    else
        log_warn "✗ Health endpoint failed: $health_response"
    fi

    # Try models endpoint
    local models_response=$(curl -s --max-time 10 "$models_url" 2>&1)
    if [ $? -eq 0 ]; then
        log_info "✓ Server models endpoint accessible"
        log_debug "Models response: $models_response"
    else
        log_error "✗ Models endpoint failed: $models_response"
        log_error "Cannot connect to server at ${SERVER_HOST}:${SERVER_PORT}"
        log_error "Please ensure the server is running and accessible"
        return 1
    fi

    # Test a simple chat completion request
    log_info "Testing chat completion endpoint..."
    local test_payload='{"model": "test", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1}'
    local chat_response=$(curl -s --max-time 30 -X POST "$chat_url" \
        -H "Content-Type: application/json" \
        -d "$test_payload" 2>&1)

    if echo "$chat_response" | grep -q '"choices"' || echo "$chat_response" | grep -q '"error"'; then
        log_info "✓ Chat completion endpoint responsive"
        log_debug "Chat test response: $chat_response"
    else
        log_warn "⚠ Chat completion endpoint may have issues: $chat_response"
    fi

    return 0
}

# Quick benchmark validation test
validate_benchmark_setup() {
    log_info "Running quick benchmark validation test..."

    # Test serving benchmark with 1 request
    local test_cmd="python bench_serving.py --backend sglang --dataset-name sharegpt"
    test_cmd="$test_cmd --host $SERVER_HOST --port $SERVER_PORT --num-prompts 1"
    test_cmd="$test_cmd --dataset-path $DATASET_PATH --model $MODEL_PATH"
    test_cmd="$test_cmd --data-scenario serving --use-fixed-data --disable-tqdm"

    log_debug "Validation command: $test_cmd"

    log_info "Running validation command directly (output shown below):"
    echo "----------------------------------------"
    if timeout 120 bash -c "$test_cmd"; then
        echo "----------------------------------------"
        log_info "✓ Benchmark validation passed"
        return 0
    else
        echo "----------------------------------------"
        log_error "✗ Benchmark validation failed"
        return 1
    fi
}

# Auto-detect model from server
auto_detect_model() {
    if [ -n "$MODEL_PATH" ]; then
        log_info "Using specified model: $MODEL_PATH"
        return 0
    fi

    local models_url="http://${SERVER_HOST}:${SERVER_PORT}/v1/models"
    local detected_model

    detected_model=$(curl -s --max-time 10 "$models_url" 2>/dev/null | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data and len(data['data']) > 0:
        print(data['data'][0]['id'])
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$detected_model" ]; then
        MODEL_PATH="$detected_model"
        log_info "✓ Auto-detected model: $MODEL_PATH"
        return 0
    else
        log_warn "Could not auto-detect model, benchmarks will run without explicit model specification"
        return 0
    fi
}

# Create output directory structure
create_output_directory() {
    log_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"/{serving,multiturn,mix,longcontext}

    # Create config file
    cat > "$OUTPUT_DIR/config.json" << EOF
{
    "server_host": "$SERVER_HOST",
    "server_port": "$SERVER_PORT",
    "num_requests": "$NUM_REQUESTS",
    "model_path": "$MODEL_PATH",
    "dataset_path": "$DATASET_PATH",
    "quick_test": $QUICK_TEST,
    "timeout": $TIMEOUT,
    "test_timestamp": "$(date -Iseconds)",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "hostname": "$(hostname)",
    "user": "$(whoami)"
}
EOF

    log_info "✓ Created output directory and config"
}

# Auto-detect dataset path
detect_dataset_path() {
    if [ -n "$DATASET_PATH" ]; then
        return 0
    fi

    local common_paths=(
        "/data/models/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"
        "./ShareGPT_V3_unfiltered_cleaned_split.json"
        "~/.cache/huggingface/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    )

    for path in "${common_paths[@]}"; do
        if [ -f "$path" ]; then
            DATASET_PATH="$path"
            log_info "✓ Auto-detected dataset at: $path"
            return 0
        fi
    done

    log_warn "Dataset path not specified and auto-detection failed"
    log_warn "Tests will download ShareGPT dataset automatically"
    DATASET_PATH=""
}

# Run individual benchmark with timeout
run_benchmark() {
    local test_name="$1"
    local script_name="$2"
    local config="$3"

    log_info "Starting $test_name benchmark..."
    
    # Create test directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR/$test_name"
    
    local log_file="$OUTPUT_DIR/$test_name/${test_name}.log"
    local metrics_file="$OUTPUT_DIR/$test_name/${test_name}_metrics.json"
    local summary_file="$OUTPUT_DIR/$test_name/${test_name}_summary.txt"

    # Construct command based on benchmark type
    local cmd="python benchmark/hicache/$script_name"
    cmd="$cmd --host $SERVER_HOST --port $SERVER_PORT"

    # Add flush-cache flag if requested
    if [ "$FLUSH_CACHE" = true ]; then
        cmd="$cmd --flush-cache"
    fi

    # Add benchmark-specific parameters
    case "$test_name" in
        "serving")
            cmd="$cmd --num-prompts $NUM_REQUESTS"
            if [ -n "$MODEL_PATH" ]; then
                cmd="$cmd --model $MODEL_PATH"
            fi
            ;;
        "multiturn")
            cmd="$cmd --num-clients $NUM_REQUESTS"
            if [ -n "$MODEL_PATH" ]; then
                cmd="$cmd --model-path $MODEL_PATH"
            fi
            ;;
        "mix")
            # Mix benchmark uses duration instead of request count
            cmd="$cmd --duration 60"  # Short duration for quick test
            if [ -n "$MODEL_PATH" ]; then
                cmd="$cmd --model-path $MODEL_PATH"
            fi
            ;;
        "longcontext")
            cmd="$cmd --num-clients $NUM_REQUESTS"
            if [ -n "$MODEL_PATH" ]; then
                cmd="$cmd --model-path $MODEL_PATH"
            fi
            ;;
    esac

    if [ -n "$DATASET_PATH" ]; then
        cmd="$cmd --dataset-path $DATASET_PATH"
    fi

    cmd="$cmd $config"

    log_debug "Running command: $cmd"

    # Run with timeout and log output
    local start_time=$(date +%s)
    
    if timeout $TIMEOUT bash -c "$cmd" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_info "✓ $test_name completed successfully in ${duration}s"
        
        # Extract metrics from log file
        extract_metrics "$test_name" "$log_file" "$metrics_file"
        
        # Generate summary
        generate_test_summary "$test_name" "$log_file" "$summary_file" "$duration"
        
        return 0
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [ $exit_code -eq 124 ]; then
            log_error "✗ $test_name timed out after ${TIMEOUT}s (total run time: ${duration}s)"
        else
            log_error "✗ $test_name failed with exit code $exit_code (total run time: ${duration}s)"
        fi
        
        # Show last few lines of log for debugging
        if [ -f "$log_file" ]; then
            log_debug "Last 10 lines of log:"
            tail -10 "$log_file" | while read line; do
                log_debug "  $line"
            done
        fi
        
        if [ "$FAIL_ON_ERROR" = "true" ]; then
            log_error "Stopping benchmark suite due to test failure"
            exit 1
        fi
        
        return $exit_code
    fi
}

# Extract metrics from log file
extract_metrics() {
    local test_name="$1"
    local log_file="$2"
    local metrics_file="$3"

    log_debug "Extracting metrics for $test_name"

    # Initialize metrics with default values
    local ttft_p50=0
    local ttft_p90=0
    local latency_p50=0
    local latency_p90=0
    local throughput=0
    local cache_hit_rate=0

    # Try multiple patterns for each metric
    if [ -f "$log_file" ] && [ -s "$log_file" ]; then
        # Try to extract TTFT P50 (支持多种格式: Median TTFT, TTFT P50, TTFT p50)
        local ttft_line=$(grep -i "median.*ttft\|ttft.*p50\|ttft.*p50" "$log_file" | tail -1)
        if [ -n "$ttft_line" ]; then
            ttft_p50=$(echo "$ttft_line" | grep -o "[0-9]\+\.[0-9]\+\|[0-9]\+" | head -1)
        fi
        # 确保不为空
        ttft_p50=${ttft_p50:-0}

        # Try to extract TTFT P90
        local ttft_p90_line=$(grep -i "p90.*ttft\|ttft.*p90" "$log_file" | tail -1)
        if [ -n "$ttft_p90_line" ]; then
            ttft_p90=$(echo "$ttft_p90_line" | grep -o "[0-9]\+\.[0-9]\+\|[0-9]\+" | head -1)
        fi
        ttft_p90=${ttft_p90:-0}

        # Try to extract Latency P50 (支持多种格式: Median E2E Latency, Latency P50)
        local latency_line=$(grep -i "median.*latency\|latency.*p50" "$log_file" | tail -1)
        if [ -n "$latency_line" ]; then
            latency_p50=$(echo "$latency_line" | grep -o "[0-9]\+\.[0-9]\+\|[0-9]\+" | head -1)
        fi
        latency_p50=${latency_p50:-0}

        # Try to extract Latency P90
        local latency_p90_line=$(grep -i "p90.*latency\|latency.*p90" "$log_file" | tail -1)
        if [ -n "$latency_p90_line" ]; then
            latency_p90=$(echo "$latency_p90_line" | grep -o "[0-9]\+\.[0-9]\+\|[0-9]\+" | head -1)
        fi
        latency_p90=${latency_p90:-0}

        # Try to extract Throughput (优先查找 Request throughput)
        local throughput_line=$(grep -i "request throughput" "$log_file" | tail -1)
        if [ -z "$throughput_line" ]; then
            throughput_line=$(grep -i "throughput" "$log_file" | tail -1)
        fi
        if [ -n "$throughput_line" ]; then
            throughput=$(echo "$throughput_line" | grep -o "[0-9]\+\.[0-9]\+\|[0-9]\+" | head -1)
        fi
        throughput=${throughput:-0}

        # Try to extract Cache Hit Rate
        local cache_line=$(grep -i "cache.*hit" "$log_file" | tail -1)
        if [ -n "$cache_line" ]; then
            cache_hit_rate=$(echo "$cache_line" | grep -o "[0-9]\+\.[0-9]\+\|[0-9]\+" | head -1)
        fi
        cache_hit_rate=${cache_hit_rate:-0}
    else
        log_warn "Log file $log_file is empty or doesn't exist"
    fi

    # Create metrics JSON (确保所有值都有默认值)
    cat > "$metrics_file" << EOF
{
    "test_name": "$test_name",
    "ttft_p50": ${ttft_p50:-0},
    "ttft_p90": ${ttft_p90:-0},
    "latency_p50": ${latency_p50:-0},
    "latency_p90": ${latency_p90:-0},
    "throughput": ${throughput:-0},
    "cache_hit_rate": ${cache_hit_rate:-0},
    "timestamp": "$(date -Iseconds)"
}
EOF

    log_debug "Extracted metrics for $test_name: TTFT P50=$ttft_p50, Throughput=$throughput"
}

# Generate test summary
generate_test_summary() {
    local test_name="$1"
    local log_file="$2"
    local summary_file="$3"
    local duration="$4"

    cat > "$summary_file" << EOF
# $test_name Benchmark Summary

**Test Configuration:**
- Server: $SERVER_HOST:$SERVER_PORT
- Requests: $NUM_REQUESTS
- Duration: ${duration}s
- Timestamp: $(date)

**Performance Metrics:**
$(grep -E "(TTFT|Latency|Throughput|Cache)" "$log_file" | head -20 || echo "Metrics not found in log")

**Test Status:** $([ -f "$OUTPUT_DIR/$test_name/${test_name}_metrics.json" ] && echo "✓ PASSED" || echo "✗ FAILED")

**Log Summary:**
$(tail -50 "$log_file" | head -20)
EOF
}

# Generate comprehensive summary report
generate_summary() {
    log_info "Generating comprehensive summary report..."

    local overall_summary="$OUTPUT_DIR/overall_summary.json"
    local comparison_html="$OUTPUT_DIR/performance_comparison.html"
    local readme_file="$OUTPUT_DIR/README.md"
    local rate_summary="$OUTPUT_DIR/rate_summary.json"

    # Check if we have rate-based results
    local has_rate_results=false
    for rate in 16 8 4 2 1; do
        if [ -d "$OUTPUT_DIR/rate_$rate" ]; then
            has_rate_results=true
            break
        fi
    done

    if [ "$has_rate_results" = true ]; then
        log_info "Generating rate-based summary..."
        
        # Generate rate summary
        echo "{" > "$rate_summary"
        echo "  \"timestamp\": \"$(date -Iseconds)\"," >> "$rate_summary"
        echo "  \"config\": $(cat "$OUTPUT_DIR/config.json")," >> "$rate_summary"
        echo "  \"rates\": {" >> "$rate_summary"
        
        local first_rate=true
        for rate in 16 8 4 2 1; do
            local rate_dir="$OUTPUT_DIR/rate_$rate"
            if [ -d "$rate_dir" ]; then
                if [ "$first_rate" = "false" ]; then
                    echo "," >> "$rate_summary"
                fi
                echo "    \"$rate\": {" >> "$rate_summary"
                
                local first_test=true
                for test_name in serving multiturn mix longcontext; do
                    local metrics_file="$rate_dir/$test_name/${test_name}_metrics.json"
                    if [ -f "$metrics_file" ]; then
                        if [ "$first_test" = "false" ]; then
                            echo "," >> "$rate_summary"
                        fi
                        echo "      \"$test_name\": $(cat "$metrics_file")" >> "$rate_summary"
                        first_test=false
                    fi
                done
                
                echo "    }" >> "$rate_summary"
                first_rate=false
            fi
        done
        
        echo "  }" >> "$rate_summary"
        echo "}" >> "$rate_summary"
        
        # Generate rate comparison HTML
        generate_rate_html_report "$comparison_html"
        
        # Generate rate-based README
        generate_rate_readme "$readme_file"
        
    else
        # Original single-scale summary
        echo "{" > "$overall_summary"
        echo "  \"timestamp\": \"$(date -Iseconds)\"," >> "$overall_summary"
        echo "  \"config\": $(cat "$OUTPUT_DIR/config.json")," >> "$overall_summary"
        echo "  \"results\": {" >> "$overall_summary"

        local first=true
        for test_name in serving multiturn mix longcontext; do
            local metrics_file="$OUTPUT_DIR/$test_name/${test_name}_metrics.json"
            if [ -f "$metrics_file" ]; then
                if [ "$first" = "false" ]; then
                    echo "," >> "$overall_summary"
                fi
                echo "    \"$test_name\": $(cat "$metrics_file")" >> "$overall_summary"
                first=false
            fi
        done

        echo "  }" >> "$overall_summary"
        echo "}" >> "$overall_summary"

        # Generate HTML comparison report
        generate_html_report "$comparison_html"

        # Generate README
        generate_readme "$readme_file"
    fi

    log_info "✓ Generated summary reports"
}

# Generate HTML comparison report
generate_html_report() {
    local html_file="$1"

    cat > "$html_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>HiCache Benchmark Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .passed { color: green; }
        .failed { color: red; }
        .metric { font-weight: bold; }
    </style>
</head>
<body>
    <h1>HiCache Benchmark Results</h1>
    <p>Generated: <span id="timestamp"></span></p>

    <h2>Test Summary</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Status</th>
            <th>TTFT P50 (ms)</th>
            <th>TTFT P90 (ms)</th>
            <th>Latency P50 (ms)</th>
            <th>Latency P90 (ms)</th>
            <th>Throughput (req/s)</th>
            <th>Cache Hit Rate (%)</th>
        </tr>
        <tbody id="results-table">
        </tbody>
    </table>

    <script>
        // Load results from JSON
        fetch('./overall_summary.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('timestamp').textContent = data.timestamp;

                const tbody = document.getElementById('results-table');
                Object.entries(data.results).forEach(([testName, metrics]) => {
                    const row = tbody.insertRow();
                    row.innerHTML = `
                        <td>${testName}</td>
                        <td class="passed">✓ PASSED</td>
                        <td>${metrics.ttft_p50}</td>
                        <td>${metrics.ttft_p90}</td>
                        <td>${metrics.latency_p50}</td>
                        <td>${metrics.latency_p90}</td>
                        <td>${metrics.throughput}</td>
                        <td>${metrics.cache_hit_rate}</td>
                    `;
                });
            })
            .catch(error => {
                console.error('Error loading results:', error);
            });
    </script>
</body>
</html>
EOF
}

# Generate README file
generate_readme() {
    local readme_file="$1"

    cat > "$readme_file" << EOF
# HiCache Benchmark Results

Generated: $(date)

## Test Configuration

- **Server**: $SERVER_HOST:$SERVER_PORT
- **Requests per test**: $NUM_REQUESTS
- **Model**: $MODEL_PATH
- **Dataset**: $DATASET_PATH
- **Quick test mode**: $QUICK_TEST

## Test Results

$(for test_name in serving multiturn mix longcontext; do
    local summary_file="$OUTPUT_DIR/$test_name/${test_name}_summary.txt"
    if [ -f "$summary_file" ]; then
        echo "### $test_name"
        echo ""
        cat "$summary_file"
        echo ""
    fi
done)

## Files Structure

- \`config.json\` - Test configuration
- \`overall_summary.json\` - Combined metrics from all tests
- \`performance_comparison.html\` - Interactive HTML report
- \`serving/\` - General serving benchmark results
- \`multiturn/\` - Multi-turn conversation benchmark results
- \`mix/\` - Mixed workload benchmark results
- \`longcontext/\` - Long context benchmark results

## Usage

To view the interactive report, open \`performance_comparison.html\` in a web browser.

For detailed logs, check the individual test directories.
EOF
}

# Compare with baseline results
compare_with_baseline() {
    if [ ! -d "$BASELINE_DIR" ]; then
        log_warn "Baseline directory not found: $BASELINE_DIR"
        return 1
    fi

    log_info "Comparing results with baseline..."

    local baseline_summary="$BASELINE_DIR/overall_summary.json"
    local current_summary="$OUTPUT_DIR/overall_summary.json"
    local comparison_file="$OUTPUT_DIR/baseline_comparison.json"

    if [ ! -f "$baseline_summary" ]; then
        log_warn "Baseline summary not found: $baseline_summary"
        return 1
    fi

    # Simple comparison (can be enhanced with more sophisticated analysis)
    python3 << EOF
import json
import sys

try:
    with open('$baseline_summary') as f:
        baseline = json.load(f)
    with open('$current_summary') as f:
        current = json.load(f)

    comparison = {
        'baseline_timestamp': baseline.get('timestamp', 'unknown'),
        'current_timestamp': current.get('timestamp', 'unknown'),
        'comparisons': {}
    }

    for test_name in ['serving', 'multiturn', 'mix', 'longcontext']:
        if test_name in baseline.get('results', {}) and test_name in current.get('results', {}):
            base = baseline['results'][test_name]
            curr = current['results'][test_name]

            comparison['comparisons'][test_name] = {
                'ttft_p50_change': (curr['ttft_p50'] - base['ttft_p50']) / base['ttft_p50'] * 100 if base['ttft_p50'] > 0 else 0,
                'latency_p50_change': (curr['latency_p50'] - base['latency_p50']) / base['latency_p50'] * 100 if base['latency_p50'] > 0 else 0,
                'throughput_change': (curr['throughput'] - base['throughput']) / base['throughput'] * 100 if base['throughput'] > 0 else 0,
            }

    with open('$comparison_file', 'w') as f:
        json.dump(comparison, f, indent=2)

    print("Comparison completed successfully")

except Exception as e:
    print(f"Error during comparison: {e}")
    sys.exit(1)
EOF

    log_info "✓ Baseline comparison completed"
}

# Main execution flow
main() {
    echo "========================================="
    echo "HiCache Benchmark Suite"
    echo "========================================="

    parse_args "$@"

    # Validate inputs
    if [ -z "$SERVER_HOST" ] || [ -z "$SERVER_PORT" ]; then
        log_error "Server host and port are required"
        show_help
        exit 1
    fi

    validate_server
    detect_dataset_path

    # Skip validation for now, run benchmarks directly
    log_info "Skipping validation, running benchmarks directly..."

    create_output_directory

    echo ""
    log_info "Starting HiCache benchmark suite at $(date)"
    log_info "Server: ${SERVER_HOST}:${SERVER_PORT}"
    log_info "Requests per test: ${NUM_REQUESTS}"
    log_info "Output directory: ${OUTPUT_DIR}"
    log_info "Quick test mode: ${QUICK_TEST}"
    echo ""

    # Handle multiple test scales
    if [ -n "$TEST_SCALES" ]; then
        log_info "Running tests with multiple scales: $TEST_SCALES"
        IFS=',' read -ra SCALES <<< "$TEST_SCALES"
        for scale in "${SCALES[@]}"; do
            log_info "Running tests with $scale requests..."
            NUM_REQUESTS="$scale"

            # Create subdirectory for this scale
            local scale_dir="$OUTPUT_DIR/scale_$scale"
            mkdir -p "$scale_dir"

            # Update output directory temporarily
            local original_output="$OUTPUT_DIR"
            OUTPUT_DIR="$scale_dir"

            # Run all benchmarks for this scale
            run_benchmark "serving" "bench_serving.py" "$SERVING_CONFIG"
            run_benchmark "multiturn" "bench_multiturn.py" "$MULTITURN_CONFIG"
            #run_benchmark "mix" "bench_mix.py" "$MIX_CONFIG"
            run_benchmark "longcontext" "bench_long_context.py" "$LONGCONTEXT_CONFIG"

            # Restore output directory
            OUTPUT_DIR="$original_output"
        done
    else
        # Run all 4 benchmarks with multiple request rates
        log_info "Running benchmarks with predefined request rates: 8, 4, 2, 1"
        
        # Define request rates to test
        REQUEST_RATES="1 2 4 8"
        
        for rate in $REQUEST_RATES; do
            log_info "Running benchmarks at ${rate} req/s..."
            
            # Create subdirectory for this rate
            local rate_dir="$OUTPUT_DIR/rate_${rate}"
            mkdir -p "$rate_dir"
            
            # Update output directory temporarily
            local original_output="$OUTPUT_DIR"
            OUTPUT_DIR="$rate_dir"
            
            # Run all benchmarks with this rate
            run_benchmark "serving" "bench_serving.py" "$SERVING_CONFIG --request-rate $rate --disable-auto-run"
            run_benchmark "multiturn" "bench_multiturn.py" "$MULTITURN_CONFIG --request-rate $rate --disable-auto-run"
            #run_benchmark "mix" "bench_mix.py" "$MIX_CONFIG --request-rate $rate --disable-auto-run"
            run_benchmark "longcontext" "bench_long_context.py" "$LONGCONTEXT_CONFIG --request-rate $rate --disable-auto-run"
            
            # Restore output directory
            OUTPUT_DIR="$original_output"
            
            # Wait between rates to let system stabilize
            if [ "$rate" != "1" ]; then
                log_info "Waiting 10 seconds before next rate test..."
                sleep 10
            fi
        done
    fi

    # Generate comprehensive report
    generate_summary

    # Compare with baseline if requested
    if [ "$COMPARE_BASELINE" = "true" ]; then
        compare_with_baseline
    fi

    echo ""
    log_info "========================================="
    log_info "HiCache benchmark suite completed at $(date)"
    log_info "========================================="
    log_info "Results saved to: ${OUTPUT_DIR}"
    log_info "View HTML report: ${OUTPUT_DIR}/performance_comparison.html"
    log_info "View summary: ${OUTPUT_DIR}/README.md"
    log_info "========================================="
}

# Handle script interruption
trap 'log_error "Benchmark suite interrupted"; exit 130' INT TERM

# Run main function with all arguments
# Generate rate-based HTML comparison report
generate_rate_html_report() {
    local html_file="$1"

    cat > "$html_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>HiCache Benchmark Results - Rate Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .passed { color: green; }
        .failed { color: red; }
        .metric { font-weight: bold; }
        .rate-header { background-color: #e6f3ff; }
        .test-header { background-color: #f0f0f0; }
        .chart-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>HiCache Benchmark Results - Rate Comparison</h1>
    <p>Generated: <span id="timestamp"></span></p>

    <div class="chart-container">
        <h2>Throughput vs Request Rate</h2>
        <canvas id="throughputChart" width="800" height="400"></canvas>
    </div>

    <div class="chart-container">
        <h2>TTFT vs Request Rate</h2>
        <canvas id="ttftChart" width="800" height="400"></canvas>
    </div>

    <div class="chart-container">
        <h2>Latency vs Request Rate</h2>
        <canvas id="latencyChart" width="800" height="400"></canvas>
    </div>

    <h2>Detailed Results by Rate</h2>
    <div id="rate-results"></div>

    <script>
        // Load results from JSON
        fetch('./rate_summary.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('timestamp').textContent = data.timestamp;
                
                const rates = Object.keys(data.rates).sort((a, b) => b - a);
                const tests = ['serving', 'multiturn', 'mix', 'longcontext'];
                
                // Prepare data for charts
                const throughputData = {};
                const ttftData = {};
                const latencyData = {};
                
                tests.forEach(test => {
                    throughputData[test] = [];
                    ttftData[test] = [];
                    latencyData[test] = [];
                });
                
                rates.forEach(rate => {
                    tests.forEach(test => {
                        const result = data.rates[rate][test];
                        if (result) {
                            throughputData[test].push(result.throughput || 0);
                            ttftData[test].push(result.ttft_p50 || 0);
                            latencyData[test].push(result.latency_p50 || 0);
                        }
                    });
                });
                
                // Create charts
                createChart('throughputChart', 'Throughput (req/s)', rates, throughputData);
                createChart('ttftChart', 'TTFT P50 (ms)', rates, ttftData);
                createChart('latencyChart', 'Latency P50 (ms)', rates, latencyData);
                
                // Generate detailed tables
                generateRateTables(data.rates, rates, tests);
            })
            .catch(error => {
                console.error('Error loading results:', error);
                document.getElementById('rate-results').innerHTML = '<p>Error loading results. Please check the console for details.</p>';
            });
        
        function createChart(canvasId, label, labels, datasets) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'];
            
            const chartData = {
                labels: labels,
                datasets: Object.keys(datasets).map((test, index) => ({
                    label: test,
                    data: datasets[test],
                    borderColor: colors[index % colors.length],
                    backgroundColor: colors[index % colors.length] + '20',
                    fill: false,
                    tension: 0.1
                }))
            };
            
            new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: label
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Request Rate (req/s)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: label
                            }
                        }
                    }
                }
            });
        }
        
        function generateRateTables(ratesData, rates, tests) {
            const container = document.getElementById('rate-results');
            let html = '';
            
            rates.forEach(rate => {
                html += `<h3>Rate: ${rate} req/s</h3>`;
                html += '<table>';
                html += '<tr><th>Test</th><th>TTFT P50 (ms)</th><th>TTFT P90 (ms)</th><th>Latency P50 (ms)</th><th>Latency P90 (ms)</th><th>Throughput (req/s)</th><th>Cache Hit Rate (%)</th></tr>';
                
                tests.forEach(test => {
                    const result = ratesData[rate][test];
                    if (result) {
                        html += `<tr>
                            <td>${test}</td>
                            <td>${result.ttft_p50?.toFixed(2) || 'N/A'}</td>
                            <td>${result.ttft_p90?.toFixed(2) || 'N/A'}</td>
                            <td>${result.latency_p50?.toFixed(2) || 'N/A'}</td>
                            <td>${result.latency_p90?.toFixed(2) || 'N/A'}</td>
                            <td>${result.throughput?.toFixed(2) || 'N/A'}</td>
                            <td>${result.cache_hit_rate?.toFixed(2) || 'N/A'}</td>
                        </tr>`;
                    }
                });
                
                html += '</table><br>';
            });
            
            container.innerHTML = html;
        }
    </script>
</body>
</html>
EOF
}

# Generate rate-based README
generate_rate_readme() {
    local readme_file="$1"

    cat > "$readme_file" << 'EOF'
# HiCache Benchmark Results - Rate Comparison

## Overview
This benchmark suite tests HiCache performance across different request rates: 8, 4, 2, and 1 requests per second.

## Test Configuration
- Server: ${SERVER_HOST}:${SERVER_PORT}
- Request Rates: 8, 4, 2, 1 req/s
- Tests: serving, multiturn, mix, longcontext
- Timestamp: $(date)

## Results Summary

### Performance Trends
1. **Throughput**: Expected to scale with request rate until system saturation
2. **TTFT (Time To First Token)**: Should remain stable or increase slightly with higher rates
3. **Latency**: May increase with higher request rates due to queuing
4. **Cache Hit Rate**: Should improve with repeated requests at lower rates

### Key Observations
$(for rate in 8 4 2 1; do
    if [ -d "rate_$rate" ]; then
        echo "- **${rate} req/s**: "
        for test in serving multiturn mix longcontext; do
            metrics_file="rate_$rate/$test/${test}_metrics.json"
            if [ -f "$metrics_file" ]; then
                throughput=$(grep -o '"throughput":[0-9.]*' "$metrics_file" | cut -d: -f2)
                ttft=$(grep -o '"ttft_p50":[0-9.]*' "$metrics_file" | cut -d: -f2)
                echo "  - $test: throughput=${throughput:-N/A} req/s, TTFT=${ttft:-N/A} ms"
            fi
        done
    fi
done)

## Detailed Results
Each request rate has its own directory with detailed results:
$(for rate in 8 4 2 1; do
    if [ -d "rate_$rate" ]; then
        echo "- [rate_$rate/](rate_$rate/): Results at ${rate} req/s"
    fi
done)

## Analysis
The rate-based testing helps identify:
1. System saturation points
2. Optimal operating range
3. Cache effectiveness at different load levels
4. Scalability characteristics

## Files
- `rate_summary.json`: Consolidated results for all rates
- `performance_comparison.html`: Interactive charts and tables
- `config.json`: Test configuration
- `rate_*/`: Individual rate results

## Next Steps
1. Review performance trends across rates
2. Identify bottlenecks at high request rates
3. Optimize cache configuration based on rate performance
4. Consider mixed workload testing with varying rates
EOF
}

main "$@"