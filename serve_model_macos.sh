#!/bin/bash
# serve_model_macos.sh - Model serving options for Apple Silicon Macs
#
# This script provides multiple options for serving the model on macOS with Apple Silicon.
# Choose the option that best fits your needs:
#
# Option 1: MLX-LM Server (Recommended for M1/M2/M3)
#   - Native Apple Silicon optimization
#   - Uses unified memory efficiently
#   - Supports chat completions API (OpenAI compatible)
#
# Option 2: llama.cpp with Metal (Alternative)
#   - Highly optimized C++ implementation
#   - Lower memory usage with quantization
#   - Best for running larger models (70B+)
#
# Usage:
#   ./serve_model_macos.sh mlx          # Start MLX-LM server
#   ./serve_model_macos.sh mlx 8080     # Start MLX-LM server on port 8080
#   ./serve_model_macos.sh llama        # Start llama.cpp server
#   ./serve_model_macos.sh info         # Show system info

set -e

# Configuration
# PokeeAI/pokee_research_7b is the primary model for this project
# It's a 7B model specifically trained for deep research tasks
MODEL_ID="${MODEL_ID:-PokeeAI/pokee_research_7b}"
MLX_PORT="${MLX_PORT:-8080}"
LLAMA_PORT="${LLAMA_PORT:-8081}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  🍎 ${GREEN}Mac-Scientist Model Server${NC}                              ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}     Optimized for Apple Silicon                            ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

show_system_info() {
    echo -e "${YELLOW}System Information:${NC}"
    echo "─────────────────────────────────────────────────────────────"
    
    # Chip info
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    echo -e "  ${GREEN}Chip:${NC} $CHIP"
    
    # Memory info
    MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
    MEM_GB=$((MEM_BYTES / 1024 / 1024 / 1024))
    echo -e "  ${GREEN}Unified Memory:${NC} ${MEM_GB}GB"
    
    echo ""
    echo -e "${YELLOW}PokeeAI Research Model:${NC}"
    echo "─────────────────────────────────────────────────────────────"
    echo -e "  ${GREEN}Model:${NC} PokeeAI/pokee_research_7b"
    echo -e "  ${GREEN}Size:${NC}  ~14GB (float16)"
    echo -e "  ${GREEN}Type:${NC}  Deep Research Agent - web search & analysis"
    
    if [ "$MEM_GB" -ge 32 ]; then
        echo ""
        echo -e "  ${GREEN}✓${NC} Your system can run this model with ${GREEN}plenty of headroom${NC}"
        echo -e "  ${GREEN}✓${NC} ~$((MEM_GB - 14 - 16))GB+ available for context and system"
    elif [ "$MEM_GB" -ge 16 ]; then
        echo ""
        echo -e "  ${YELLOW}△${NC} Your system can run this model but may be ${YELLOW}memory constrained${NC}"
    else
        echo ""
        echo -e "  ${RED}✗${NC} Your system may have ${RED}insufficient memory${NC} for this model"
    fi
    
    echo ""
}

start_mlx_server() {
    local port="${1:-$MLX_PORT}"
    
    echo -e "${GREEN}Starting MLX-LM Server...${NC}"
    echo "─────────────────────────────────────────────────────────────"
    echo -e "  Model: ${BLUE}$MODEL_ID${NC}"
    echo -e "  Port:  ${BLUE}$port${NC}"
    echo -e "  API:   ${BLUE}http://localhost:$port/v1${NC}"
    echo ""
    echo -e "${YELLOW}Note:${NC} First run will download the model (~14GB for 7B)"
    echo ""
    
    # Check if mlx_lm is installed
    if ! python3 -c "import mlx_lm" 2>/dev/null; then
        echo -e "${RED}Error: mlx_lm not installed${NC}"
        echo "Install with: pip install mlx mlx-lm"
        exit 1
    fi
    
    # Start the server
    python3 -m mlx_lm.server \
        --model "$MODEL_ID" \
        --port "$port" \
        --host 0.0.0.0
}

start_llama_server() {
    local port="${1:-$LLAMA_PORT}"
    
    echo -e "${GREEN}Starting llama.cpp Server...${NC}"
    echo "─────────────────────────────────────────────────────────────"
    echo -e "  Port: ${BLUE}$port${NC}"
    echo ""
    
    # Check if llama-server is available
    if ! command -v llama-server &> /dev/null; then
        echo -e "${RED}Error: llama-server not found${NC}"
        echo ""
        echo "Install llama.cpp with Metal support:"
        echo ""
        echo "  # Option 1: Via Homebrew"
        echo "  brew install llama.cpp"
        echo ""
        echo "  # Option 2: Build from source"
        echo "  git clone https://github.com/ggerganov/llama.cpp"
        echo "  cd llama.cpp"
        echo "  make LLAMA_METAL=1"
        echo ""
        echo "  # Option 3: Via pip (Python bindings)"
        echo "  CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python"
        exit 1
    fi
    
    # You'll need to download a GGUF model first
    echo -e "${YELLOW}Note:${NC} llama.cpp requires GGUF format models."
    echo "Download from: https://huggingface.co/models?library=gguf"
    echo ""
    echo "Example:"
    echo "  llama-server -m path/to/model.gguf -ngl 99 --port $port"
}

show_usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  mlx [port]     Start MLX-LM server (default port: 8080)"
    echo "  llama [port]   Start llama.cpp server (default port: 8081)"
    echo "  info           Show system information and recommendations"
    echo "  help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_ID       Model to serve (default: PokeeAI/pokee_research_7b)"
    echo "  MLX_PORT       Default port for MLX-LM server (default: 8080)"
    echo "  LLAMA_PORT     Default port for llama.cpp server (default: 8081)"
    echo ""
    echo "Examples:"
    echo "  $0 mlx                    # Start MLX-LM on port 8080"
    echo "  $0 mlx 9000               # Start MLX-LM on port 9000"
    echo "  MODEL_ID=mlx-community/Qwen2.5-7B-Instruct-4bit $0 mlx"
    echo ""
}

# Main
print_header

case "${1:-help}" in
    mlx)
        start_mlx_server "$2"
        ;;
    llama)
        start_llama_server "$2"
        ;;
    info)
        show_system_info
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac

