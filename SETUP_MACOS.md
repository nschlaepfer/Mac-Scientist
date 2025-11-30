# 🍎 Mac-Scientist: macOS Setup Guide

This guide covers setting up the Pokee Deep Research Agent on macOS with Apple Silicon (M1/M2/M3).

## 📋 Prerequisites

- **macOS**: 13.0 (Ventura) or later recommended
- **Chip**: Apple Silicon (M1, M2, M3 series)
- **Memory**: Minimum 16GB, recommended 32GB+ (this guide optimized for 128GB M3 Max)
- **Python**: 3.10 or later
- **Homebrew**: For installing system dependencies

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/Pokee-AI/PokeeResearchOSS.git
cd PokeeResearchOSS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install macOS-specific dependencies
pip install -r requirements_macos.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# Required API Keys
SERPER_API_KEY=your_serper_api_key_here      # Web search (https://serper.dev)
JINA_API_KEY=your_jina_api_key_here          # Web reading (https://jina.ai)
GEMINI_API_KEY=your_gemini_api_key_here      # Summarization (https://aistudio.google.com)
```

### 3. Start the Tool Server

```bash
python start_tool_server.py --enable-cache
```

### 4. Run the Agent

```bash
# Interactive mode with MLX (recommended for Apple Silicon)
python cli_app.py --serving-mode mlx

# Or use the Gradio web interface
python gradio_app.py --serving-mode mlx
```

---

## 🎛️ Serving Mode Options

### Option 1: MLX (Recommended for Apple Silicon)

MLX is Apple's native machine learning framework, providing the best performance on M-series chips.

**Pros:**
- Native Apple Silicon optimization
- Efficient unified memory usage
- Fast inference with Metal acceleration
- Simple setup - just install `mlx-lm`

**Setup:**
```bash
pip install mlx mlx-lm

# Run directly
python cli_app.py --serving-mode mlx
```

**Memory Usage (7B model):**
- Float16: ~14GB
- 4-bit quantized: ~4GB

---

### Option 2: MLX-LM Server

Run MLX as a server for sharing the model across processes or for stability.

**Start the server:**
```bash
./serve_model_macos.sh mlx
# Or manually:
python -m mlx_lm.server --model PokeeAI/pokee_research_7b --port 8080
```

**Connect the agent:**
```bash
python cli_app.py --serving-mode mlx-server --mlx-server-url http://localhost:8080/v1
```

---

### Option 3: Local (HuggingFace Transformers with MPS)

Uses PyTorch with Metal Performance Shaders (MPS) backend.

**Pros:**
- Familiar HuggingFace ecosystem
- Works with most models out of the box
- Good for debugging and development

**Setup:**
```bash
# Already included in requirements_macos.txt
python cli_app.py --serving-mode local --device mps
```

**Note:** MPS uses float16 for best compatibility. Some operations fall back to CPU.

---

### Option 4: llama.cpp (For Larger Models)

Best for running quantized models (GGUF format) with minimal memory.

**Install:**
```bash
# Via Homebrew
brew install llama.cpp

# Or via pip with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

**Usage:**
```bash
# Download a GGUF model first
# Example: https://huggingface.co/TheBloke/models

# Start server
llama-server -m path/to/model.gguf -ngl 99 --port 8081
```

---

## 💾 Memory Guidelines for M3 Max (128GB)

With 128GB unified memory, you have exceptional headroom:

| Model Size | Precision | Memory Usage | Recommendation |
|------------|-----------|--------------|----------------|
| 7B         | float16   | ~14GB        | ✅ Excellent   |
| 13B        | float16   | ~26GB        | ✅ Excellent   |
| 33B        | float16   | ~66GB        | ✅ Very Good   |
| 70B        | 4-bit     | ~35GB        | ✅ Very Good   |
| 70B        | float16   | ~140GB       | ⚠️ Tight       |

**Reserved Memory:**
- System: ~8-16GB
- Tool server & browser: ~2-4GB
- Context window: ~4-8GB (depends on usage)

**Recommendation:** Use ~100GB for model loading, leaving 28GB for system overhead.

### The PokeeAI Research Model

This project uses **PokeeAI/pokee_research_7b** - a specialized 7B model trained for deep research tasks. With 128GB unified memory, you can run it in full precision with plenty of headroom:

```bash
# Default - uses PokeeAI research model (recommended)
python cli_app.py --serving-mode mlx

# Explicitly specify the model
python cli_app.py --serving-mode mlx --model-path PokeeAI/pokee_research_7b
```

The 7B model uses approximately 14GB in float16, leaving over 100GB free for context and system overhead on your M3 Max.

---

## 🔧 Configuration Tips

### Using the PokeeAI Research Model

```python
from agent.mlx_agent import MLXDeepResearchAgent

# PokeeAI's 7B Deep Research model - optimized for research tasks
agent = MLXDeepResearchAgent(
    model_path="PokeeAI/pokee_research_7b",
)
```

This is the default model and is specifically trained for deep research with web search and content analysis. With your 128GB M3 Max, it runs comfortably in full precision (~14GB usage).

### Performance Tuning

```bash
# Increase max tokens for longer research
python cli_app.py --serving-mode mlx --max-turns 20

# Lower temperature for more focused results
python cli_app.py --serving-mode mlx --temperature 0.3

# Verbose mode for debugging
python cli_app.py --serving-mode mlx --verbose
```

---

## 🌐 Web Interface (Gradio)

The Gradio app provides a user-friendly interface:

```bash
# Start with MLX backend
python gradio_app.py --serving-mode mlx --port 7777

# Open in browser
open http://localhost:7777
```

### Features:
- API key configuration UI
- Tool server management
- Real-time research progress
- Multiple concurrent sessions

---

## 📊 Benchmarking Your Setup

Check your system capabilities:

```bash
./serve_model_macos.sh info
```

This shows:
- Your chip and memory
- Recommended model sizes
- Memory usage estimates

---

## 🔍 Troubleshooting

### "MPS backend out of memory"

```python
# Reduce memory pressure
import torch
torch.mps.empty_cache()

# Or use 4-bit quantization
agent = MLXDeepResearchAgent(use_4bit=True)
```

### "MLX not available"

```bash
# Ensure you're on Apple Silicon
python -c "import platform; print(platform.machine())"  # Should print 'arm64'

# Reinstall MLX
pip uninstall mlx mlx-lm
pip install mlx mlx-lm
```

### "Model download slow"

Models are cached in `~/.cache/huggingface/`. First download may take time:
- 7B model: ~14GB download
- 70B model: ~35-140GB download

### "Tool server not responding"

```bash
# Check if running
curl http://localhost:8888/health

# Restart
pkill -f start_tool_server.py
python start_tool_server.py --enable-cache
```

---

## 📁 Project Structure

```
Mac-Scientist/
├── requirements_macos.txt    # macOS dependencies
├── SETUP_MACOS.md           # This guide
├── serve_model_macos.sh     # Model server helper
├── agent/
│   ├── mlx_agent.py         # MLX native agent
│   ├── simple_agent.py      # PyTorch/MPS agent
│   └── vllm_agent.py        # vLLM server agent
├── cli_app.py               # Command-line interface
├── gradio_app.py            # Web interface
└── start_tool_server.py     # Tool server
```

---

## 🎯 Recommended Workflow

1. **First Run:**
   ```bash
   source venv/bin/activate
   ./serve_model_macos.sh info  # Check your system
   pip install -r requirements_macos.txt
   ```

2. **Daily Use:**
   ```bash
   # Terminal 1: Tool server
   python start_tool_server.py --enable-cache
   
   # Terminal 2: Agent
   python cli_app.py --serving-mode mlx
   ```

3. **For Development:**
   ```bash
   # Use verbose mode
   python cli_app.py --serving-mode mlx --verbose
   
   # Or web interface
   python gradio_app.py --serving-mode mlx
   ```

---

## 📚 Additional Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM Models](https://huggingface.co/mlx-community)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

---

## ⚡ Performance Comparison

On M3 Max (128GB) with 7B model:

| Mode | Tokens/sec | Memory | Notes |
|------|------------|--------|-------|
| MLX (float16) | ~50-80 | 14GB | Best overall |
| MLX (4-bit) | ~80-120 | 4GB | Faster, slight quality loss |
| MPS (float16) | ~30-50 | 14GB | HuggingFace compatible |
| llama.cpp (Q4) | ~60-100 | 4GB | GGUF format required |

---

Happy researching! 🔬🍎

