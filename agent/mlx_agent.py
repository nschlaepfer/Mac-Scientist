# Copyright 2025 Pokee AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MLX Deep Research Agent for Apple Silicon

This agent uses MLX-LM for native Apple Silicon inference.
Optimized for M1/M2/M3 chips with unified memory.

MLX provides:
- Native Metal acceleration
- Unified memory support (no CPU-GPU transfers)
- Efficient memory usage for large models
- Automatic dtype optimization for Apple Silicon
"""

import threading
from typing import Optional

from agent.base_agent import BaseDeepResearchAgent
from logging_utils import setup_colored_logger

logger = setup_colored_logger(__name__)


class MLXDeepResearchAgent(BaseDeepResearchAgent):
    """Deep research agent using MLX for Apple Silicon native inference.
    
    This agent is optimized for M1/M2/M3 Macs with unified memory.
    It uses Apple's MLX framework for maximum performance on Apple Silicon.
    
    Default Model: PokeeAI/pokee_research_7b
    - Specialized 7B model trained for deep research tasks
    - Uses ~14GB in float16 precision
    - Optimized for web search and content analysis
    
    Memory Usage on M3 Max (128GB):
    - PokeeAI/pokee_research_7b: ~14GB - runs with plenty of headroom
    - Context window and tools: ~4-8GB additional
    - System overhead: ~8-16GB
    - Available for other tasks: ~90GB+
    
    MLX automatically converts HuggingFace models to MLX format on first load.
    The converted model is cached for faster subsequent loads.
    """

    # Class-level (singleton) model and tokenizer
    _model = None
    _tokenizer = None
    _model_path = None
    _quantized = None
    _model_lock = None

    def __init__(
        self,
        model_path: str = "PokeeAI/pokee_research_7b",
        tool_config_path: str = "config/tool_config/pokee_tool_config.yaml",
        max_turns: int = 10,
        max_tool_response_length: int = 32768,
        quantize_on_load: bool = False,  # Quantize model after loading (slower startup)
    ):
        """Initialize the MLX agent.

        Args:
            model_path: Path to model or HuggingFace model ID.
                       For 4-bit inference, use pre-quantized models from mlx-community
                       (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit")
            tool_config_path: Path to tool configuration
            max_turns: Maximum conversation turns
            max_tool_response_length: Max length for tool responses
            quantize_on_load: If True, quantize the model to 4-bit after loading.
                             This is slower than using pre-quantized models.
                             Recommended: use pre-quantized models instead.
        """
        super().__init__(
            tool_config_path=tool_config_path,
            max_turns=max_turns,
            max_tool_response_length=max_tool_response_length,
        )

        self.quantize_on_load = quantize_on_load

        # Initialize lock on first use
        if MLXDeepResearchAgent._model_lock is None:
            MLXDeepResearchAgent._model_lock = threading.Lock()

        # Load model only once (singleton pattern)
        with MLXDeepResearchAgent._model_lock:
            if (
                MLXDeepResearchAgent._model is None
                or MLXDeepResearchAgent._model_path != model_path
            ):
                self._load_model(model_path, quantize_on_load)

        self.model = MLXDeepResearchAgent._model
        self.tokenizer = MLXDeepResearchAgent._tokenizer

    @classmethod
    def _load_model(cls, model_path: str, quantize_on_load: bool = False):
        """Load model and tokenizer using MLX-LM.
        
        Args:
            model_path: HuggingFace model ID or local path
            quantize_on_load: Whether to quantize the model after loading
        """
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "MLX-LM is required for Apple Silicon inference. "
                "Install with: pip install mlx mlx-lm"
            )

        logger.info(f"Loading model from {model_path} with MLX...")
        logger.info(f"MLX backend: Metal (Apple Silicon)")
        logger.info(f"Default device: {mx.default_device()}")

        # Check if this is a pre-quantized model (common naming convention)
        is_quantized_model = any(q in model_path.lower() for q in ["-4bit", "-8bit", "4bit", "8bit", "-quantized"])
        
        # Load model with MLX-LM
        cls._model, cls._tokenizer = load(model_path)
        cls._model_path = model_path
        cls._quantized = is_quantized_model
        
        # Optionally quantize after loading (not recommended - use pre-quantized models)
        if quantize_on_load and not is_quantized_model:
            logger.info("Quantizing model to 4-bit (this may take a moment)...")
            try:
                from mlx_lm.utils import quantize as mlx_quantize
                # Quantize to 4-bit with group size 64
                cls._model = mlx_quantize(cls._model, bits=4, group_size=64)
                cls._quantized = True
                logger.info("Model quantized to 4-bit successfully")
            except ImportError:
                logger.warning("mlx_lm.utils.quantize not available, skipping quantization")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}, continuing with full precision")

        # Log memory info
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            total_mem_bytes = int(result.stdout.strip())
            total_mem_gb = total_mem_bytes / (1024**3)
            logger.info(f"Total unified memory: {total_mem_gb:.1f}GB")
        except Exception:
            pass

        precision_str = "4-bit quantized" if cls._quantized else "float16"
        logger.info(f"Model loaded successfully on Apple Silicon ({precision_str})!")

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> str:
        """Generate response using MLX-LM.

        Args:
            messages: Conversation messages in chat format
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        try:
            from mlx_lm import generate
        except ImportError:
            raise ImportError(
                "MLX-LM is required. Install with: pip install mlx mlx-lm"
            )

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Generate with MLX-LM
        # MLX-LM's generate function handles all the optimization
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature if temperature > 0 else None,
            top_p=top_p,
            verbose=False,
        )

        return response


class MLXServerAgent(BaseDeepResearchAgent):
    """Deep research agent using MLX-LM server for inference.
    
    This agent connects to a running MLX-LM server for inference.
    Use this when you want to share a model across multiple processes
    or when running the model in a separate process for stability.
    
    Start the MLX-LM server with:
        mlx_lm.server --model PokeeAI/pokee_research_7b --port 8080
    """

    _client = None
    _server_url = None
    _client_lock = None

    def __init__(
        self,
        server_url: str = "http://localhost:8080/v1",
        model_name: str = "PokeeAI/pokee_research_7b",
        tool_config_path: str = "config/tool_config/pokee_tool_config.yaml",
        max_turns: int = 10,
        max_tool_response_length: int = 32768,
        timeout: float = 300.0,
    ):
        """Initialize the MLX server agent.

        Args:
            server_url: Base URL of the MLX-LM server
            model_name: Model name for reference
            tool_config_path: Path to tool configuration
            max_turns: Maximum conversation turns
            max_tool_response_length: Max length for tool responses
            timeout: HTTP request timeout in seconds
        """
        super().__init__(
            tool_config_path=tool_config_path,
            max_turns=max_turns,
            max_tool_response_length=max_tool_response_length,
        )

        self.server_url = server_url
        self.model_name = model_name
        self.timeout = timeout

        # Initialize lock
        if MLXServerAgent._client_lock is None:
            MLXServerAgent._client_lock = threading.Lock()

        # Create HTTP client
        with MLXServerAgent._client_lock:
            if MLXServerAgent._client is None:
                self._create_client(server_url, timeout)

        self.client = MLXServerAgent._client
        logger.debug(f"MLX server agent ready (server: {server_url})")

    @classmethod
    def _create_client(cls, server_url: str, timeout: float):
        """Create HTTP client for MLX-LM server."""
        import httpx

        logger.info(f"Creating HTTP client for MLX-LM server at {server_url}...")

        cls._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=timeout,
                connect=10.0,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )
        cls._server_url = server_url
        logger.info("HTTP client created successfully!")

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate response using MLX-LM server.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text response
        """
        import asyncio
        import httpx

        request_data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 4096,
            "stream": False,
        }

        try:
            response = await self.client.post(
                f"{self.server_url}/chat/completions",
                json=request_data,
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content if content else ""
            else:
                raise ValueError(f"Unexpected response format: {result}")

        except asyncio.CancelledError:
            logger.info("MLX generation cancelled")
            raise

        except httpx.HTTPStatusError as e:
            logger.error(f"MLX HTTP error {e.response.status_code}: {e.response.text}")
            raise

        except httpx.TimeoutException:
            logger.error(f"MLX request timeout after {self.timeout}s")
            raise

        except Exception as e:
            logger.error(f"Unexpected error calling MLX server: {e}")
            raise


def check_mlx_availability() -> dict:
    """Check if MLX is available and get system info.
    
    Returns:
        Dictionary with MLX availability info and system specs
    """
    info = {
        "mlx_available": False,
        "mlx_lm_available": False,
        "device": None,
        "total_memory_gb": None,
        "chip": None,
        "recommended_models": [],
    }

    try:
        import mlx.core as mx
        info["mlx_available"] = True
        info["device"] = str(mx.default_device())
    except ImportError:
        pass

    try:
        import mlx_lm
        info["mlx_lm_available"] = True
    except ImportError:
        pass

    # Get system memory
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
        )
        total_mem_bytes = int(result.stdout.strip())
        info["total_memory_gb"] = total_mem_bytes / (1024**3)
        
        # Recommend models based on memory
        # PokeeAI/pokee_research_7b is the primary model for this project
        mem_gb = info["total_memory_gb"]
        if mem_gb >= 32:
            # 32GB+ can run the full PokeeAI research model
            info["recommended_models"] = [
                "PokeeAI/pokee_research_7b",  # Primary - Deep Research Agent
            ]
        else:
            # Under 32GB - may need quantized version (if available)
            info["recommended_models"] = [
                "PokeeAI/pokee_research_7b",  # Try primary first
            ]
    except Exception:
        pass

    # Get chip info
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        info["chip"] = result.stdout.strip()
    except Exception:
        pass

    return info

