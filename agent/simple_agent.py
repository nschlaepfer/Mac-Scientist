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
Simple Deep Research Agent
Uses local model loading for inference.

Supports:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon M1/M2/M3)
- CPU (fallback)

For Apple Silicon with large unified memory (e.g., M3 Max 128GB):
- Uses float16 for MPS compatibility (bfloat16 not fully supported on MPS)
- Automatically detects and uses available unified memory
- Optimized attention for long context
"""

import platform
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.base_agent import BaseDeepResearchAgent
from logging_utils import setup_colored_logger

logger = setup_colored_logger(__name__)


def get_optimal_device() -> str:
    """Determine the optimal device for inference.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


def get_system_memory_gb() -> float:
    """Get total system memory in GB (useful for Apple Silicon unified memory)."""
    try:
        if platform.system() == "Darwin":  # macOS
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip()) / (1024**3)
        else:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return 16.0  # Default assumption


class SimpleDeepResearchAgent(BaseDeepResearchAgent):
    """Simple standalone deep research agent with local model.
    
    Optimized for:
    - NVIDIA GPUs (CUDA) with bfloat16
    - Apple Silicon (MPS) with float16 and unified memory
    - CPU fallback with float32
    
    For M3 Max with 128GB unified memory:
    - Can load 7B models in full precision
    - Can load up to ~60B models in float16
    - Automatic memory management via unified memory
    """

    # Class-level (singleton) model and tokenizer
    _model = None
    _tokenizer = None
    _model_path = None
    _device = None
    _model_lock = None  # For thread-safe initialization

    def __init__(
        self,
        model_path: str = "PokeeAI/pokee_research_7b",
        tool_config_path: str = "config/tool_config/pokee_tool_config.yaml",
        device: str = "auto",
        max_turns: int = 10,
        max_tool_response_length: int = 32768,
        max_memory_gb: float = None,  # Auto-detect for Apple Silicon
    ):
        """Initialize the agent.

        Args:
            model_path: Path to model or HuggingFace model ID
            tool_config_path: Path to tool configuration
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            max_turns: Maximum conversation turns
            max_tool_response_length: Max length for tool responses
            max_memory_gb: Maximum memory to use (auto-detected on Apple Silicon)
        """
        # Initialize base class
        super().__init__(
            tool_config_path=tool_config_path,
            max_turns=max_turns,
            max_tool_response_length=max_tool_response_length,
        )

        # Auto-detect device if needed
        if device == "auto":
            device = get_optimal_device()
            logger.info(f"Auto-detected device: {device}")

        # Auto-detect memory on Apple Silicon
        if max_memory_gb is None and device == "mps":
            max_memory_gb = get_system_memory_gb() * 0.8  # Use 80% of system memory
            logger.info(f"Apple Silicon detected - allocating up to {max_memory_gb:.1f}GB")

        self.max_memory_gb = max_memory_gb

        # Initialize lock on first use
        if SimpleDeepResearchAgent._model_lock is None:
            import threading

            SimpleDeepResearchAgent._model_lock = threading.Lock()

        # Load model only once (singleton pattern) with thread safety
        with SimpleDeepResearchAgent._model_lock:
            if (
                SimpleDeepResearchAgent._model is None
                or SimpleDeepResearchAgent._model_path != model_path
            ):
                self._load_model(model_path, device)
            elif SimpleDeepResearchAgent._device != device:
                logger.warning(
                    f"Model already loaded on {SimpleDeepResearchAgent._device}, ignoring device={device}"
                )

        # Use the shared model and tokenizer
        self.model = SimpleDeepResearchAgent._model
        self.tokenizer = SimpleDeepResearchAgent._tokenizer

    @classmethod
    def _load_model(cls, model_path: str, device: str):
        """Load model and tokenizer (class method for singleton).
        
        Automatically selects optimal dtype and settings based on device:
        - CUDA: bfloat16 with flash attention, uses device_map="auto"
        - MPS: float16, loads to CPU then moves to MPS (device_map not supported)
        - CPU: float32
        """
        logger.info(f"Loading model from {model_path}...")
        logger.info(f"Target device: {device}")
        
        # Log system info for Apple Silicon
        if device == "mps":
            total_mem = get_system_memory_gb()
            logger.info(f"Unified memory available: {total_mem:.1f}GB")
            logger.info("Using float16 for MPS compatibility")

        cls._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        
        # Select optimal dtype based on device
        if device == "cuda":
            dtype = torch.bfloat16
        elif device == "mps":
            # MPS works best with float16 (bfloat16 has limited support)
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Build model kwargs based on device
        # Note: device_map only accepts "auto", "balanced", "sequential", or dict
        # It does NOT accept "cuda", "mps", "cpu" directly
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        
        # Device-specific loading strategy
        if device == "cuda":
            # CUDA: use device_map="auto" for automatic GPU placement
            model_kwargs["device_map"] = "auto"
            # Add attention implementation
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                model_kwargs["attn_implementation"] = "sdpa"
                logger.info("Using SDPA attention (flash-attn not installed)")
        elif device == "mps":
            # MPS: device_map doesn't support MPS, load to CPU first then move
            # Don't set device_map - we'll move to MPS after loading
            model_kwargs["attn_implementation"] = "sdpa"
            logger.info("Using SDPA attention for MPS")
        else:
            # CPU: no device_map needed
            pass

        logger.info(f"Loading with dtype={dtype}")

        cls._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
        
        # For MPS: explicitly move model to MPS after loading
        if device == "mps":
            logger.info("Moving model to MPS (Apple Silicon GPU)...")
            cls._model = cls._model.to("mps")
        elif device == "cpu":
            cls._model = cls._model.to("cpu")
        # For CUDA with device_map="auto", model is already on GPU
        
        cls._model.eval()
        cls._model_path = model_path
        cls._device = device
        
        # Log memory usage
        if device == "mps":
            param_count = sum(p.numel() for p in cls._model.parameters())
            bytes_per_param = 2 if dtype == torch.float16 else 4
            mem_estimate = param_count * bytes_per_param / (1024**3)
            logger.info(f"Estimated model memory: {mem_estimate:.2f}GB")
        elif device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"GPU memory allocated: {allocated:.2f}GB")

        logger.info(f"Model loaded successfully on {device}!")

    async def generate(
        self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        """Generate response from messages using local model.

        Optimized for:
        - CUDA: Uses KV cache and flash attention
        - MPS: Uses autocast for optimal performance on Apple Silicon
        - CPU: Standard generation

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        # Apply chat template (reuse tool_schemas from base class)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize with padding for efficiency
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,  # No padding needed for single sequence
            truncation=True,  # Prevent OOM from overly long prompts
            max_length=32768,  # Match model's max context
        ).to(self.model.device)

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": 2048,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,  # Only sample if temp > 0
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,  # Enable KV cache for faster generation
            "num_beams": 1,  # Greedy decoding (faster than beam search)
        }

        # Generate with device-specific optimizations
        device = str(self.model.device)
        
        with torch.no_grad():
            if "mps" in device:
                # MPS-specific optimization: use autocast for float16
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    outputs = self.model.generate(**inputs, **gen_kwargs)
            elif "cuda" in device:
                # CUDA: use autocast for bfloat16
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model.generate(**inputs, **gen_kwargs)
            else:
                # CPU fallback
                outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the generated part (more efficient)
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response
