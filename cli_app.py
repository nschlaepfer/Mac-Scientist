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
Deep Research Agent - User Interface
This script provides a simple CLI interface to interact with the trained deep research agent.

Supports multiple serving modes:
- local: HuggingFace Transformers with auto device detection (CUDA/MPS/CPU)
- mlx: Apple Silicon native inference via MLX-LM (fastest on M1/M2/M3)
- mlx-server: Connect to MLX-LM server for shared model access
- vllm: Connect to vLLM server (NVIDIA GPUs)
"""

import argparse
import asyncio
import platform
import time

import torch

from logging_utils import setup_colored_logger

logger = setup_colored_logger("cli_app")


def get_default_device() -> str:
    """Get the optimal default device based on hardware."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return (
        platform.system() == "Darwin" 
        and platform.machine() == "arm64"
    )


async def interactive_mode_async(
    agent,
    temperature: float,
    top_p: float,
    verbose: bool,
):
    """Async interactive mode loop."""
    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            print("\nAgent: Researching...\n")
            start_time = time.time()
            answer = await agent.run(
                question_raw=question,
                temperature=temperature,
                top_p=top_p,
                verbose=verbose,
            )

            print(f"\nAgent: {answer}\n")
            print("Time taken: {:.2f} seconds".format(time.time() - start_time))
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"\nError: {e}")
            if verbose:
                import traceback

                traceback.print_exc()


def create_agent(
    serving_mode: str,
    model_path: str,
    tool_config_path: str,
    device: str,
    max_turns: int,
    vllm_url: str = None,
    mlx_server_url: str = None,
):
    """Create an agent based on serving mode.
    
    Args:
        serving_mode: One of 'local', 'mlx', 'mlx-server', 'vllm'
        model_path: Model path or HuggingFace model ID
        tool_config_path: Path to tool configuration
        device: Device for local mode ('auto', 'cuda', 'mps', 'cpu')
        max_turns: Maximum conversation turns
        vllm_url: URL for vLLM server
        mlx_server_url: URL for MLX-LM server
        
    Returns:
        Configured agent instance
    """
    if serving_mode == "vllm":
        if not vllm_url:
            raise ValueError("VLLM URL must be provided when using VLLM agent")
        from agent.vllm_agent import VLLMDeepResearchAgent

        logger.info(f"Using VLLM agent at {vllm_url}")
        return VLLMDeepResearchAgent(
            vllm_url=vllm_url,
            model_name=model_path,
            tool_config_path=tool_config_path,
            max_turns=max_turns,
        )
    
    elif serving_mode == "mlx":
        # MLX native inference (Apple Silicon only)
        if not is_apple_silicon():
            raise ValueError("MLX is only available on Apple Silicon Macs")
        from agent.mlx_agent import MLXDeepResearchAgent

        logger.info("Using MLX agent (Apple Silicon native)")
        return MLXDeepResearchAgent(
            model_path=model_path,
            tool_config_path=tool_config_path,
            max_turns=max_turns,
        )
    
    elif serving_mode == "mlx-server":
        # MLX-LM server
        if not mlx_server_url:
            mlx_server_url = "http://localhost:8080/v1"
        from agent.mlx_agent import MLXServerAgent

        logger.info(f"Using MLX server agent at {mlx_server_url}")
        return MLXServerAgent(
            server_url=mlx_server_url,
            model_name=model_path,
            tool_config_path=tool_config_path,
            max_turns=max_turns,
        )
    
    else:  # local mode
        from agent.simple_agent import SimpleDeepResearchAgent

        logger.info(f"Using local model agent (device: {device})")
        return SimpleDeepResearchAgent(
            model_path=model_path,
            tool_config_path=tool_config_path,
            device=device,
            max_turns=max_turns,
        )


def interactive_mode(
    serving_mode: str,
    model_path: str,
    tool_config_path: str,
    device: str,
    max_turns: int,
    temperature: float,
    top_p: float,
    verbose: bool,
    vllm_url: str = None,
    mlx_server_url: str = None,
):
    """Run interactive mode."""
    agent = create_agent(
        serving_mode=serving_mode,
        model_path=model_path,
        tool_config_path=tool_config_path,
        device=device,
        max_turns=max_turns,
        vllm_url=vllm_url,
        mlx_server_url=mlx_server_url,
    )

    # Display mode info
    device_info = device if serving_mode == "local" else serving_mode.upper()
    print("\n" + "=" * 80)
    print("Deep Research Agent - Interactive Mode")
    print(f"Serving Mode: {serving_mode.upper()}")
    if serving_mode == "local":
        print(f"Device: {device}")
    print(f"Model: {model_path}")
    if is_apple_silicon():
        print("Platform: Apple Silicon (M-series)")
    print("=" * 80)
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 80 + "\n")

    # Run entire interactive session in single event loop
    asyncio.run(interactive_mode_async(agent, temperature, top_p, verbose))


def single_query_mode(
    question: str,
    serving_mode: str,
    model_path: str,
    tool_config_path: str,
    device: str,
    max_turns: int,
    temperature: float,
    top_p: float,
    verbose: bool,
    vllm_url: str = None,
    mlx_server_url: str = None,
) -> str:
    """Run single query."""
    agent = create_agent(
        serving_mode=serving_mode,
        model_path=model_path,
        tool_config_path=tool_config_path,
        device=device,
        max_turns=max_turns,
        vllm_url=vllm_url,
        mlx_server_url=mlx_server_url,
    )

    start_time = time.time()
    try:
        answer = asyncio.run(
            agent.run(
                question_raw=question,
                temperature=temperature,
                top_p=top_p,
                verbose=verbose,
            )
        )
        print("Time taken: {:.2f} seconds".format(time.time() - start_time))
        return answer
    except Exception as e:
        logger.error(f"\nError: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return "Error occurred while processing the query."


def main():
    # Determine best default mode based on platform
    default_mode = "mlx" if is_apple_silicon() else "local"
    default_device = get_default_device()
    
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - User Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with local model (auto device detection)
  python cli_app.py --serving-mode local
  
  # Apple Silicon: Use MLX for fastest inference (recommended)
  python cli_app.py --serving-mode mlx
  
  # Apple Silicon: Use MLX-LM server
  python cli_app.py --serving-mode mlx-server --mlx-server-url http://localhost:8080/v1
  
  # Force MPS device with local mode
  python cli_app.py --serving-mode local --device mps
  
  # Interactive mode with VLLM (NVIDIA GPUs)
  python cli_app.py --serving-mode vllm --vllm-url http://localhost:9999/v1
  
  # Single query with MLX
  python cli_app.py --serving-mode mlx --question "What is the capital of France?"

Platform-specific recommendations:
  - Apple Silicon (M1/M2/M3): Use --serving-mode mlx for best performance
  - NVIDIA GPU: Use --serving-mode local or vllm
  - CPU only: Use --serving-mode local --device cpu
        """,
    )
    parser.add_argument(
        "--serving-mode",
        type=str,
        choices=["local", "mlx", "mlx-server", "vllm"],
        default=default_mode,
        help=f"Serving mode (default: {default_mode}). "
             "'local' = HuggingFace Transformers, "
             "'mlx' = Apple Silicon native (M1/M2/M3), "
             "'mlx-server' = MLX-LM server, "
             "'vllm' = vLLM server",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:9999/v1",
        help="URL of the vLLM server (for --serving-mode vllm)",
    )
    parser.add_argument(
        "--mlx-server-url",
        type=str,
        default="http://localhost:8080/v1",
        help="URL of the MLX-LM server (for --serving-mode mlx-server)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="PokeeAI/pokee_research_7b",
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--tool-config",
        type=str,
        default="config/tool_config/pokee_tool_config.yaml",
        help="Path to tool configuration file",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer (non-interactive mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["auto", "cuda", "mps", "cpu"],
        help=f"Device for local mode (default: {default_device}). "
             "'auto' = automatic detection, "
             "'mps' = Apple Silicon GPU, "
             "'cuda' = NVIDIA GPU, "
             "'cpu' = CPU only",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10, help="Maximum number of agent turns"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Nucleus sampling parameter"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate mode-specific requirements
    if args.serving_mode == "vllm" and not args.vllm_url:
        parser.error("--vllm-url is required when using --serving-mode vllm")
    
    if args.serving_mode == "mlx" and not is_apple_silicon():
        parser.error("--serving-mode mlx requires Apple Silicon (M1/M2/M3)")

    # Print platform info
    if is_apple_silicon():
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            total_mem_gb = int(result.stdout.strip()) / (1024**3)
            logger.info(f"Apple Silicon detected - {total_mem_gb:.0f}GB unified memory")
        except Exception:
            logger.info("Apple Silicon detected")

    if args.question:
        # Single query mode
        answer = single_query_mode(
            question=args.question,
            serving_mode=args.serving_mode,
            model_path=args.model_path,
            tool_config_path=args.tool_config,
            device=args.device,
            max_turns=args.max_turns,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
            vllm_url=args.vllm_url,
            mlx_server_url=args.mlx_server_url,
        )
        print(f"\nQuestion: {args.question}")
        print(f"\nAnswer: {answer}\n")
    else:
        # Interactive mode
        interactive_mode(
            serving_mode=args.serving_mode,
            model_path=args.model_path,
            tool_config_path=args.tool_config,
            device=args.device,
            max_turns=args.max_turns,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
            vllm_url=args.vllm_url,
            mlx_server_url=args.mlx_server_url,
        )


if __name__ == "__main__":
    main()
