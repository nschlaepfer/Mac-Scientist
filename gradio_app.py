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
Gradio Web Interface for Pokee Deep Research Agent

This module provides a web-based interface for interacting with the deep research agent.
It supports both local model loading and VLLM server-based inference with concurrent
user sessions and real-time progress streaming.

Features:
    - Secure API key configuration with environment variable storage
    - Manual tool server lifecycle management
    - Multiple concurrent user sessions with independent state management
    - Real-time research progress updates with tool call visibility
    - Graceful cancellation and cleanup of research tasks
    - Support for both local and VLLM-based agent backends
    - Customizable research parameters (temperature, top_p, max_turns)

Usage:
    # Start with VLLM backend (NVIDIA GPUs)
    python gradio_app.py --serving-mode vllm --vllm-url http://localhost:9999/v1

    # Start with local backend (auto device detection: CUDA/MPS/CPU)
    python gradio_app.py --serving-mode local --model-path path/to/model

    # Apple Silicon: Use MLX for native performance (recommended for M1/M2/M3)
    python gradio_app.py --serving-mode mlx
    
    # Apple Silicon: Use MLX-LM server
    python gradio_app.py --serving-mode mlx-server --mlx-server-url http://localhost:8080/v1

    # Enable public sharing
    python gradio_app.py --share --port 7777
"""

import argparse
import asyncio
import atexit
import os
import platform
import socket
import subprocess
import sys
import threading
import time

import gradio as gr
import requests
import torch

from logging_utils import setup_colored_logger

logger = setup_colored_logger(__name__)


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


def get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip()) / (1024**3)
    except Exception:
        pass
    return 16.0

# Global configuration (read-only after initialization, safe for concurrent access)
serving_mode = None
agent_config = {}

# Track running research tasks per session
running_tasks: dict[str, asyncio.Task] = {}

# Global tool server process and configuration
tool_server_proc = None
tool_server_port = 8888


def is_port_available(port: int) -> bool:
    """Check if a TCP port is available for binding.

    Args:
        port: Port number to check (1024-65535)

    Returns:
        bool: True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except OSError:
        return False


def start_tool_server(port: int, timeout: int = 30) -> subprocess.Popen:
    """Start the tool server as a background subprocess.

    Launches the tool server process and waits for it to become healthy
    by polling the health endpoint. Logs stderr output in a background thread.

    Args:
        port: Port number for the tool server (1024-65535)
        timeout: Maximum seconds to wait for server readiness

    Returns:
        subprocess.Popen: The running server process

    Raises:
        RuntimeError: If server fails to start, crashes, or doesn't become ready within timeout
    """
    logger.info(f"Starting tool server on port {port}...")

    # Start the server process with proper output handling
    proc = subprocess.Popen(
        [sys.executable, "start_tool_server.py", "--port", str(port), "--enable-cache"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=os.environ.copy(),  # Pass current environment with API keys
    )

    # Start background thread to log stderr
    def log_stderr():
        for line in proc.stderr:
            logger.debug(f"[Tool Server] {line.rstrip()}")

    stderr_thread = threading.Thread(target=log_stderr, daemon=True)
    stderr_thread.start()

    # Check if process started successfully
    time.sleep(0.5)
    if proc.poll() is not None:
        remaining_stderr = proc.stderr.read()
        logger.error(f"Tool server failed to start: {remaining_stderr}")
        raise RuntimeError(
            f"Tool server process terminated immediately: {remaining_stderr}"
        )

    # Wait for server to become ready by polling health endpoint
    server_url = f"http://localhost:{port}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(server_url, timeout=1.0)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    logger.info(f"✅ Tool server is ready at {server_url}")
                    return proc
        except requests.exceptions.RequestException:
            pass

        # Check if process crashed
        if proc.poll() is not None:
            logger.error("Tool server crashed during startup")
            raise RuntimeError("Tool server crashed during startup")

        time.sleep(0.5)

    # Timeout reached - terminate process
    logger.error("Tool server failed to become ready within timeout")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    raise RuntimeError(f"Tool server failed to become ready within {timeout}s")


def cleanup_tool_server(proc: subprocess.Popen):
    """Gracefully shutdown the tool server process.

    Attempts graceful termination first, then forces kill if necessary.

    Args:
        proc: The server process to shutdown (can be None)
    """
    if proc is None:
        return

    if proc.poll() is None:  # Process still running
        logger.info("Shutting down tool server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
            logger.info("Tool server shut down gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Tool server did not terminate, forcing kill...")
            proc.kill()
            proc.wait()
            logger.info("Tool server killed")


def get_tool_server_status() -> dict:
    """Check if tool server is running and healthy.

    Checks both process status and health endpoint response.

    Returns:
        dict: Status information with keys:
            - 'running' (bool): True if server is healthy
            - 'message' (str): Human-readable status message
    """
    global tool_server_proc

    if tool_server_proc is None:
        return {"running": False, "message": "Tool server not started"}

    if tool_server_proc.poll() is not None:
        return {"running": False, "message": "Tool server has stopped"}

    try:
        response = requests.get(f"http://localhost:{tool_server_port}", timeout=2.0)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy":
                return {"running": True, "message": "Tool server is healthy"}
    except requests.exceptions.RequestException:
        pass

    return {"running": False, "message": "Tool server not responding"}


def save_api_keys(serper_key: str, jina_key: str, gemini_key: str) -> str:
    """Save API keys to environment variables.

    Validates that all keys are provided and stores them in the current process
    environment for use by the tool server.

    Args:
        serper_key: Serper API key for web search functionality
        jina_key: Jina AI API key for web content reading
        gemini_key: Gemini API key for content summarization

    Returns:
        str: Status message for UI display (includes ✅/❌ emoji)
    """
    try:
        # Validate that all keys are provided
        if not all([serper_key, jina_key, gemini_key]):
            return "❌ Please provide all API keys"

        # Set environment variables
        if serper_key:
            os.environ["SERPER_API_KEY"] = serper_key.strip()
            logger.info("✅ Serper API key configured")

        if jina_key:
            os.environ["JINA_API_KEY"] = jina_key.strip()
            logger.info("✅ Jina AI API key configured")

        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key.strip()
            logger.info("✅ Gemini API key configured")

        return "✅ API keys saved successfully! You can now start the tool server."

    except Exception as e:
        logger.error(f"Failed to save API keys: {e}")
        return f"❌ Failed to save API keys: {str(e)}"


def start_tool_server_ui(port: int) -> tuple[str, str]:
    """Start the tool server from UI button click.

    Validates port availability and API keys, then starts the server process.

    Args:
        port: Port number to use for the tool server (1024-65535)

    Returns:
        tuple[str, str]: (status_message, server_status_display) for UI updates
    """
    global tool_server_proc, tool_server_port

    # Check if already running
    if tool_server_proc is not None and tool_server_proc.poll() is None:
        return "⚠️ Tool server is already running", "🟢 Running"

    # Validate port range
    if not (1024 <= port <= 65535):
        return "❌ Invalid port number. Must be between 1024 and 65535", "🔴 Stopped"

    # Check port availability
    if not is_port_available(port):
        return (
            f"❌ Port {port} is already in use. Please choose another port.",
            "🔴 Stopped",
        )

    # Update global port
    tool_server_port = port

    # Check if API keys are configured
    required_keys = ["SERPER_API_KEY", "JINA_API_KEY", "GEMINI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]

    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        return (
            f"⚠️ Warning: Missing API keys: {', '.join(missing_keys)}\n"
            "Tool server will start but some features may not work.",
            "🟡 Starting...",
        )

    try:
        tool_server_proc = start_tool_server(tool_server_port)
        return (
            f"✅ Tool server started successfully on port {tool_server_port}!",
            "🟢 Running",
        )
    except Exception as e:
        logger.error(f"Failed to start tool server: {e}")
        return f"❌ Failed to start tool server: {str(e)}", "🔴 Stopped"


def stop_tool_server_ui() -> tuple[str, str]:
    """Stop the tool server from UI button click.

    Returns:
        tuple[str, str]: (status_message, server_status_display) for UI updates
    """
    global tool_server_proc

    if tool_server_proc is None or tool_server_proc.poll() is not None:
        return "⚠️ Tool server is not running", "🔴 Stopped"

    try:
        cleanup_tool_server(tool_server_proc)
        tool_server_proc = None
        return "✅ Tool server stopped successfully", "🔴 Stopped"
    except Exception as e:
        logger.error(f"Failed to stop tool server: {e}")
        return f"❌ Failed to stop tool server: {str(e)}", "🔴 Stopped"


def refresh_server_status() -> str:
    """Refresh and return current tool server status.

    Polls the server and returns formatted status string for UI display.

    Returns:
        str: Server status display with emoji indicator (🟢/🔴)
    """
    status = get_tool_server_status()
    if status["running"]:
        return f"🟢 Running - {status['message']}"
    else:
        return f"🔴 Stopped - {status['message']}"


def create_agent():
    """Create a new agent instance for a user session.

    Factory function that creates the appropriate agent type based on
    the global serving_mode configuration.

    Supports:
    - vllm: vLLM server (NVIDIA GPUs)
    - mlx: MLX native inference (Apple Silicon)
    - mlx-server: MLX-LM server
    - local: HuggingFace Transformers (auto device detection)

    Returns:
        Configured agent instance
    """
    if serving_mode == "vllm":
        from agent.vllm_agent import VLLMDeepResearchAgent

        return VLLMDeepResearchAgent(
            vllm_url=agent_config["vllm_url"],
            model_name=agent_config["model_path"],
            tool_config_path=agent_config["tool_config_path"],
            max_turns=agent_config["max_turns"],
        )
    
    elif serving_mode == "mlx":
        from agent.mlx_agent import MLXDeepResearchAgent

        return MLXDeepResearchAgent(
            model_path=agent_config["model_path"],
            tool_config_path=agent_config["tool_config_path"],
            max_turns=agent_config["max_turns"],
        )
    
    elif serving_mode == "mlx-server":
        from agent.mlx_agent import MLXServerAgent

        return MLXServerAgent(
            server_url=agent_config.get("mlx_server_url", "http://localhost:8080/v1"),
            model_name=agent_config["model_path"],
            tool_config_path=agent_config["tool_config_path"],
            max_turns=agent_config["max_turns"],
        )
    
    else:  # local
        from agent.simple_agent import SimpleDeepResearchAgent

        return SimpleDeepResearchAgent(
            model_path=agent_config["model_path"],
            tool_config_path=agent_config["tool_config_path"],
            device=agent_config["device"],
            max_turns=agent_config["max_turns"],
        )


def get_session_id(request: gr.Request) -> str:
    """Get unique session ID for the current user.

    Extracts session hash from Gradio request for session-specific state management.

    Args:
        request: Gradio request object containing session information

    Returns:
        str: Unique session identifier (or "default" if unavailable)
    """
    if request and hasattr(request, "session_hash"):
        return request.session_hash
    return "default"


def format_timeline_item(item_type: str, title: str, details: str = "", items: list = None) -> str:
    """Format a single timeline item with proper styling."""
    items_html = ""
    if items:
        items_html = "<ul>" + "".join(f"<li>{item}</li>" for item in items[:5])
        if len(items) > 5:
            items_html += f"<li>... and {len(items) - 5} more</li>"
        items_html += "</ul>"
    
    return f"""
<div class="timeline-item timeline-{item_type}">
    <div class="timeline-dot"></div>
    <div class="timeline-content">
        <div class="timeline-title">{title}</div>
        <div class="timeline-details">
            {details}
            {items_html}
        </div>
    </div>
</div>
"""


async def research_stream(
    question: str,
    temperature: float,
    top_p: float,
    max_turns: int,
    request: gr.Request,
):
    """Execute research with real-time step-by-step updates.

    This async generator streams research progress updates to the UI, showing:
    - Tool calls (web searches, web reads)
    - Agent thinking process
    - Verification steps
    - Final answer or errors

    Manages session-specific task tracking for cancellation support.

    Args:
        question: Research question from user
        temperature: Sampling temperature for generation (0.0-1.0, lower = more focused)
        top_p: Nucleus sampling parameter (0.0-1.0)
        max_turns: Maximum number of agent iterations allowed
        request: Gradio request object for session tracking

    Yields:
        tuple[str, str]: (thinking_log, answer) for progressive UI updates
    """
    session_id = get_session_id(request)

    # Check if tool server is running
    status = get_tool_server_status()
    if not status["running"]:
        error_html = """
<div class="timeline-item timeline-error">
    <div class="timeline-dot"></div>
    <div class="timeline-content">
        <div class="timeline-title">⚠️ Tool Server Not Running</div>
        <div class="timeline-details">
            Please start the tool server in the Setup tab before conducting research.
        </div>
    </div>
</div>
        """
        yield error_html, "❌ Tool server is not running. Please start it in the Setup tab."
        return

    logger.info(f"Starting research for session {session_id[:8]}...")

    agent = None
    timeline_items = []
    
    # Add initial "started" item
    timeline_items.append(format_timeline_item(
        "think",
        "🚀 Research Started",
        f"Analyzing your question and preparing search strategy..."
    ))

    try:
        agent = create_agent()
        current_task = asyncio.current_task()
        running_tasks[session_id] = current_task

        async for update in agent.run_stream(
            question_raw=question,
            temperature=temperature,
            top_p=top_p,
            max_turns=int(max_turns),
        ):
            update_type = update["type"]

            if update_type == "answer_found":
                think = update.get("think", "")
                think_preview = (think[:200] + "...") if len(think) > 200 else think
                timeline_items.append(format_timeline_item(
                    "verify",
                    "🔎 Verifying Answer",
                    think_preview if think_preview else "Cross-referencing findings for accuracy..."
                ))
                yield "".join(timeline_items), ""

            elif update_type == "tool_call":
                think = update.get("think", "")
                tool_name = update["tool_name"]
                
                if tool_name == "web_search":
                    queries = update["queries"]
                    timeline_items.append(format_timeline_item(
                        "search",
                        f"🔍 Web Search ({len(queries)} queries)",
                        "",
                        queries
                    ))

                elif tool_name == "web_read":
                    urls = update["urls"]
                    # Truncate URLs for display
                    display_urls = [u[:60] + "..." if len(u) > 60 else u for u in urls]
                    timeline_items.append(format_timeline_item(
                        "read",
                        f"📖 Reading Sources ({len(urls)} pages)",
                        "",
                        display_urls
                    ))

                yield "".join(timeline_items), ""

            elif update_type == "done":
                timeline_items.append(format_timeline_item(
                    "done",
                    "✅ Research Complete",
                    "Successfully synthesized findings into a comprehensive answer."
                ))
                yield "".join(timeline_items), update["answer"]
                break

            elif update_type == "error":
                logger.error(
                    f"Research error for session {session_id[:8]}: {update['message']}",
                    exc_info=True,
                )
                timeline_items.append(format_timeline_item(
                    "error",
                    "❌ Error Occurred",
                    f"An error occurred during research. Please try again."
                ))
                yield "".join(timeline_items), "❌ Error occurred during research."
                break

    except asyncio.CancelledError:
        logger.info(f"Research cancelled for session {session_id[:8]}")
        if agent:
            await agent.cleanup_tool_instances()
        timeline_items.append(format_timeline_item(
            "error",
            "🛑 Cancelled",
            "Research was cancelled by user."
        ))
        yield "".join(timeline_items), "❌ Research cancelled by user."
        raise

    except Exception as e:
        logger.error(f"Research error for session {session_id[:8]}: {e}", exc_info=True)
        timeline_items.append(format_timeline_item(
            "error",
            "❌ Unexpected Error",
            f"Something went wrong. Please try again."
        ))
        yield "".join(timeline_items), "❌ Error occurred during research."

    finally:
        if session_id in running_tasks:
            del running_tasks[session_id]


def cancel_research(request: gr.Request):
    """Cancel the currently running research task for this session.

    Looks up the session's task and requests cancellation via asyncio.

    Args:
        request: Gradio request object for session tracking

    Returns:
        tuple[gr.update, gr.update]: UI update objects for progress and answer outputs
    """
    session_id = get_session_id(request)

    if session_id not in running_tasks:
        logger.debug(f"No running research for session {session_id[:8]}")
        return gr.update(), gr.update()

    task = running_tasks[session_id]
    if task.done():
        logger.debug(f"Task already done for session {session_id[:8]}")
        return gr.update(), gr.update()

    try:
        task.cancel()
        logger.info(f"Cancellation requested for session {session_id[:8]}")
        return "\n\n🛑 **Cancellation requested...**", gr.update()
    except Exception as e:
        logger.error(f"Failed to cancel session {session_id[:8]}: {e}")
        return f"\n\n❌ **Failed to cancel:** {e}", gr.update()


def create_demo():
    """Create and configure the Gradio interface.

    Builds a premium multi-tab interface with:
    - Setup tab: API configuration and tool server management
    - Research tab: Question input and real-time progress display
    - About tab: Documentation and usage tips

    Returns:
        gr.Blocks: Configured Gradio interface ready to launch
    """
    # Build display strings
    mode_display_map = {
        "vllm": "vLLM Server",
        "mlx": "MLX (Apple Silicon)",
        "mlx-server": "MLX-LM Server",
        "local": f"Local ({agent_config.get('device', 'auto').upper()})",
    }
    serving_mode_display = mode_display_map.get(serving_mode, serving_mode.upper())
    
    # Platform badges
    platform_badge = ""
    mem_badge = ""
    if is_apple_silicon():
        mem_gb = get_system_memory_gb()
        platform_badge = "Apple Silicon"
        mem_badge = f"{mem_gb:.0f}GB"

    # Premium CSS with dark scientific theme
    custom_css = """
    /* ===== IMPORTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* ===== ROOT VARIABLES ===== */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a24;
        --bg-card: rgba(26, 26, 36, 0.8);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(99, 102, 241, 0.4);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-glow: rgba(99, 102, 241, 0.2);
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        --gradient-dark: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
        --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.15);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
    }

    /* ===== BASE STYLES ===== */
    .gradio-container {
        background: var(--gradient-dark) !important;
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
        max-width: 1400px !important;
    }
    
    .dark {
        --background-fill-primary: var(--bg-primary) !important;
        --background-fill-secondary: var(--bg-secondary) !important;
    }

    /* ===== HEADER HERO ===== */
    .hero-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xl);
        padding: 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: var(--gradient-primary);
        opacity: 0.6;
    }
    
    .hero-header h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem 0 !important;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin: 0;
    }
    
    .hero-badges {
        display: flex;
        gap: 0.75rem;
        margin-top: 1.25rem;
        flex-wrap: wrap;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.85rem;
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
        backdrop-filter: blur(10px);
    }
    
    .badge-accent {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
    }
    
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
    }

    /* ===== TABS ===== */
    .tabs {
        border: none !important;
        background: transparent !important;
    }
    
    .tabitem {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    button.tab-nav {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.85rem 1.5rem !important;
        margin-right: 0.5rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: var(--text-secondary) !important;
        transition: all 0.2s ease !important;
    }
    
    button.tab-nav:hover {
        background: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
        color: var(--text-primary) !important;
    }
    
    button.tab-nav.selected {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: var(--accent-primary) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2) !important;
    }

    /* ===== CARDS ===== */
    .glass-card {
        background: var(--bg-card) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
        box-shadow: var(--shadow-glow) !important;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .card-icon {
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--gradient-primary);
        border-radius: var(--radius-md);
        font-size: 1.25rem;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .card-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* ===== INPUTS ===== */
    .input-group label {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        color: var(--text-secondary) !important;
        margin-bottom: 0.5rem !important;
    }
    
    input[type="text"], input[type="password"], textarea {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.85rem 1rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        color: var(--text-primary) !important;
        transition: all 0.2s ease !important;
    }
    
    input[type="text"]:focus, input[type="password"]:focus, textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        outline: none !important;
    }
    
    input::placeholder, textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* ===== BUTTONS ===== */
    .primary-btn, button.primary {
        background: var(--gradient-primary) !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 0.85rem 1.75rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: white !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    
    .primary-btn:hover, button.primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.4) !important;
    }
    
    .secondary-btn, button.secondary {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.85rem 1.75rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        transition: all 0.2s ease !important;
    }
    
    .secondary-btn:hover, button.secondary:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: var(--text-muted) !important;
        color: var(--text-primary) !important;
    }
    
    .danger-btn, button.stop {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #fca5a5 !important;
    }
    
    .danger-btn:hover, button.stop:hover {
        background: rgba(239, 68, 68, 0.25) !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
    }

    /* ===== STATUS INDICATORS ===== */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1rem;
        border-radius: 100px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .status-running {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
    }
    
    .status-stopped {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #fca5a5;
    }
    
    .status-dot {
            width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-dot-green {
        background: var(--success);
        box-shadow: 0 0 10px var(--success);
    }
    
    .status-dot-red {
        background: var(--error);
        box-shadow: 0 0 10px var(--error);
        animation: none;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ===== PROGRESS TIMELINE ===== */
    .research-progress {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        min-height: 450px;
        max-height: 550px;
        overflow-y: auto;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .research-progress::-webkit-scrollbar {
        width: 6px;
    }
    
    .research-progress::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .research-progress::-webkit-scrollbar-thumb {
        background: var(--border-subtle);
        border-radius: 3px;
    }
    
    .research-progress::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    .progress-empty {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--text-muted);
            text-align: center;
        }
    
    .progress-empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .timeline-item {
        position: relative;
        padding-left: 2rem;
        padding-bottom: 1.5rem;
        border-left: 2px solid var(--border-subtle);
        margin-left: 0.5rem;
    }
    
    .timeline-item:last-child {
        border-left-color: transparent;
        padding-bottom: 0;
    }
    
    .timeline-dot {
        position: absolute;
        left: -7px;
        top: 0;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--accent-primary);
        border: 2px solid var(--bg-secondary);
    }
    
    .timeline-search .timeline-dot { background: #3b82f6; }
    .timeline-read .timeline-dot { background: #8b5cf6; }
    .timeline-think .timeline-dot { background: #f59e0b; }
    .timeline-verify .timeline-dot { background: #10b981; }
    .timeline-done .timeline-dot { background: #10b981; box-shadow: 0 0 10px rgba(16, 185, 129, 0.5); }
    .timeline-error .timeline-dot { background: #ef4444; }
    
    .timeline-content {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 1rem;
    }
    
    .timeline-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .timeline-details {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .timeline-details ul {
        margin: 0.5rem 0 0 0;
        padding-left: 1.25rem;
    }
    
    .timeline-details li {
        margin-bottom: 0.25rem;
        word-break: break-all;
    }

    /* ===== ANSWER BOX ===== */
    .answer-container {
        margin-top: 1rem;
    }
    
    .answer-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
    }
    
    .answer-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        color: #6ee7b7;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ===== ACCORDION ===== */
    .accordion {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-glass) !important;
        margin-top: 1rem !important;
    }
    
    .accordion > button {
        background: transparent !important;
        border: none !important;
        padding: 1rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
    }
    
    .accordion > button:hover {
        color: var(--text-primary) !important;
    }

    /* ===== SLIDER ===== */
    input[type="range"] {
        accent-color: var(--accent-primary) !important;
    }
    
    .slider-label {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ===== EXAMPLES ===== */
    .examples-table {
        margin-top: 1.5rem;
    }
    
    .examples-table button {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.75rem 1rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-secondary) !important;
        transition: all 0.2s ease !important;
        text-align: left !important;
    }
    
    .examples-table button:hover {
        background: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
        color: var(--text-primary) !important;
    }

    /* ===== STEP INDICATOR ===== */
    .setup-steps {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .step-item {
        flex: 1;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        transition: all 0.2s ease;
    }
    
    .step-item.active {
        border-color: var(--accent-primary);
        background: rgba(99, 102, 241, 0.1);
    }
    
    .step-item.complete {
        border-color: var(--success);
        background: rgba(16, 185, 129, 0.1);
    }
    
    .step-number {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--bg-tertiary);
        border-radius: 50%;
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--text-muted);
    }
    
    .step-item.active .step-number {
        background: var(--accent-primary);
        color: white;
    }
    
    .step-item.complete .step-number {
        background: var(--success);
        color: white;
    }
    
    .step-text {
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    
    .step-item.active .step-text,
    .step-item.complete .step-text {
        color: var(--text-primary);
    }

    /* ===== LOADING ANIMATION ===== */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .loading-shimmer {
        background: linear-gradient(90deg, var(--bg-tertiary) 25%, var(--bg-secondary) 50%, var(--bg-tertiary) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }

    /* ===== FOOTER ===== */
    .footer-info {
        text-align: center;
        padding: 1.5rem;
        color: var(--text-muted);
        font-size: 0.85rem;
        border-top: 1px solid var(--border-subtle);
        margin-top: 2rem;
    }
    
    .footer-info a {
        color: var(--accent-primary);
        text-decoration: none;
    }
    
    .footer-info a:hover {
        text-decoration: underline;
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .hero-header h1 {
            font-size: 1.75rem !important;
        }
        
        .setup-steps {
            flex-direction: column;
        }
    }
    """

    with gr.Blocks(
        title="Mac-Scientist | Deep Research Agent",
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=["Space Grotesk", "system-ui", "sans-serif"],
            font_mono=["JetBrains Mono", "monospace"],
        ).set(
            body_background_fill="#0a0a0f",
            body_background_fill_dark="#0a0a0f",
            block_background_fill="#12121a",
            block_background_fill_dark="#12121a",
            block_border_color="rgba(255, 255, 255, 0.08)",
            block_border_color_dark="rgba(255, 255, 255, 0.08)",
            input_background_fill="#1a1a24",
            input_background_fill_dark="#1a1a24",
            button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
            button_primary_background_fill_dark="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
        ),
        css=custom_css,
    ) as demo:
        
        # ===== HERO HEADER =====
        gr.HTML(f"""
        <div class="hero-header">
            <h1>🔬 Mac-Scientist</h1>
            <p class="hero-subtitle">AI-powered deep research agent with web search and analysis capabilities</p>
            <div class="hero-badges">
                <span class="badge badge-accent">
                    <span>⚡</span> {serving_mode_display}
                </span>
                {"<span class='badge badge-success'><span>🍎</span> " + platform_badge + "</span>" if platform_badge else ""}
                {"<span class='badge'><span>💾</span> " + mem_badge + " Unified Memory</span>" if mem_badge else ""}
                <span class="badge">
                    <span>🧠</span> PokeeAI/pokee_research_7b
                </span>
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:
            # ===== SETUP TAB =====
            with gr.Tab("⚙️ Setup", id="setup"):
                gr.HTML("""
                <div class="setup-steps">
                    <div class="step-item active" id="step-1">
                        <div class="step-number">1</div>
                        <div class="step-text">Configure API Keys</div>
                    </div>
                    <div class="step-item" id="step-2">
                        <div class="step-number">2</div>
                        <div class="step-text">Start Tool Server</div>
                    </div>
                    <div class="step-item" id="step-3">
                        <div class="step-number">3</div>
                        <div class="step-text">Begin Research</div>
                    </div>
                </div>
                """)
                
                with gr.Row(equal_height=True):
                    # API Keys Card
                    with gr.Column(scale=3):
                        gr.HTML("""
                        <div class="card-header">
                            <div class="card-icon">🔑</div>
                            <div>
                                <div class="card-title">API Configuration</div>
                                <div class="card-subtitle">Connect your external services</div>
                            </div>
                        </div>
                        """)
                        
                        with gr.Group():
                    serper_input = gr.Textbox(
                                label="🔍 Serper API Key",
                                placeholder="Enter your Serper API key for web search...",
                        type="password",
                        value=os.environ.get("SERPER_API_KEY", ""),
                                info="Get your key at serper.dev",
                    )

                    jina_input = gr.Textbox(
                                label="📖 Jina AI API Key",
                                placeholder="Enter your Jina AI API key for web reading...",
                        type="password",
                        value=os.environ.get("JINA_API_KEY", ""),
                                info="Get your key at jina.ai",
                    )

                    gemini_input = gr.Textbox(
                                label="✨ Gemini API Key",
                                placeholder="Enter your Gemini API key for summarization...",
                        type="password",
                        value=os.environ.get("GEMINI_API_KEY", ""),
                                info="Get your key at aistudio.google.com",
                    )

                    save_keys_btn = gr.Button(
                            "💾 Save API Keys",
                            variant="primary",
                            size="lg",
                    )
                    save_status = gr.Markdown("")

                    # Server Control Card
                    with gr.Column(scale=2):
                        gr.HTML("""
                        <div class="card-header">
                            <div class="card-icon">🖥️</div>
                            <div>
                                <div class="card-title">Tool Server</div>
                                <div class="card-subtitle">Manage the research backend</div>
                            </div>
                        </div>
                        """)
                        
                        server_status_display = gr.HTML(
                            value=f"""
                            <div class="status-pill status-stopped">
                                <span class="status-dot status-dot-red"></span>
                                Server Stopped
                            </div>
                            """
                        )

                    port_input = gr.Number(
                            label="Server Port",
                        value=tool_server_port,
                        minimum=1024,
                        maximum=65535,
                        step=1,
                    )

                    with gr.Row():
                        start_server_btn = gr.Button(
                                "▶️ Start",
                                variant="primary",
                        )
                        stop_server_btn = gr.Button(
                                "⏹️ Stop",
                                variant="stop",
                            )
                        
                        refresh_status_btn = gr.Button(
                            "🔄 Refresh Status",
                            variant="secondary",
                            size="sm",
                        )

                    server_message = gr.Markdown("")

            # ===== RESEARCH TAB =====
            with gr.Tab("🔬 Research", id="research"):
                with gr.Row(equal_height=False):
                    # Input Column
                    with gr.Column(scale=2):
                        gr.HTML("""
                        <div class="card-header">
                            <div class="card-icon">❓</div>
                            <div>
                                <div class="card-title">Research Query</div>
                                <div class="card-subtitle">Ask any question requiring deep research</div>
                            </div>
                        </div>
                        """)
                        
                        question_input = gr.Textbox(
                            label="",
                            placeholder="What would you like to research today? Ask complex questions that require searching and analyzing multiple sources...",
                            lines=4,
                            show_label=False,
                        )
                        
            with gr.Row():
                            submit_btn = gr.Button(
                                "🚀 Start Research",
                                variant="primary",
                                size="lg",
                            )
                            cancel_btn = gr.Button(
                                "⛔ Cancel",
                                variant="stop",
                                size="lg",
                            )
                        
                        with gr.Accordion("⚙️ Advanced Parameters", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.1,
                            label="Temperature",
                                info="Lower = focused & consistent, Higher = creative & varied",
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.05,
                                label="Top P (Nucleus Sampling)",
                                info="Controls diversity of token selection",
                        )
                        max_turns_research = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=10,
                            step=1,
                                label="Max Research Iterations",
                                info="More iterations = deeper research but longer time",
                            )
                        
                        gr.HTML("<div style='margin-top: 1.5rem'><strong style='color: var(--text-secondary); font-size: 0.85rem;'>💡 TRY THESE EXAMPLES</strong></div>")
                        
                        gr.Examples(
                            examples=[
                                ["What are the latest breakthroughs in quantum computing in 2024?"],
                                ["Compare the economic policies of the G7 nations on climate change"],
                                ["Explain the scientific consensus on microplastics in drinking water"],
                                ["What is the current state of nuclear fusion energy research?"],
                            ],
                            inputs=question_input,
                            label="",
                        )
                    
                    # Output Column
                    with gr.Column(scale=3):
                        gr.HTML("""
                        <div class="card-header">
                            <div class="card-icon">📊</div>
                            <div>
                                <div class="card-title">Research Progress</div>
                                <div class="card-subtitle">Real-time activity timeline</div>
                            </div>
                        </div>
                        """)
                        
                        progress_output = gr.Markdown(
                            value="""
<div class="progress-empty">
    <div class="progress-empty-icon">🔬</div>
    <div><strong>Ready to Research</strong></div>
    <div style="margin-top: 0.5rem; font-size: 0.9rem;">Enter a question and click "Start Research" to begin</div>
</div>
                            """,
                            elem_classes=["research-progress"],
                        )
                        
                        gr.HTML("""
                        <div class="answer-container">
                            <div class="answer-label">
                                <span>✅</span> Research Findings
                            </div>
                        </div>
                        """)
                        
                    answer_output = gr.Textbox(
                            label="",
                            lines=8,
                            max_lines=15,
                            show_label=False,
                        show_copy_button=True,
                            placeholder="The final answer will appear here after research is complete...",
                        )

            # ===== ABOUT TAB =====
            with gr.Tab("📚 About", id="about"):
                gr.HTML("""
                <div style="max-width: 800px; margin: 0 auto;">
                    <div class="card-header">
                        <div class="card-icon">🧠</div>
                        <div>
                            <div class="card-title">How It Works</div>
                            <div class="card-subtitle">Understanding the research process</div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
                        <div class="glass-card" style="text-align: center; padding: 1.5rem;">
                            <div style="font-size: 2rem; margin-bottom: 0.75rem;">🔍</div>
                            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">Search</div>
                            <div style="font-size: 0.85rem; color: var(--text-muted);">Queries multiple sources for relevant information</div>
                        </div>
                        <div class="glass-card" style="text-align: center; padding: 1.5rem;">
                            <div style="font-size: 2rem; margin-bottom: 0.75rem;">📖</div>
                            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">Read</div>
                            <div style="font-size: 0.85rem; color: var(--text-muted);">Analyzes web content in depth</div>
                        </div>
                        <div class="glass-card" style="text-align: center; padding: 1.5rem;">
                            <div style="font-size: 2rem; margin-bottom: 0.75rem;">🧪</div>
                            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">Analyze</div>
                            <div style="font-size: 0.85rem; color: var(--text-muted);">Synthesizes findings into insights</div>
                        </div>
                        <div class="glass-card" style="text-align: center; padding: 1.5rem;">
                            <div style="font-size: 2rem; margin-bottom: 0.75rem;">✅</div>
                            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">Verify</div>
                            <div style="font-size: 0.85rem; color: var(--text-muted);">Cross-references for accuracy</div>
                        </div>
                    </div>
                    
                    <div class="glass-card" style="margin-top: 2rem;">
                        <h3 style="margin: 0 0 1rem 0; color: var(--text-primary);">🎯 Best Practices</h3>
                        <ul style="color: var(--text-secondary); line-height: 1.8; padding-left: 1.5rem;">
                            <li><strong>Be specific</strong> — Include context and constraints in your questions</li>
                            <li><strong>Complex queries</strong> — This agent excels at multi-step research requiring multiple sources</li>
                            <li><strong>Adjust iterations</strong> — Increase max turns for deeper topics, decrease for simple lookups</li>
                            <li><strong>Temperature 0.1</strong> — Best for factual research; higher values for creative exploration</li>
                            <li><strong>Watch the timeline</strong> — Follow along as the agent searches and reads sources</li>
                        </ul>
                    </div>
                    
                    <div class="glass-card" style="margin-top: 1.5rem;">
                        <h3 style="margin: 0 0 1rem 0; color: var(--text-primary);">🛠️ Technical Stack</h3>
                        <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                            <span class="badge">PokeeAI Research 7B</span>
                            <span class="badge">MLX / MPS</span>
                            <span class="badge">Serper Search API</span>
                            <span class="badge">Jina Reader</span>
                            <span class="badge">Gemini Flash</span>
                            <span class="badge badge-accent">Apple Silicon Optimized</span>
                        </div>
                    </div>
                </div>
                """)
        
        # Footer
        gr.HTML("""
        <div class="footer-info">
            Built with 🔬 by <a href="https://pokee.ai" target="_blank">Pokee AI</a> • 
            Optimized for Apple Silicon • 
            <a href="https://github.com/Pokee-AI/PokeeResearchOSS" target="_blank">GitHub</a>
        </div>
        """)

        # ===== EVENT HANDLERS =====
        
        def update_server_status_html(is_running: bool, message: str) -> str:
            if is_running:
                return f"""
                <div class="status-pill status-running">
                    <span class="status-dot status-dot-green"></span>
                    {message}
                </div>
                """
            else:
                return f"""
                <div class="status-pill status-stopped">
                    <span class="status-dot status-dot-red"></span>
                    {message}
                </div>
                """
        
        def refresh_status_html() -> str:
            status = get_tool_server_status()
            return update_server_status_html(status["running"], "Running" if status["running"] else "Stopped")
        
        def start_server_with_html(port: int) -> tuple[str, str]:
            msg, _ = start_tool_server_ui(port)
            status = get_tool_server_status()
            html = update_server_status_html(status["running"], "Running" if status["running"] else "Stopped")
            return msg, html
        
        def stop_server_with_html() -> tuple[str, str]:
            msg, _ = stop_tool_server_ui()
            return msg, update_server_status_html(False, "Stopped")
        
        save_keys_btn.click(
            fn=save_api_keys,
            inputs=[serper_input, jina_input, gemini_input],
            outputs=[save_status],
        )

        start_server_btn.click(
            fn=start_server_with_html,
            inputs=[port_input],
            outputs=[server_message, server_status_display],
        )

        stop_server_btn.click(
            fn=stop_server_with_html,
            inputs=[],
            outputs=[server_message, server_status_display],
        )

        refresh_status_btn.click(
            fn=refresh_status_html,
            inputs=[],
            outputs=[server_status_display],
        )

        submit_event = submit_btn.click(
            fn=research_stream,
            inputs=[
                question_input,
                temperature_slider,
                top_p_slider,
                max_turns_research,
            ],
            outputs=[progress_output, answer_output],
            concurrency_limit=30,
        )

        cancel_btn.click(
            fn=cancel_research,
            inputs=[],
            outputs=[progress_output, answer_output],
            cancels=[submit_event],
            concurrency_limit=30,
        )

    return demo


def main():
    """Main entry point for the Gradio application.

    Parses command-line arguments, configures the agent backend, optionally
    pre-loads models (for local/mlx modes), and launches the Gradio interface.
    Registers cleanup handlers for graceful shutdown.
    """
    global tool_server_proc

    # Determine best defaults based on platform
    default_mode = "mlx" if is_apple_silicon() else "local"
    default_device = get_default_device()

    parser = argparse.ArgumentParser(
        description="Pokee Deep Research Agent - Web Interface"
    )
    parser.add_argument(
        "--serving-mode",
        type=str,
        choices=["local", "mlx", "mlx-server", "vllm"],
        default=default_mode,
        help=f"Serving mode (default: {default_mode}). "
             "'local' = HuggingFace Transformers, "
             "'mlx' = Apple Silicon native, "
             "'mlx-server' = MLX-LM server, "
             "'vllm' = vLLM server",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:9999/v1",
        help="vLLM server URL (for --serving-mode vllm)",
    )
    parser.add_argument(
        "--mlx-server-url",
        type=str,
        default="http://localhost:8080/v1",
        help="MLX-LM server URL (for --serving-mode mlx-server)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="PokeeAI/pokee_research_7b",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--tool-config",
        type=str,
        default="config/tool_config/pokee_tool_config.yaml",
        help="Tool configuration file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["auto", "cuda", "mps", "cpu"],
        help=f"Device for local mode (default: {default_device})",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7777,
        help="Port to run the web interface on",
    )

    args = parser.parse_args()

    # Validate mode-specific requirements
    if args.serving_mode == "vllm" and not args.vllm_url:
        parser.error("--vllm-url is required when using --serving-mode vllm")
    
    if args.serving_mode == "mlx" and not is_apple_silicon():
        parser.error("--serving-mode mlx requires Apple Silicon (M1/M2/M3)")

    # Configure global settings
    global serving_mode, agent_config
    serving_mode = args.serving_mode
    agent_config = {
        "model_path": args.model_path,
        "device": args.device,
        "max_turns": 10,
        "tool_config_path": args.tool_config,
        "vllm_url": args.vllm_url if args.serving_mode == "vllm" else None,
        "mlx_server_url": args.mlx_server_url if args.serving_mode == "mlx-server" else None,
    }

    # Log platform info
    if is_apple_silicon():
        mem_gb = get_system_memory_gb()
        logger.info(f"Apple Silicon detected - {mem_gb:.0f}GB unified memory")

    logger.info(f"Starting Gradio app with {args.serving_mode.upper()} serving mode")
    logger.info(f"Configuration: {agent_config}")

    # Pre-initialize resources for local/mlx agents
    if args.serving_mode in ["local", "mlx"]:
        logger.info("Pre-loading model...")
        create_agent()
        logger.info("Model loaded successfully!")
    else:
        logger.info(f"Using {args.serving_mode.upper()} server (no pre-loading needed)")

    # Register cleanup handler for tool server
    atexit.register(cleanup_tool_server, tool_server_proc)

    # Launch interface
    try:
        demo = create_demo()
        demo.launch(share=args.share, server_port=args.port)
    finally:
        if tool_server_proc is not None:
            cleanup_tool_server(tool_server_proc)


if __name__ == "__main__":
    main()
