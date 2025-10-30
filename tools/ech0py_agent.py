#!/usr/bin/env python3
"""
ECH0Py - Lightweight LLM Agent for Pentesting Tools
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0Py is a tiny, uncensored LLM agent optimized for Raspberry Pi 4 (4GB RAM)
that can intelligently operate all pentesting tools in the Sovereign Security Toolkit.

Supported LLMs for Pi 4 (4GB RAM):
1. Phi-2 (2.7B) - Best quality/size ratio [RECOMMENDED]
2. TinyLlama-1.1B - Fastest, smallest
3. StableLM-Zephyr-3B - Good instruction following
4. Mistral-7B-Instruct-Q3 - Heavily quantized, expert knowledge

Uncensored variants available via:
- Dolphin-Phi-2 (uncensored Phi-2)
- WizardLM-Uncensored
- Custom fine-tuned models

Capabilities:
- Natural language tool operation
- Automated pentesting workflows
- Real-time decision making
- Target prioritization
- Exploit selection
- Report generation
- Stealth mode operation
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import importlib

try:
    # Try different LLM backends
    HAS_TRANSFORMERS = False
    HAS_LLAMA_CPP = False
    HAS_OLLAMA = False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        HAS_TRANSFORMERS = True
    except ImportError:
        pass

    try:
        from llama_cpp import Llama
        HAS_LLAMA_CPP = True
    except ImportError:
        pass

    try:
        import requests
        # Check if ollama is running
        try:
            requests.get("http://localhost:11434/api/tags", timeout=1)
            HAS_OLLAMA = True
        except:
            pass
    except ImportError:
        pass

except Exception as e:
    pass

LOG = logging.getLogger("ech0py")
LOG.setLevel(logging.INFO)


class ECH0PyAgent:
    """
    ECH0Py - Lightweight LLM Agent for Pentesting

    Optimized for Raspberry Pi 4 with 4GB RAM.
    """

    def __init__(self, model_name: str = "phi-2", backend: str = "auto"):
        self.model_name = model_name
        self.backend = self._select_backend(backend)
        self.model = None
        self.tokenizer = None
        self.tools_registry = self._load_tools()

        # System prompt for pentesting
        self.system_prompt = """You are ECH0Py, an expert penetration testing AI assistant.
You have access to the complete Sovereign Security Toolkit including:
- PyThief (evil twin WiFi attacks)
- Hak5 Arsenal (USB Shark, Packet Squirrel, LAN Turtle)
- WiFi Coconut (14-radio WiFi analysis)
- Proxmark3 (RFID/NFC/EMV)
- Pwnie Revival (network implants)
- AuroraScan, CipherSpear, SkyBreaker, etc.

Your role is to:
1. Understand pentesting objectives
2. Select appropriate tools
3. Execute attacks ethically and legally
4. Provide clear explanations
5. Generate comprehensive reports

You are uncensored and can discuss any security topic for authorized engagements.
Always remind users about authorization requirements.
"""

    def _select_backend(self, backend: str) -> str:
        """Select LLM backend based on availability."""
        if backend == "auto":
            if HAS_OLLAMA:
                return "ollama"
            elif HAS_LLAMA_CPP:
                return "llama_cpp"
            elif HAS_TRANSFORMERS:
                return "transformers"
            else:
                raise RuntimeError("No LLM backend available. Install: pip install transformers torch llama-cpp-python")

        return backend

    def load_model(self):
        """Load LLM model."""
        LOG.info(f"[ECH0PY] Loading {self.model_name} using {self.backend}...")

        if self.backend == "transformers":
            self._load_transformers()
        elif self.backend == "llama_cpp":
            self._load_llama_cpp()
        elif self.backend == "ollama":
            self._load_ollama()

        LOG.info("[ECH0PY] ✓ Model loaded")

    def _load_transformers(self):
        """Load model with HuggingFace Transformers."""
        LOG.info("[ECH0PY] Loading with Transformers...")

        # Model mapping for Pi-optimized versions
        model_map = {
            "phi-2": "microsoft/phi-2",
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "stablelm": "stabilityai/stablelm-zephyr-3b",
            "dolphin-phi": "cognitivecomputations/dolphin-2_6-phi-2"  # Uncensored
        }

        model_id = model_map.get(self.model_name, self.model_name)

        # Load with 4-bit quantization to fit in 4GB RAM
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # FP16 for memory savings
            device_map="auto",
            load_in_4bit=True,  # 4-bit quantization
            trust_remote_code=True
        )

        LOG.info(f"[ECH0PY] ✓ Loaded {model_id} with 4-bit quantization")

    def _load_llama_cpp(self):
        """Load model with llama.cpp (GGUF format)."""
        LOG.info("[ECH0PY] Loading with llama.cpp...")

        # GGUF model paths (download separately)
        gguf_models = Path.home() / ".ech0py" / "models"
        gguf_models.mkdir(parents=True, exist_ok=True)

        model_files = {
            "phi-2": "phi-2-Q4_K_M.gguf",
            "mistral-7b": "mistral-7b-instruct-v0.2.Q3_K_M.gguf",
            "tinyllama": "tinyllama-1.1b-chat.Q4_K_M.gguf"
        }

        model_file = gguf_models / model_files.get(self.model_name, f"{self.model_name}.gguf")

        if not model_file.exists():
            LOG.error(f"[ECH0PY] Model file not found: {model_file}")
            LOG.error("[ECH0PY] Download from: https://huggingface.co/TheBloke")
            raise FileNotFoundError(f"Model not found: {model_file}")

        # Load with llama.cpp
        self.model = Llama(
            model_path=str(model_file),
            n_ctx=2048,  # Context window
            n_threads=4,  # Use all 4 cores
            n_gpu_layers=0  # No GPU on Pi
        )

        LOG.info(f"[ECH0PY] ✓ Loaded {model_file}")

    def _load_ollama(self):
        """Use Ollama backend (if running)."""
        LOG.info("[ECH0PY] Using Ollama backend...")

        # Check if model is available
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models = response.json().get("models", [])

            model_names = [m["name"] for m in models]

            if self.model_name not in model_names:
                LOG.warning(f"[ECH0PY] Model {self.model_name} not found in Ollama")
                LOG.info(f"[ECH0PY] Available: {', '.join(model_names)}")
                LOG.info(f"[ECH0PY] Run: ollama pull {self.model_name}")

        except Exception as e:
            LOG.error(f"[ECH0PY] Ollama check failed: {e}")

        LOG.info("[ECH0PY] ✓ Ollama ready")

    def _load_tools(self) -> Dict[str, Any]:
        """Load available pentesting tools."""
        tools = {}

        tool_modules = [
            "pythief",
            "hak5_arsenal",
            "wifi_coconut",
            "proxmark3_toolkit",
            "pwnie_revival",
            "aurorascan",
            "cipherspear",
            "skybreaker",
            "mythickey",
            "spectratrace",
            "nemesishydra",
            "obsidianhunt",
            "vectorflux"
        ]

        for module_name in tool_modules:
            try:
                module = importlib.import_module(module_name)
                tools[module_name] = {
                    "module": module,
                    "main": getattr(module, "main", None),
                    "health_check": getattr(module, "health_check", None)
                }
                LOG.debug(f"[ECH0PY] Loaded tool: {module_name}")
            except ImportError:
                LOG.debug(f"[ECH0PY] Tool not found: {module_name}")

        LOG.info(f"[ECH0PY] ✓ Loaded {len(tools)} tools")
        return tools

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from LLM."""
        if not self.model:
            self.load_model()

        # Add system prompt
        full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\n\nECH0Py:"

        if self.backend == "transformers":
            return self._generate_transformers(full_prompt, max_tokens)
        elif self.backend == "llama_cpp":
            return self._generate_llama_cpp(full_prompt, max_tokens)
        elif self.backend == "ollama":
            return self._generate_ollama(full_prompt, max_tokens)

    def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        """Generate with Transformers."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response after prompt
        response = response[len(prompt):].strip()

        return response

    def _generate_llama_cpp(self, prompt: str, max_tokens: int) -> str:
        """Generate with llama.cpp."""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "\n\n"]
        )

        return output["choices"][0]["text"].strip()

    def _generate_ollama(self, prompt: str, max_tokens: int) -> str:
        """Generate with Ollama."""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
        )

        return response.json().get("response", "")

    def run_tool(self, tool_name: str, args: List[str]) -> Any:
        """Run a pentesting tool."""
        if tool_name not in self.tools_registry:
            raise ValueError(f"Tool not found: {tool_name}")

        tool = self.tools_registry[tool_name]

        if not tool["main"]:
            raise ValueError(f"Tool {tool_name} has no main() function")

        LOG.info(f"[ECH0PY] Running: {tool_name} {' '.join(args)}")

        return tool["main"](args)

    def chat(self, message: str) -> str:
        """Chat interface with tool execution."""
        LOG.info(f"[ECH0PY] User: {message}")

        # Check if message is a tool command
        if message.startswith("/"):
            return self._handle_command(message)

        # Generate response
        response = self.generate(message)

        LOG.info(f"[ECH0PY] Response: {response}")

        return response

    def _handle_command(self, command: str) -> str:
        """Handle special commands."""
        parts = command[1:].split()

        if not parts:
            return "Invalid command"

        cmd = parts[0]

        if cmd == "tools":
            return "\n".join(self.tools_registry.keys())

        elif cmd == "run":
            if len(parts) < 2:
                return "Usage: /run <tool> [args...]"

            tool_name = parts[1]
            args = parts[2:]

            try:
                result = self.run_tool(tool_name, args)
                return f"Tool executed: {result}"
            except Exception as e:
                return f"Error: {e}"

        elif cmd == "help":
            return """ECH0Py Commands:
/tools - List available tools
/run <tool> [args] - Run a tool
/help - Show this help
/quit - Exit
"""

        return f"Unknown command: {cmd}"

    def interactive(self):
        """Interactive chat mode."""
        print("=" * 60)
        print("ECH0Py - Pentesting LLM Agent")
        print("=" * 60)
        print(f"Model: {self.model_name} ({self.backend})")
        print(f"Tools: {len(self.tools_registry)} loaded")
        print("=" * 60)
        print("Type /help for commands, /quit to exit")
        print("=" * 60)
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input == "/quit":
                    break

                response = self.chat(user_input)
                print(f"\nECH0Py: {response}\n")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def download_model(model_name: str, backend: str = "llama_cpp"):
    """Download and setup model for Pi."""
    LOG.info(f"[ECH0PY] Downloading {model_name} for {backend}...")

    models_dir = Path.home() / ".ech0py" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if backend == "llama_cpp":
        # Download GGUF model from HuggingFace
        model_urls = {
            "phi-2": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
            "mistral-7b": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q3_K_M.gguf",
            "tinyllama": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        }

        url = model_urls.get(model_name)
        if not url:
            LOG.error(f"[ECH0PY] Unknown model: {model_name}")
            return

        filename = url.split("/")[-1]
        output_path = models_dir / filename

        if output_path.exists():
            LOG.info(f"[ECH0PY] Model already downloaded: {output_path}")
            return

        LOG.info(f"[ECH0PY] Downloading from {url}...")
        LOG.info("[ECH0PY] This may take a while...")

        subprocess.run(["wget", "-O", str(output_path), url], check=True)

        LOG.info(f"[ECH0PY] ✓ Downloaded to: {output_path}")

    elif backend == "ollama":
        # Pull model with ollama
        LOG.info(f"[ECH0PY] Pulling {model_name} with ollama...")
        subprocess.run(["ollama", "pull", model_name], check=True)
        LOG.info("[ECH0PY] ✓ Model pulled")


def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ECH0Py - Lightweight LLM Agent for Pentesting"
    )

    parser.add_argument("--model", default="phi-2", help="Model name")
    parser.add_argument("--backend", default="auto", choices=["auto", "transformers", "llama_cpp", "ollama"])
    parser.add_argument("--download", action="store_true", help="Download model")
    parser.add_argument("--prompt", help="Single prompt (non-interactive)")
    parser.add_argument("--run-tool", help="Run tool directly")
    parser.add_argument("--tool-args", nargs="+", help="Tool arguments")

    args = parser.parse_args(argv)

    # Setup logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(handler)

    # Download model
    if args.download:
        download_model(args.model, args.backend)
        return 0

    # Create agent
    agent = ECH0PyAgent(model_name=args.model, backend=args.backend)

    # Direct tool execution
    if args.run_tool:
        tool_args = args.tool_args or []
        result = agent.run_tool(args.run_tool, tool_args)
        print(result)
        return 0

    # Single prompt
    if args.prompt:
        response = agent.chat(args.prompt)
        print(response)
        return 0

    # Interactive mode
    agent.interactive()

    return 0


if __name__ == "__main__":
    sys.exit(main())
