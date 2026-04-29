#!/usr/bin/env python3
"""
Text-only JARVIS for local model providers
"""
import asyncio
import json
import threading
from pathlib import Path
from openai import OpenAI
from ui import JarvisUI
from config import get_ollama_endpoint, get_ollama_model, get_model_type
from local_router import route_local_command
from memory.memory_manager import load_memory, update_memory, format_memory_for_prompt

# Import all actions
from actions.flight_finder import flight_finder
from actions.open_app import open_app
from actions.weather_report import weather_action
from actions.send_message import send_message
from actions.reminder import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor import screen_process
from actions.youtube_video import youtube_video
from actions.desktop import desktop_control
from actions.browser_control import browser_control
from actions.file_controller import file_controller
from actions.code_helper import code_helper
from actions.dev_agent import dev_agent
from actions.web_search import web_search as web_search_action
from actions.computer_control import computer_control
from actions.game_updater import game_updater


def _load_system_prompt() -> str:
    """Load system prompt from core/prompt.txt"""
    try:
        prompt_path = Path(__file__).parent / "core" / "prompt.txt"
        return prompt_path.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )


class TextJarvis:
    def __init__(self, ui: JarvisUI):
        self.ui = ui
        self.client = OpenAI(
            api_key="not-needed",
            base_url=get_ollama_endpoint(),
            timeout=600.0,
        )
        self.model = get_ollama_model()
        self.ui.on_text_command = self._on_text_command

    def _on_text_command(self, text: str):
        """Handle text input from user"""
        threading.Thread(target=self._process_text, args=(text,), daemon=True).start()

    def _process_text(self, user_text: str):
        """Process user text and respond"""
        try:
            self.ui.set_state("THINKING")

            routed = route_local_command(user_text, player=self.ui)
            if routed is not None:
                self.ui.write_log(f"JARVIS: {routed}")
                return

            self.ui.write_log("SYS: Processing request on local model (first response may be slow)...")

            # Build context
            memory = load_memory()
            mem_str = format_memory_for_prompt(memory)
            sys_prompt = _load_system_prompt()

            context = f"{sys_prompt}\n\n"
            if mem_str:
                context += f"{mem_str}\n\n"

            # Call local model provider
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=96,
                temperature=0.4,
            )

            message = response.choices[0].message

            content = (message.content or "").strip()
            if not content:
                content = "No text response produced yet. Try a shorter prompt."
            self.ui.write_log(f"JARVIS: {content}")

        except Exception as e:
            self.ui.write_log(f"ERR: {str(e)}")
        finally:
            self.ui.set_state("LISTENING")

    def _execute_tool(self, tool_call):
        """Execute a tool call"""
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        try:
            if name == "open_app":
                return open_app(parameters=args, response=None, player=self.ui) or f"Opened {args.get('app_name')}"

            elif name == "web_search":
                return web_search_action(parameters=args, player=self.ui) or "Search completed"

            elif name == "weather_report":
                return weather_action(parameters=args, player=self.ui) or "Weather delivered"

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            return f"Tool failed: {e}"


def main():
    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()
        jarvis = TextJarvis(ui)
        # Keep alive - UI handles the loop

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()


if __name__ == "__main__":
    main()
