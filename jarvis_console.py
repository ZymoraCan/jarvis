#!/usr/bin/env python3
"""
Console JARVIS for local model providers
"""
import json
from pathlib import Path
from openai import OpenAI
from config import get_ollama_endpoint, get_ollama_model
from memory.memory_manager import load_memory, format_memory_for_prompt

def _load_system_prompt() -> str:
    try:
        prompt_path = Path(__file__).parent / "core" / "prompt.txt"
        return prompt_path.read_text(encoding="utf-8")
    except Exception:
        return "You are JARVIS, Tony Stark's AI assistant. Be helpful and direct."

def main():
    print("🤖 JARVIS Console Mode (Local Model)")
    print("=" * 50)

    # Test connection
    try:
        client = OpenAI(
            api_key="not-needed",
            base_url=get_ollama_endpoint()
        )
        model = get_ollama_model()
        print(f"✅ Connected to local model: {model}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    print("\n💬 Type your commands (or 'quit' to exit)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("JARVIS: Goodbye!")
                break

            if not user_input:
                continue

            # Build simple message
            full_message = user_input

            # Call local model provider
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": full_message}
                ],
                max_tokens=500
            )

            message = response.choices[0].message

            if message.tool_calls:
                print("JARVIS: Executing tool...")
                # For now, just show the tool call
                for tool_call in message.tool_calls:
                    print(f"🔧 Tool: {tool_call.function.name}")
                    print(f"📋 Args: {tool_call.function.arguments}")
            else:
                print(f"JARVIS: {message.content}")

        except KeyboardInterrupt:
            print("\nJARVIS: Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()