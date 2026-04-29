#!/usr/bin/env python3
"""
JARVIS with Voice Support (Local Model)
Supports both voice and text input
"""
import sys
import threading
import pyttsx3
import speech_recognition as sr
from pathlib import Path
from openai import OpenAI
from config import get_ollama_endpoint, get_ollama_model
from memory.memory_manager import load_memory, format_memory_for_prompt

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed
tts_engine.setProperty('volume', 1.0)  # Volume

def _load_system_prompt() -> str:
    try:
        prompt_path = Path(__file__).parent / "core" / "prompt.txt"
        return prompt_path.read_text(encoding="utf-8")
    except Exception:
        return "You are JARVIS, Tony Stark's AI assistant. Be helpful and direct."

def speak(text: str):
    """Speak text using TTS"""
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"❌ TTS Error: {e}")

def listen() -> str:
    """Listen for voice input"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=10)
        
        # Try multiple languages
        try:
            # Try Turkish first
            text = recognizer.recognize_google(audio, language="tr-TR")
            print(f"You: {text}")
            return text
        except:
            # Fallback to English
            text = recognizer.recognize_google(audio, language="en-US")
            print(f"You: {text}")
            return text
            
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")
        return ""
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return ""

def main():
    print("🤖 JARVIS Voice Mode (Local Model)")
    print("=" * 50)
    print("Commands:")
    print("  'voice' - Switch to voice input")
    print("  'text'  - Switch to text input")
    print("  'quit'  - Exit")
    print("-" * 50)

    # Test connection
    try:
        client = OpenAI(
            api_key="not-needed",
            base_url=get_ollama_endpoint()
        )
        model = get_ollama_model()
        print(f"✅ Connected to local model: {model}")
        speak("Connected to JARVIS. How can I help?")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    print()
    input_mode = "text"  # Start with text mode
    
    while True:
        try:
            # Get user input
            if input_mode == "voice":
                user_input = listen().strip()
                if not user_input:
                    continue
            else:
                user_input = input("\nYou: ").strip()
                
            if user_input.lower() == "voice":
                input_mode = "voice"
                speak("Switched to voice mode")
                print("🎤 Voice mode enabled")
                continue
                
            if user_input.lower() == "text":
                input_mode = "text"
                print("📝 Text mode enabled")
                speak("Switched to text mode")
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q', 'çık']:
                speak("Goodbye!")
                print("JARVIS: Goodbye!")
                break

            if not user_input:
                continue

            # Get response from local model
            print("🔄 Processing...", end=" ", flush=True)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300
            )

            answer = response.choices[0].message.content
            print("\r" + " " * 30 + "\r", end="")  # Clear line
            print(f"JARVIS: {answer}")
            
            # Speak the response
            speak(answer)

        except KeyboardInterrupt:
            speak("Goodbye!")
            print("\nJARVIS: Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            if input_mode == "voice":
                speak("I encountered an error")

if __name__ == "__main__":
    main()
