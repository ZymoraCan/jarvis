#!/usr/bin/env python3
"""
JARVIS GUI with Voice Support (Local Model)
Full GUI + Voice I/O integration
"""
import asyncio
import threading
import pyttsx3
import speech_recognition as sr
from pathlib import Path
from openai import OpenAI
from ui import JarvisUI
from config import get_ollama_endpoint, get_ollama_model
from memory.memory_manager import load_memory, format_memory_for_prompt

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)

class JarvisVoiceGUI:
    def __init__(self):
        # Initialize UI
        face_path = Path(__file__).parent / "assets" / "face.png"
        self.ui = JarvisUI(str(face_path))
        
        # Initialize local model client
        try:
            self.client = OpenAI(
                api_key="not-needed",
                base_url=get_ollama_endpoint()
            )
            self.model = get_ollama_model()
            self.ui.write_log(f"✅ Connected to local model: {self.model}")
        except Exception as e:
            self.ui.write_log(f"❌ Connection failed: {e}")
            self.client = None
            
        self.ui.set_state("LISTENING")
        
        # Bind UI callbacks
        self.ui.on_text_command = self._on_text_command
        self.recognizer = sr.Recognizer()
        self.listening = False
        self.is_speaking = False
        
    def speak(self, text: str):
        """Speak text using TTS"""
        self.is_speaking = True
        self.ui.set_state("SPEAKING")
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            self.ui.write_log(f"❌ TTS Error: {e}")
        finally:
            self.is_speaking = False
            self.ui.set_state("LISTENING")
    
    def listen_voice(self):
        """Listen for voice input in background thread"""
        if not self.listening:
            return
            
        try:
            with sr.Microphone() as source:
                self.ui.write_log("🎤 Listening for voice...")
                self.ui.set_state("LISTENING")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5)
            
            # Try Turkish first, then English
            try:
                text = self.recognizer.recognize_google(audio, language="tr-TR")
            except:
                text = self.recognizer.recognize_google(audio, language="en-US")
            
            self.ui.write_log(f"You: {text}")
            self._process_command(text)
            
        except sr.UnknownValueError:
            self.ui.write_log("❌ Could not understand audio")
        except sr.RequestError as e:
            self.ui.write_log(f"❌ Speech recognition error: {e}")
        except Exception as e:
            self.ui.write_log(f"❌ Mic error: {e}")
    
    def _on_text_command(self, text: str):
        """Handle text input from UI"""
        self.ui.write_log(f"You: {text}")
        self._process_command(text)
    
    def _process_command(self, user_input: str):
        """Process command and get response"""
        if not self.client:
            self.ui.write_log("❌ Not connected to local model")
            return
        
        def process_thread():
            try:
                self.ui.set_state("THINKING")
                self.ui.write_log("🔄 Processing...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=300
                )
                
                answer = response.choices[0].message.content
                self.ui.write_log(f"JARVIS: {answer}")
                
                # Speak the response
                self.speak(answer)
                
            except Exception as e:
                self.ui.write_log(f"❌ Error: {e}")
                self.ui.set_state("LISTENING")
        
        # Run in background thread to not block UI
        threading.Thread(target=process_thread, daemon=True).start()
    
    def start_voice_mode(self):
        """Start listening for voice commands"""
        self.listening = True
        self.ui.write_log("🎤 Voice mode enabled")
        
        def voice_loop():
            while self.listening:
                self.listen_voice()
                threading.Event().wait(0.5)  # Small delay
        
        threading.Thread(target=voice_loop, daemon=True).start()
    
    def stop_voice_mode(self):
        """Stop listening for voice commands"""
        self.listening = False
        self.ui.write_log("📝 Voice mode disabled")
    
    def run(self):
        """Start the GUI"""
        self.start_voice_mode()
        self.ui.root.mainloop()

if __name__ == "__main__":
    jarvis = JarvisVoiceGUI()
    jarvis.run()
