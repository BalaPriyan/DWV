import os
import logging
import threading
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DWV_Main")

logger.info("Starting DWV initialization...")

espeak_path = os.environ.get("ESPEAK_PATH", r"C:\Program Files\eSpeak NG")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(espeak_path, "libespeak-ng.dll")
os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(espeak_path, "espeak-ng.exe")
os.environ["PATH"] = espeak_path + os.pathsep + os.environ.get("PATH", "")

from engine.soundEngine import SoundEngine
from engine.whisperEngine import WhisperEngine
from engine.actionEngine import ActionEngine
from engine.ttsEngine import TTSEngine

logger.info("Initializing SoundEngine...")
engine = SoundEngine()

logger.info("Initializing WhisperEngine...")
whisper_engine = WhisperEngine()

logger.info("Initializing ActionEngine...")
action_engine = ActionEngine()

logger.info("Initializing TTSEngine (this may take a moment)...")
tts_engine = TTSEngine()

last_text = None

def consumer_loop():
    global last_text
    logger.info("Waiting for audio...")
    
    # engine.get_audio() audio buffers from the thread-safe queue
    for audio in engine.get_audio():
        try:
            logger.info("Processing captured audio...")
            text = whisper_engine.transcribe(audio)

            if not text:
                continue

            if text == last_text:
                continue

            last_text = text
            logger.info(f"You said: {text}")
            
            ai_response_text = action_engine.execute(text)
            logger.info(f"Agent Reply: {ai_response_text}")

            if ai_response_text:
                engine.pause()
                tts_engine.speak(ai_response_text)
                engine.resume()

        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    try:
        # Start the sound engine
        engine.start()
        
        # Run the consumer in the main thread
        consumer_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        engine.stop()