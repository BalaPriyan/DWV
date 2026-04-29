import logging
import threading
from engine.soundEngine import SoundEngine
from engine.whisperEngine import WhisperEngine
from engine.actionEngine import ActionEngine

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DWV_Main")

engine = SoundEngine()
whisper_engine = WhisperEngine()
action_engine = ActionEngine()

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
            
            # Send text to Ollama-powered Action Engine
            ai_response = action_engine.execute(text)
            logger.info(f"Agent Reply: {ai_response}")

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