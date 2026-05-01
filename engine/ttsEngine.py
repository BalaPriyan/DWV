import logging
import torch
import os
import sounddevice as sd
from TTS.api import TTS

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = os.environ.get("TTS_MODEL", "tts_models/en/ljspeech/vits")
            
        logger.info(f"Initializing TTSEngine with model: {model_name}...")
        try:
            self.use_gpu = False
            logger.info("Using CPU for TTS to ensure stability.")
            
            self.tts = TTS(model_name, progress_bar=False, gpu=self.use_gpu)
            
            if hasattr(self.tts, 'synthesizer') and self.tts.synthesizer:
                self.sample_rate = self.tts.synthesizer.output_sample_rate
            else:
                self.sample_rate = 22050
                
            logger.info(f"Coqui TTS loaded successfully on CPU (Sample Rate: {self.sample_rate}Hz)")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTSEngine: {e}")
            self.tts = None

    def speak(self, text):
        if not self.tts:
            logger.error("TTS Engine is not initialized properly.")
            return

        if not text or len(text.strip()) == 0:
            return

        try:
            logger.info(f"Synthesizing speech for: '{text}'")
            wav = self.tts.tts(text=text)
            
            logger.debug("Playing audio...")
            sd.play(wav, self.sample_rate)
            sd.wait()
            
        except Exception as e:
            logger.error(f"TTS Synthesis error: {e}")
