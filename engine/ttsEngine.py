import logging
import torch
import os
import sounddevice as sd
from TTS.api import TTS
from exceptions import TTSError

logger = logging.getLogger(__name__)

import queue
import threading

class TTSEngine:
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = os.environ.get("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
            
        logger.info(f"Initializing TTSEngine with model: {model_name}...")
        try:
            self.use_gpu = torch.cuda.is_available() and "xtts" in model_name.lower()
            
            if self.use_gpu:
                logger.info("Using GPU for XTTS to get realistic, fast speech.")
            else:
                logger.info("Using CPU for TTS stability.")
            
            self.tts = TTS(model_name, progress_bar=False, gpu=self.use_gpu)
            
            self.speaker_wav = os.path.join(os.getcwd(), "speaker.wav")
            if os.path.exists(self.speaker_wav):
                logger.info(f"Using custom human voice clone from: {self.speaker_wav}")
                self.speaker_name = None
            else:
                logger.info("No 'speaker.wav' found. Using built-in speaker.")
                self.speaker_wav = None
                self.speaker_name = os.environ.get("TTS_SPEAKER_NAME", "Claribel Dervla")
            
            if hasattr(self.tts, 'synthesizer') and self.tts.synthesizer:
                self.sample_rate = self.tts.synthesizer.output_sample_rate
            else:
                self.sample_rate = 24000 if "xtts" in model_name.lower() else 22050
                
            self.sentence_queue = queue.Queue()
            self.play_queue = queue.Queue()
            self.is_running = True
            
            self.synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
            self.synth_thread.start()
            
            self.play_thread = threading.Thread(target=self._play_worker, daemon=True)
            self.play_thread.start()
            
            logger.info(f"Coqui TTS loaded successfully (Sample Rate: {self.sample_rate}Hz)")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTSEngine: {e}")
            self.tts = None
            raise TTSError(f"TTS Initialization failed: {e}")

    def _synth_worker(self):
        """Background thread to synthesize sentences into audio chunks."""
        while self.is_running:
            try:
                sentence = self.sentence_queue.get(timeout=1)
                if sentence is None: break
                
                logger.info(f"Background synthesizing: '{sentence}'")
                if "xtts" in self.tts.model_name.lower():
                    if self.speaker_wav:
                        wav = self.tts.tts(text=sentence, speaker_wav=self.speaker_wav, language="en")
                    else:
                        wav = self.tts.tts(text=sentence, speaker=self.speaker_name, language="en")
                else:
                    wav = self.tts.tts(text=sentence)
                
                self.play_queue.put(wav)
                self.sentence_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Synth worker error: {e}")

    def _play_worker(self):
        """Background thread to play audio chunks sequentially."""
        while self.is_running:
            try:
                wav = self.play_queue.get(timeout=1)
                if wav is None: break
                
                logger.debug("Playing next chunk from queue...")
                sd.play(wav, self.sample_rate)
                sd.wait()
                self.play_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Play worker error: {e}")

    def speak(self, text):
        if not self.tts:
            logger.error("TTS Engine is not initialized properly.")
            return

        if not text or len(text.strip()) == 0:
            return

        try:
            import re
            sentences = re.split(r'(?<=[.!?]) +', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    self.sentence_queue.put(sentence)
            
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            raise TTSError(f"Speech synthesis failed: {e}")
            
    def wait_until_done(self):
        """Blocks until all queued speech is finished."""
        self.sentence_queue.join()
        self.play_queue.join()
