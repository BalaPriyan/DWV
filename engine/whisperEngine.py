import numpy as np
import whisper
import torch
import logging
import os

logger = logging.getLogger(__name__)

class WhisperEngine:
    def __init__(self, model_size=None):
        if model_size is None:
            model_size = os.environ.get("WHISPER_MODEL", "tiny")
            
        try:
            logger.info(f"Loading Whisper model ({model_size})...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model(model_size, device=self.device)
            logger.info(f"Whisper model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def transcribe(self, audio):
        try:
            if audio is None or not isinstance(audio, np.ndarray) or audio.size == 0:
                logger.warning("Invalid or empty audio buffer passed to Whisper.")
                return None

            # Flatten audio
            audio = audio.flatten()

            # Normalize safely
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            else:
                logger.debug("Silent audio detected")
                return None

            # Convert dtype
            audio = audio.astype("float32")

            # Transcribe (fp16 is only supported on CUDA)
            fp16 = torch.cuda.is_available()
            result = self.model.transcribe(audio, fp16=fp16)

            if not result or "text" not in result:
                logger.warning("Whisper returned invalid result")
                return None

            text = result["text"].strip()

            if not text or text == "..." or len(text) < 2:
                logger.debug("Ignored noise")
                return None

            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None