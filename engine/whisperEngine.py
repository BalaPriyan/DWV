import numpy as np
import whisper


class WhisperEngine:
    def __init__(self, model_size="tiny"):
        try:
            print("Loading Whisper model...")
            self.model = whisper.load_model(model_size)
            print("Whisper model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def transcribe(self, audio):
        try:
            if audio is None:
                print("Audio is None")
                return None

            if not isinstance(audio, np.ndarray):
                print("Invalid audio type")
                return None

            if audio.size == 0:
                print("Empty audio buffer")
                return None

            # Flatten audio
            audio = audio.flatten()

            # Normalize safely
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            else:
                print("Silent audio detected")
                return None

            # Convert dtype
            audio = audio.astype("float32")

            # Transcribe
            result = self.model.transcribe(audio)

            if not result or "text" not in result:
                print("Whisper returned invalid result")
                return None

            text = result["text"].strip()

            if not text or text == "..." or len(text) < 2:
                print("Ignored noise")
                return None

            return text

        except Exception as e:
            print(f"Transcription error: {e}")
            return None