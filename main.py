from engine.soundEngine import SoundEngine
from engine.whisperEngine import WhisperEngine

engine = SoundEngine()
whisper_engine = WhisperEngine()

last_text = None


def handleAudio(audio):
    global last_text

    try:
        print("Processing audio...")

        text = whisper_engine.transcribe(audio)

        if not text:
            print("Ignored noise")
            return

        if text == last_text:
            return

        last_text = text

        print("You said:", text)

    except Exception as e:
        print(f"Handler error: {e}")


engine.on_audio_captured = handleAudio
engine.start()