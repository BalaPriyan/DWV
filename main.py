from engine.soundEngine import SoundEngine

engine = SoundEngine()

def handleAudio(audio):
    print("Audio: " , audio.shape)
    engine.save_audio(audio)

engine.on_audio_captured = handleAudio

engine.start()

