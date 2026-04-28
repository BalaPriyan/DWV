import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav


class SoundEngine:
    def __init__(self,
                 samplerate=16000,
                 channels=1,
                 silence_threshold=0.01,
                 silence_duration=1.0):

        self.samplerate = samplerate
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

        self.audio_buffer = []
        self.is_recording = False
        self.silence_counter = 0

        self.on_audio_captured = None

    def _audio_callback(self, indata, frames, time, status):
        try:
            if status:
                print(f"Audio status warning: {status}")

            if indata is None or len(indata) == 0:
                return

            volume = np.linalg.norm(indata)

            if volume > self.silence_threshold:
                if not self.is_recording:
                    print("Start speaking...")
                    self.is_recording = True
                    self.audio_buffer = []

                self.audio_buffer.append(indata.copy())
                self.silence_counter = 0

            else:
                if self.is_recording:
                    self.silence_counter += frames / self.samplerate
                    self.audio_buffer.append(indata.copy())

                    if self.silence_counter > self.silence_duration:
                        print("Stop recording")

                        if len(self.audio_buffer) == 0:
                            print("Empty buffer, skipping")
                            return

                        full_audio = np.concatenate(self.audio_buffer, axis=0)

                        if self.on_audio_captured:
                            try:
                                self.on_audio_captured(full_audio)
                            except Exception as e:
                                print(f"Handler error: {e}")

                        self.audio_buffer = []
                        self.is_recording = False
                        self.silence_counter = 0

        except Exception as e:
            print(f"Callback error: {e}")
            self._reset_state()

    def _reset_state(self):
        self.audio_buffer = []
        self.is_recording = False
        self.silence_counter = 0

    def start(self):
        print("SoundEngine started...")

        try:
            with sd.InputStream(callback=self._audio_callback,
                                channels=self.channels,
                                samplerate=self.samplerate):
                while True:
                    sd.sleep(1000)

        except Exception as e:
            print(f"Failed to start audio stream: {e}")

    def save_audio(self, audio, filename="output.wav"):
        try:
            if audio is None or len(audio) == 0:
                print("No audio to save")
                return

            wav.write(filename, self.samplerate, audio)
            print(f"Saved: {filename}")

        except Exception as e:
            print(f"Failed to save audio: {e}")