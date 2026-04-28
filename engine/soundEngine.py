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

                    full_audio = np.concatenate(self.audio_buffer, axis=0)

                    if self.on_audio_captured:
                        self.on_audio_captured(full_audio)

                    # reset
                    self.audio_buffer = []
                    self.is_recording = False
                    self.silence_counter = 0

    def start(self):
        print("SoundEngine started...")

        with sd.InputStream(callback=self._audio_callback,
                            channels=self.channels,
                            samplerate=self.samplerate):
            while True:
                sd.sleep(1000)


    def save_audio(self, audio, filename="output.wav"):
        wav.write(filename, self.samplerate, audio)
        print(f"Saved: {filename}")