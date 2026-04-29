import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import queue
import logging

logger = logging.getLogger(__name__)

class SoundEngine:
    def __init__(self,
                 samplerate=16000,
                 channels=1,
                 silence_duration=1.0):

        self.samplerate = samplerate
        self.channels = channels
        self.silence_threshold = 0.01 # Will be calibrated
        self.silence_duration = silence_duration

        self.audio_buffer = []
        self.is_recording = False
        self.silence_counter = 0

        self.audio_queue = queue.Queue()
        self.is_running = False

    def calibrate(self, duration=2.0):
        logger.info(f"Calibrating ambient noise for {duration} seconds... Please stay quiet.")
        volumes = []
        
        def calibration_callback(indata, frames, time, status):
            if indata is not None and len(indata) > 0:
                volumes.append(np.linalg.norm(indata))
                
        with sd.InputStream(callback=calibration_callback, channels=self.channels, samplerate=self.samplerate):
            sd.sleep(int(duration * 1000))
            
        if volumes:
            # Set threshold slightly above the 95th percentile of ambient noise
            avg_noise = np.percentile(volumes, 95)
            self.silence_threshold = max(0.005, avg_noise * 1.5)
            logger.info(f"Calibration complete. Noise threshold set to: {self.silence_threshold:.4f}")
        else:
            logger.warning("Calibration failed, using default threshold.")
            self.silence_threshold = 0.01

    def _audio_callback(self, indata, frames, time, status):
        try:
            if status:
                logger.warning(f"Audio status warning: {status}")

            if indata is None or len(indata) == 0:
                return

            volume = np.linalg.norm(indata)

            if volume > self.silence_threshold:
                if not self.is_recording:
                    self.is_recording = True
                    self.audio_buffer = []

                self.audio_buffer.append(indata.copy())
                self.silence_counter = 0

            else:
                if self.is_recording:
                    self.silence_counter += frames / self.samplerate
                    self.audio_buffer.append(indata.copy())

                    if self.silence_counter > self.silence_duration:
                        if len(self.audio_buffer) > 0:
                            full_audio = np.concatenate(self.audio_buffer, axis=0)
                            # Put the audio block into the queue for the consumer
                            self.audio_queue.put(full_audio)

                        self.audio_buffer = []
                        self.is_recording = False
                        self.silence_counter = 0

        except Exception as e:
            logger.error(f"Callback error: {e}")
            self._reset_state()

    def _reset_state(self):
        self.audio_buffer = []
        self.is_recording = False
        self.silence_counter = 0

    def start(self):
        self.calibrate()
        logger.info("SoundEngine started listening...")
        
        self.is_running = True
        self.stream = sd.InputStream(callback=self._audio_callback,
                                     channels=self.channels,
                                     samplerate=self.samplerate)
        self.stream.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        logger.info("SoundEngine stopped.")

    def get_audio(self):
        """Generator that yields audio segments as they are recorded."""
        while self.is_running:
            try:
                # Block for a short time to allow checking is_running
                audio = self.audio_queue.get(timeout=0.5)
                yield audio
            except queue.Empty:
                continue