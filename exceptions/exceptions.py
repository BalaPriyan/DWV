class DWVError(Exception):
    """Base class for exceptions in this project."""
    pass

class AudioError(DWVError):
    """Exception raised for errors in the audio engine."""
    pass

class TranscriptionError(DWVError):
    """Exception raised for errors during speech transcription."""
    pass

class LLMError(DWVError):
    """Exception raised for errors in LLM communication."""
    pass

class TTSError(DWVError):
    """Exception raised for errors during text-to-speech synthesis."""
    pass

class CommandExecutionError(DWVError):
    """Exception raised when a system command fails to execute."""
    pass
