# D W V .
> Do With Voice.

A highly modular, voice-driven AI Operating System agent that performs real-world local actions using spoken commands. Designed for speed, privacy, and seamless multi-provider routing.

<br>

### A R C H I T E C T U R E

The system is decoupled into four primary engines:

- **STT Engine**: Hardware-accelerated local transcription.
- **Sound Engine**: Asynchronous capture with dynamic ambient noise calibration and echo cancellation.
- **Action Engine**: JSON-enforced intent parser and terminal command executor.
- **TTS Engine**: High-fidelity local voice synthesis.

<br>

### T O O L S

> **[SoundDevice](https://python-sounddevice.readthedocs.io/en/0.5.3)** 
> *Raw audio stream capture.*
> 
> **[OpenAI Whisper](https://github.com/openai/whisper)** 
> *Local Speech-to-Text inference.*
> 
> **[Coqui TTS](https://github.com/idiap/coqui-ai-TTS)** 
> *Neural Text-to-Speech generation.*
> 
> **[Ollama](https://ollama.com/) / [OpenRouter](https://openrouter.ai/)** 
> *Modular LLM intelligence router.*
> 
> **[OpenWakeWord](https://openwakeword.com/)** 
> *Passive wake-word detection.*

<br>

### S T A C K

`Python 3.11` `Torch` `Sounddevice` `OpenAI` `Ollama` `OpenRouter` `OpenWakeWord`

<br>

### C R E A T O R

[Balapriyan B](https://github.com/BalaPriyan)
