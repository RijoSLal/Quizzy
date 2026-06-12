from kittentts import KittenTTS
import whisper
import io
import soundfile as sf
import logging
import tempfile
from asgiref.sync import sync_to_async

import torch

logger = logging.getLogger("django")

class STTGenerator:
    """
    Handles speech-to-text transcription using Whisper.
    """
    def __init__(self, model_size="tiny"):
        self._model = None
        self.model_size = model_size
        self.device = "cpu" # Force CPU to save VRAM for other models

    def get_model(self):
        """Lazy loader for Whisper model."""
        if self._model is None:
            logger.info(f"Initializing Whisper ({self.model_size}) on {self.device}...")
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    async def transcribe(self, audio_source, language: str = "en") -> dict:
        """
        Transcribes audio from a file path or an UploadedFile object.
        """
        model = self.get_model()
        
        # Determine if FP16 is supported (only on GPU)
        fp16_supported = self.device != "cpu"

        # If it's an uploaded file (has a 'read' method), handle it via a temporary file
        if hasattr(audio_source, 'read'):
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio:
                # Use chunks to avoid memory issues with large files
                for chunk in audio_source.chunks():
                    temp_audio.write(chunk)
                temp_audio.flush()
                return await sync_to_async(model.transcribe)(temp_audio.name, language=language, fp16=fp16_supported)
        
        # Otherwise, assume it's a file path or numpy array
        return await sync_to_async(model.transcribe)(audio_source, language=language, fp16=fp16_supported)

class SpeechGenerator:
    """
    Handles text-to-speech generation using KittenTTS.
    """
    def __init__(self):
        self._model = None

    def get_model(self):
        """Lazy loader for KittenTTS model."""
        if self._model is None:
            logger.info("Initializing KittenTTS (15M Nano)...")
            self._model = KittenTTS("KittenML/kitten-tts-nano-0.8")
        return self._model

    async def text_to_speech(self, text: str, voice: str|None = None, rate:int=0, pitch:int=0) -> bytes:
        """
        Converts text into speech using Kitten TTS (15M Nano).

        Args:
            text (str): The text to be converted into speech.
            voice (str | None, optional): The voice to use ('male' or 'female', defaults to 'female').
            rate (int, optional): Speech speed adjustment in percentage (default is 0).
            pitch (int, optional): Pitch adjustment (currently not supported by KittenTTS, kept for compatibility).

        Returns:
            bytes: The generated speech audio as a byte stream (WAV).
        """
        try:
            voice_options = {
                "female": "Bella",
                "male": "Jasper",
            }

            selected_voice = voice_options.get(voice.lower() if voice else "male", "Jasper")
            
            # Convert rate percentage to speed multiplier (e.g. 10 -> 1.1)
            speed = 1.17 + (rate / 100.0)

            model = self.get_model()
            
            # KittenTTS generate returns a numpy array.
            # We run it via sync_to_async because it's a CPU-bound operation.
            audio = await sync_to_async(model.generate)(text, voice=selected_voice, speed=speed, clean_text=True)

            audio_buffer = io.BytesIO()
            # KittenTTS defaults to 24000 Hz sample rate
            sf.write(audio_buffer, audio, 24000, format='WAV')

            audio_buffer.seek(0)
            logger.info("KittenTTS speech generation successful")
            return audio_buffer.getvalue() 
        
        except Exception as e:
            logger.error(f"KittenTTS faced an error: {str(e)}", exc_info=True)
            return b""
