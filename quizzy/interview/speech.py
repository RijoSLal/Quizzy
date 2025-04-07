import edge_tts # type: ignore
from io import BytesIO
import logging
logger = logging.getLogger("django")

async def text_to_speech(text: str, voice: str|None = None, rate:int=0, pitch:int=0) -> bytes:
        """
        Converts text into speech using Edge TTS.

        Args:
            text (str): The text to be converted into speech.
            voice (str | None, optional): The voice to use ('male' or 'female', defaults to 'female').
            rate (int, optional): Speech speed adjustment in percentage (default is 0).
            pitch (int, optional): Pitch adjustment in Hz (default is 0).

        Returns:
            bytes: The generated speech audio as a byte stream.
        """
        try:
            voice_options = {
                "female": "en-US-AvaNeural - en-US (Female)",
                "male": "en-US-AndrewNeural - en-US (Male)",
            }

            # get voice from dictionary default to female if not found
            voice = voice_options.get(voice.lower() if voice else "female", voice_options["female"])

            voice_short_name = voice.split(" - ")[0]
            communicate = edge_tts.Communicate(text, voice_short_name, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz")

            audio_buffer = BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

            audio_buffer.seek(0)
            logger.info(f"speech generation successful")
            return audio_buffer.getvalue() 
        
        except Exception as e:
            logger.error(f"edge_tts faced a major error: {str(e)}",exc_info=True)
             

