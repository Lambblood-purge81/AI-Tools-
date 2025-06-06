from gtts import gTTS
import os

def play_audio(text, output_file="response.wav"):
    """Convert text to speech and save to file"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_file)
        return True
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return False