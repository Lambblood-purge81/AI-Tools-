import sounddevice as sd
import soundfile as sf
import queue
import json
import os
import numpy as np
from vosk import Model, KaldiRecognizer

# Audio recorder class for continuous recording
class AudioRecorder:
    def __init__(self, samplerate=16000, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.recording = None
        self.stream = None
        self.q = queue.Queue()
        
        # Initialize Vosk model
        model_path = r"D:\Smart IT _ support chatbot\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, samplerate)
    
    def callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        self.q.put(indata.copy())
    
    def start(self):
        """Start audio recording"""
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self.callback,
            dtype='float32'
        )
        self.stream.start()
    
    def stop(self):
        """Stop audio recording"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def get_audio(self):
        """Get accumulated audio data"""
        if not self.q.empty():
            return np.concatenate(list(self.q.queue))
        return None

def transcribe_audio(recorder):
    """Transcribe audio from recorder"""
    audio_data = recorder.get_audio()
    if audio_data is not None:
        # Convert to 16-bit PCM for Vosk
        audio_data = (audio_data * 32767).astype(np.int16)
        if recorder.recognizer.AcceptWaveform(audio_data.tobytes()):
            result = json.loads(recorder.recognizer.Result())
            return result.get("text", "")
    return None

def record_audio(filename=None, duration=5):
    """Record audio for a fixed duration"""
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    
    if filename:
        sf.write(filename, recording, fs)
    return recording