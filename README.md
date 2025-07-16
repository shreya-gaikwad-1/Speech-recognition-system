# Speech-recognition-system

A simple Python-based Speech-to-Text tool using:
- ✅ Google Speech Recognition API (online)
- 🔒 Wav2Vec2.0 from Facebook (offline via Hugging Face)

Requirements: pip install SpeechRecognition torch transformers librosa pyaudio

For Windows (if PyAudio fails):
pip install pipwin
pipwin install pyaudio

Usage
1. Record audio (5 seconds): python record_audio.py
2. Transcribe audio: python main.py
Transcribes test_audio.wav using both methods.

Files
- record_audio.py – Records microphone input
- main.py – Transcribes using Google or Wav2Vec
  
Notes
- Wav2Vec model: facebook/wav2vec2-base-960h
- Recommended: .wav files, 16kHz sample rate
