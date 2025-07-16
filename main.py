import speech_recognition as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_audio(audio_path, method="google", wav2vec_model="facebook/wav2vec2-base-960h"):
    """
    Transcribe audio using either Google's API or Wav2Vec 2.0
    
    Args:
        audio_path: Path to audio file (WAV format recommended)
        method: 'google' for online API or 'wav2vec' for offline model
        wav2vec_model: Hugging Face model identifier for Wav2Vec
    
    Returns:
        Transcribed text string
    """
    if method == "google":
        return transcribe_google(audio_path)
    elif method == "wav2vec":
        return transcribe_wav2vec(audio_path, wav2vec_model)
    else:
        raise ValueError("Invalid method. Choose 'google' or 'wav2vec'")

def transcribe_google(audio_path):
    """Transcribe using Google's Speech Recognition API"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        logger.error("Google API could not understand audio")
    except sr.RequestError as e:
        logger.error(f"Google API request error: {e}")
    except Exception as e:
        logger.error(f"Google processing error: {e}")
    return ""

def transcribe_wav2vec(audio_path, model_name):
    """Transcribe using Wav2Vec 2.0 model"""
    try:
        # Load and preprocess audio
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        # Load model and processor
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        # Process audio
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        
        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0].lower()
    
    except Exception as e:
        logger.error(f"Wav2Vec processing error: {e}")
        return ""

if __name__ == "__main__":
    # Example usage
    audio_file = "test_audio.wav"  # Replace with your audio file
    
    print("Google API Transcription:")
    print(transcribe_audio(audio_file, method="google"))
    
    print("\nWav2Vec Transcription:")
    print(transcribe_audio(audio_file, method="wav2vec", 
                          wav2vec_model="facebook/wav2vec2-base-960h"))