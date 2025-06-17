from transformers import pipeline

def transcribe_audio(file_path):
    # Create an automatic speech recognition (ASR) pipeline using Whisper-base
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    
    # Transcribe the audio file
    result = asr(file_path)
    return result["text"]

if __name__ == "__main__":
    # Replace with the full path to your MP3 file
    file_path = "C:\\Users\\evanm\\Downloads\\Review.mp3"
    
    transcription = transcribe_audio(file_path)
    print("Transcription:")
    print(transcription)
