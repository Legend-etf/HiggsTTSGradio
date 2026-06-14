import os
import whisper
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VOICE_DIR = BASE_DIR / "voices"
audio_dir = VOICE_DIR
def transcribe_audio_files(directory: str, model_name: str = "large-v3"):
    model = whisper.load_model(model_name, device="cuda")
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
        
    for filename in os.listdir(directory):
        filename_noext=os.path.splitext(filename)[0]
        filepath = os.path.join(directory, filename)
        txt_path = os.path.join(directory, f"{filename_noext}.txt")
        
        if os.path.isfile(filepath) and os.path.splitext(filename)[1].lower() in audio_extensions :
            print(f"Checking for: {txt_path}")
            if os.path.exists(os.path.join(directory, f"{filename_noext}.txt")):
                print(f"File already exists. Skipping transcription for {filename}.\n")
                continue
            print(f"Transcribing {filename}...")
            result = model.transcribe(filepath)
            with open(os.path.join(directory, f"{filename_noext}.txt"), "w", encoding="utf-8") as file:
                file.write(result["text"].strip())
            print(f"Transcription for {filename} finished..")
            

transcribe_audio_files(audio_dir)
