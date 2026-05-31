import os
import subprocess
from pathlib import Path

input_dir = Path("datasets/corpus_AHL/audio/")
output_dir = Path("datasets/corpus_AHL/audio/wav")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

#if wav files already exist, skip conversion
existing_wavs = set(output_dir.glob("*.wav"))

# Convert all MP3 files to WAV
for file in Path(input_dir).glob("*.mp3"):
    filename = file.stem
    output_file = output_dir / f"{filename}.wav"

    if output_file in existing_wavs:
        print(f"WAV file already exists, skipping: {output_file}")
        continue
    subprocess.run([
        "ffmpeg", "-i", str(file), 
        "-ac", "1", "-ar", "16000", 
        str(output_file)
    ])