from pathlib import Path
import librosa

audio_dir = Path("datasets/corpus_AHL/audio/wav")

total_seconds = 0
durations = []

for audio_file in sorted(audio_dir.glob("*.wav")):
    duration = librosa.get_duration(path=str(audio_file))
    durations.append((audio_file.name, duration))
    total_seconds += duration

print("Nombre d'audios:", len(durations))
print("Durée totale en heures:", round(total_seconds / 3600, 2))
print("Nombre estimé de chunks de 20s:", round(total_seconds / 20))

for name, duration in durations:
    print(name, ":", round(duration / 60, 2), "minutes")