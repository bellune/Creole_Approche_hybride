from pathlib import Path
import shutil

source_dir = Path("datasets/corpus_AHL/audio/wav")
target_dir = Path("datasets/corpus_AHL/audio/selected_wav")
target_dir.mkdir(parents=True, exist_ok=True)

audio_files = sorted(source_dir.glob("*.wav"))

copied = 0
skipped = 0

for src in audio_files:
    dst = target_dir / src.name

    if dst.exists():
        print("Already exists, skipped:", src.name)
        skipped += 1
        continue

    shutil.copy2(src, dst)
    print("Copied:", src.name)
    copied += 1

print("Total copied:", copied)
print("Total skipped:", skipped)