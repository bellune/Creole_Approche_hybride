from pathlib import Path

chunks_base_dir = Path("datasets/corpus_AHL/audio/chunks")
transcripts_dir = Path("datasets/corpus_AHL/transcripts_raw")

transcript_files = sorted(transcripts_dir.glob("*_zeeshan_full.txt"))

print("Transcriptions trouvées:", len(transcript_files))

for transcript_file in transcript_files:
    audio_name = transcript_file.name.replace("_zeeshan_full.txt", "")
    chunk_dir = chunks_base_dir / audio_name

    if not chunk_dir.exists():
        print(f"Dossier chunks introuvable pour: {audio_name}")
        continue

    chunks = list(chunk_dir.glob("chunk_*.wav"))

    if len(chunks) == 0:
        print(f"Aucun chunk trouvé pour: {audio_name}")
        continue

    done_file = chunk_dir / ".done"
    done_file.touch()

    print(f".done créé pour {audio_name} ({len(chunks)} chunks)")