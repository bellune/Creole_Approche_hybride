from pathlib import Path

chunks_base_dir = Path("datasets/corpus_AHL/audio/chunks")

done_files = list(chunks_base_dir.glob("*/.done"))

print("Audios marqués comme découpés:", len(done_files))

for f in done_files[:20]:
    print(f.parent.name)