import re
from pathlib import Path

input_dir = Path("datasets/corpus_AHL/transcripts_raw")
output_dir = Path("datasets/corpus_AHL/segmentation")
output_dir.mkdir(parents=True, exist_ok=True)

MIN_WORDS = 3
MAX_WORDS = 25
MIN_COMMAS_FOR_SPLIT = 2

def normalize_for_dedup(text):
    """
    Normalisation utilisée seulement pour détecter les doublons.
    Elle ne modifie pas le texte final sauvegardé.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # enlève ponctuation
    text = re.sub(r"\s+", " ", text)
    return text

def word_count(text):
    return len(text.split())

def split_long_segment(segment):
    segment = segment.strip()

    if word_count(segment) > MAX_WORDS and segment.count(",") >= MIN_COMMAS_FOR_SPLIT:
        parts = [p.strip() for p in segment.split(",")]
        return [p for p in parts if word_count(p) >= MIN_WORDS]

    return [segment]

def segment_text(text):
    # Retirer les noms de chunks : [chunk_0000.wav]
    text = re.sub(r"\[chunk_\d+\.wav\]", " ", text)

    # Nettoyage léger
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Séparation principale sur . ? !
    first_segments = re.split(r"(?<=[.!?])\s+", text)

    final_segments = []

    for seg in first_segments:
        seg = seg.strip()

        if word_count(seg) < MIN_WORDS:
            continue

        subsegments = split_long_segment(seg)

        for sub in subsegments:
            sub = sub.strip()

            if word_count(sub) >= MIN_WORDS:
                final_segments.append(sub)

    return final_segments

for txt_file in input_dir.glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    segments = segment_text(text)

    seen = set()
    unique_segments = []

    for seg in segments:
        key = normalize_for_dedup(seg)

        if key not in seen:
            seen.add(key)
            unique_segments.append(seg)

    output_file = output_dir / f"{txt_file.stem}_segments.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(unique_segments, start=1):
            f.write(f"{seg}\n")

    print(
        f"Saved: {output_file} | "
        f"{len(segments)} segments before dedup | "
        f"{len(unique_segments)} after dedup"
    )