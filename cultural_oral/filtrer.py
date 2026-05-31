import re
from pathlib import Path

# Dossiers
input_dir = Path("datasets/corpus_AHL/segmentation")
output_dir = Path("datasets/corpus_AHL/filtered")
output_dir.mkdir(parents=True, exist_ok=True)

# Fichiers de sortie globaux
filtered_output = output_dir / "ALH_filtered_all.txt"
removed_output = output_dir / "ALH_removed_all.txt"

# Paramètres
MIN_WORDS = 3
MAX_WORDS = 25

BAD_CHARS_PATTERN = r"[�💁牙要ى谢谢спасибоСпасиб]"
ALLOWED_PATTERN = r"[^a-zA-ZÀ-ÿ0-9\s.,?!'’\-:;]"

def is_valid_segment(text):
    text = text.strip()

    if not text:
        return False, "empty"

    words = text.split()
    n_words = len(words)

    if n_words < MIN_WORDS:
        return False, "too_short"

    if n_words > MAX_WORDS:
        return False, "too_long"

    if re.search(BAD_CHARS_PATTERN, text):
        return False, "bad_characters"

    abnormal_symbols = re.findall(ALLOWED_PATTERN, text)
    if len(abnormal_symbols) > 2:
        return False, "too_many_symbols"

    letters = re.findall(r"[a-zA-ZÀ-ÿ]", text)
    if len(letters) < 5:
        return False, "too_few_letters"

    lower_words = [w.lower().strip(".,?!'’:-;") for w in words]
    if len(lower_words) >= 5:
        most_common_count = max(lower_words.count(w) for w in set(lower_words))
        if most_common_count / len(lower_words) > 0.5:
            return False, "repetition"

    return True, "valid"


total_before = 0
total_after = 0
reason_counts = {}

with open(filtered_output, "w", encoding="utf-8") as fout, \
     open(removed_output, "w", encoding="utf-8") as rout:

    for file_path in sorted(input_dir.glob("*.txt")):
        audio_source = file_path.stem.replace("_segments", "")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            total_before += 1

            # Format attendu : segment_id \t texte
            if "\t" in line:
                segment_id, text = line.split("\t", 1)
            else:
                segment_id = "unknown"
                text = line

            valid, reason = is_valid_segment(text)

            if valid:
                fout.write(f"{text}\n")
                total_after += 1
            else:
                rout.write(f"{audio_source}\t{segment_id}\t{reason}\t{text}\n")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

print("Filtered file saved:", filtered_output)
print("Removed file saved:", removed_output)

print("\n===== GLOBAL STATS =====")
print("Total before:", total_before)
print("Total after:", total_after)
print("Total removed:", total_before - total_after)

if total_before > 0:
    print("Retention rate:", round((total_after / total_before) * 100, 2), "%")

print("\nRemoval reasons:")
for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
    print(reason, ":", count)