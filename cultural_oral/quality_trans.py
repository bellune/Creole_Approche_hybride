from pathlib import Path
from openai import OpenAI
import time

# ======================
# Paths
# ======================

cr_path = Path("datasets/corpus_AHL/final/ahl_llm_cr.txt")
en_path = Path("datasets/corpus_AHL/final/ahl_llm_en.txt")

output_dir = Path("datasets/corpus_AHL/final_quality")
output_dir.mkdir(parents=True, exist_ok=True)

quality_path = output_dir / "ahl_llm_quality.tsv"
clean_cr_path = output_dir / "ahl_cr.txt"
clean_en_path = output_dir / "ahl_en.txt"

# ======================
# Load data
# ======================

with open(cr_path, "r", encoding="utf-8") as f:
    cr_lines = [line.strip() for line in f if line.strip()]

with open(en_path, "r", encoding="utf-8") as f:
    en_lines = [line.strip() for line in f if line.strip()]

assert len(cr_lines) == len(en_lines), "CR and EN files are not aligned."

pairs = list(zip(cr_lines, en_lines))

print("Total pairs:", len(pairs))

# ======================
# Optional: test on sample first
# ======================

# Pour tester sur 50 paires seulement au début
# pairs = pairs[:50]

# ======================
# LLM client
# ======================

client = OpenAI()  # nécessite OPENAI_API_KEY dans ton environnement

MODEL_NAME = "gpt-5.4-mini"

def judge_translation(cr_text, en_text):
    prompt = f"""
You are evaluating Haitian Creole to English translations.

Classify the English translation according to how well it preserves the meaning of the Haitian Creole source.

Labels:
good = the translation preserves the meaning well.
medium = the translation preserves the general meaning but has minor errors or awkwardness.
bad = the translation is wrong, incomplete, unrelated, or hallucinates information.

Return only one label: good, medium, or bad.

Haitian Creole:
{cr_text}

English translation:
{en_text}

Label:
""".strip()

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0
    )

    label = response.output_text.strip().lower()

    if "good" in label:
        return "good"
    elif "medium" in label:
        return "medium"
    elif "bad" in label:
        return "bad"
    else:
        return "unknown"

# ======================
# Evaluate pairs
# ======================

counts = {"good": 0, "medium": 0, "bad": 0, "unknown": 0}

with open(quality_path, "w", encoding="utf-8") as qout, \
     open(clean_cr_path, "w", encoding="utf-8") as crout, \
     open(clean_en_path, "w", encoding="utf-8") as enout:

    qout.write("id\tlabel\tcr\ten\n")

    for i, (cr_text, en_text) in enumerate(pairs, start=1):
        try:
            label = judge_translation(cr_text, en_text)
        except Exception as e:
            print(f"Error at pair {i}: {e}")
            label = "unknown"
            time.sleep(2)

        counts[label] = counts.get(label, 0) + 1

        qout.write(f"{i}\t{label}\t{cr_text}\t{en_text}\n")

        # Garder good + medium
        if label in ["good", "medium"]:
            crout.write(cr_text + "\n")
            enout.write(en_text + "\n")

        if i % 20 == 0:
            print(f"Processed {i}/{len(pairs)} | counts:", counts)

print("Saved:", quality_path)
print("Saved:", clean_cr_path)
print("Saved:", clean_en_path)
print("Final counts:", counts)