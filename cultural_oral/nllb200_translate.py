from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Chemins
input_path = Path("datasets/corpus_AHL/filtered/ALH_filtered_all.txt")
output_dir = Path("datasets/corpus_AHL/final")
output_dir.mkdir(parents=True, exist_ok=True)

cr_output_path = output_dir / "ahl_cr.txt"
en_output_path = output_dir / "ahl_en.txt"

# Modèle NLLB
model_id = "facebook/nllb-200-distilled-600M"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

# Langues
tokenizer.src_lang = "hat_Latn"
forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

# Lire uniquement les phrases créoles
creole_sentences = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")

        # Format attendu : audio_source \t segment_id \t creole_text
        if len(parts) >= 3:
            creole_text = parts[2].strip()
            creole_sentences.append(creole_text)

print("Segments to translate:", len(creole_sentences))

# Traduction
BATCH_SIZE = 16

with open(cr_output_path, "w", encoding="utf-8") as cr_file, \
     open(en_output_path, "w", encoding="utf-8") as en_file:

    for start in range(0, len(creole_sentences), BATCH_SIZE):
        batch = creole_sentences[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=4
            )

        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for cr_text, en_text in zip(batch, translations):
            cr_text = cr_text.replace("\n", " ").strip()
            en_text = en_text.replace("\n", " ").strip()

            if cr_text and en_text:
                cr_file.write(cr_text + "\n")
                en_file.write(en_text + "\n")

        print(f"Translated {min(start + BATCH_SIZE, len(creole_sentences))}/{len(creole_sentences)}")

print("Saved:", cr_output_path)
print("Saved:", en_output_path)