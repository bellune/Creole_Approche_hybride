from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# Paths
# =========================

input_path = Path("datasets/corpus_AHL/filtered/ALH_filtered_all.txt")
reference_path = Path("datasets/corpus_AHL/reference/creole_reference.txt")

output_dir = Path("datasets/corpus_AHL/filtered")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "ALH_filtered_semantic.txt"
removed_path = output_dir / "ALH_removed_semantic.txt"

# =========================
# Parameters
# =========================

KEEP_RATIO = 0.90  # garde les 90% meilleurs segments
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# =========================
# Load data
# =========================

with open(input_path, "r", encoding="utf-8") as f:
    candidate_sentences = [line.strip() for line in f if line.strip()]

with open(reference_path, "r", encoding="utf-8") as f:
    reference_sentences = [line.strip() for line in f if line.strip()]

print("Candidate sentences:", len(candidate_sentences))
print("Reference sentences:", len(reference_sentences))

if len(candidate_sentences) == 0:
    raise ValueError("No candidate sentences found.")

if len(reference_sentences) == 0:
    raise ValueError("No reference sentences found.")

# =========================
# Load embedding model
# =========================

model = SentenceTransformer(MODEL_NAME)

# =========================
# Encode sentences
# =========================

print("Encoding reference sentences...")
ref_emb = model.encode(
    reference_sentences,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Encoding candidate sentences...")
cand_emb = model.encode(
    candidate_sentences,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# =========================
# Compute max similarity efficiently
# =========================

print("Computing similarity scores...")

batch_size = 1024
scores = []

for start in range(0, len(cand_emb), batch_size):
    batch_emb = cand_emb[start:start + batch_size]

    # cosine similarity because embeddings are normalized
    sim_matrix = np.matmul(batch_emb, ref_emb.T)

    # max similarity with any reference sentence
    batch_scores = sim_matrix.max(axis=1)

    scores.extend(batch_scores)

scores = np.array(scores)

# =========================
# Score statistics
# =========================

print("\n===== SCORE DISTRIBUTION =====")
print("Min score:", round(float(scores.min()), 4))
print("Max score:", round(float(scores.max()), 4))
print("Mean score:", round(float(scores.mean()), 4))
print("Median score:", round(float(np.median(scores)), 4))

for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"Percentile {p}:", round(float(np.percentile(scores, p)), 4))

# =========================
# Automatic threshold
# =========================

threshold = np.percentile(scores, (1 - KEEP_RATIO) * 100)

print("\nAutomatic threshold:", round(float(threshold), 4))
print("Keep ratio:", KEEP_RATIO)

# =========================
# Filter
# =========================

kept = []
removed = []

for sent, score in zip(candidate_sentences, scores):
    if score >= threshold:
        kept.append((sent, score))
    else:
        removed.append((sent, score))

# =========================
# Save outputs
# =========================

with open(output_path, "w", encoding="utf-8") as f:
    for sent, score in kept:
        f.write(sent + "\n")

with open(removed_path, "w", encoding="utf-8") as f:
    for sent, score in removed:
        f.write(f"{score:.4f}\t{sent}\n")

# =========================
# Final stats
# =========================

print("\n===== SEMANTIC FILTERING STATS =====")
print("Total before:", len(candidate_sentences))
print("Total after:", len(kept))
print("Total removed:", len(removed))
print("Retention rate:", round((len(kept) / len(candidate_sentences)) * 100, 2), "%")

print("\nSaved filtered file:", output_path)
print("Saved removed file:", removed_path)