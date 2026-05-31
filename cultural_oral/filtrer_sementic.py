from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
input_path = Path("datasets/corpus_AHL/filtered/ALH_filtered_all.txt")
reference_path = Path("datasets/corpus_AHL/reference/creole_reference.txt")

output_path = Path("datasets/corpus_AHL/filtered/ALH_filtered_semantic.txt")
removed_path = Path("datasets/corpus_AHL/filtered/ALH_removed_semantic.txt")

# Load data
with open(input_path, "r", encoding="utf-8") as f:
    candidate_sentences = [line.strip() for line in f if line.strip()]

with open(reference_path, "r", encoding="utf-8") as f:
    reference_sentences = [line.strip() for line in f if line.strip()]

print("Candidate sentences:", len(candidate_sentences))
print("Reference sentences:", len(reference_sentences))

# Multilingual embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Encode references
ref_emb = model.encode(
    reference_sentences,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Encode candidates
cand_emb = model.encode(
    candidate_sentences,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Similarity: each candidate vs all references
scores = cosine_similarity(cand_emb, ref_emb).max(axis=1)

# Threshold
# Commence avec 0.35. Ajuste après inspection.
THRESHOLD = 0.35

kept = []
removed = []

for sent, score in zip(candidate_sentences, scores):
    if score >= THRESHOLD:
        kept.append((sent, score))
    else:
        removed.append((sent, score))

with open(output_path, "w", encoding="utf-8") as f:
    for sent, score in kept:
        f.write(sent + "\n")

with open(removed_path, "w", encoding="utf-8") as f:
    for sent, score in removed:
        f.write(f"{score:.4f}\t{sent}\n")

print("Saved:", output_path)
print("Removed saved:", removed_path)

print("\n===== SEMANTIC FILTERING STATS =====")
print("Total before:", len(candidate_sentences))
print("Total after:", len(kept))
print("Total removed:", len(removed))
print("Retention rate:", round(len(kept) / len(candidate_sentences) * 100, 2), "%")