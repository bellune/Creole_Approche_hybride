import json
from pathlib import Path

input_files = [
    "datasets/corpus_culturel/code-switching/train/cr_CSS_en.jsonl",
    "datasets/corpus_culturel/code-switching/dev/cr_CSS_en.jsonl",
    "datasets/corpus_culturel/code-switching/test/cr_CSS_en.jsonl"
]

output_base_dir = Path("datasets/corpus_culturel/code-switching/")

def load_json_or_jsonl(path):
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    # Cas 1 : fichier JSON classique sous forme de liste
    if text.startswith("["):
        return json.loads(text)

    # Cas 2 : fichier JSONL, un objet JSON par ligne
    items = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Erreur JSON dans {path}, ligne {line_num}")
            print("Contenu de la ligne:", repr(line[:200]))
            raise e

    return items

for input_file in input_files:
    input_path = Path(input_file)

    split_name = input_path.parent.name  # train, dev, test
    output_dir = output_base_dir / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / input_path.name

    data = load_json_or_jsonl(input_path)

    processed_items = []

    for item in data:
        item = dict(item)

        # Remplacer la source par la phrase code-switchée
        if "src_text" in item:
            item["translation"]["src_text"] = item["src_text"]

        # Supprimer le champ inutile
        item.pop("src_text", None)

        processed_items.append(item)

    # Sauvegarder en JSONL propre
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved:", output_path)
    print("Nombre d'exemples:", len(processed_items))