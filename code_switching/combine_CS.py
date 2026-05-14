import json

original_file =  ["datasets/corpus_culturel/train/cr_en.jsonl",
                "datasets/corpus_culturel/dev/cr_en.jsonl",
                "datasets/corpus_culturel/test/cr_en.jsonl" ]

token_file = ["datasets/corpus_culturel/code-switching/train/cr_cs_en.jsonl",
               "datasets/corpus_culturel/code-switching/dev/cr_cs_en.jsonl",
                 "datasets/corpus_culturel/code-switching/test/cr_cs_en.jsonl"]
# sentence_file = "datasets/corpus_culturel/code-switching/CS_Sentence_cr_en.jsonl"

OUTPUT_FILE = ["datasets/corpus_culturel/code-switching/train/cr_en.jsonl"]

from datasets import load_dataset

   
for original_file, token_file, OUTPUT_FILE in zip(original_file, token_file, OUTPUT_FILE):  
   
    combined_items = []
    combined_data = []

    # =========================
    # LOAD TOKEN CS FILES
    # =========================

    data = load_dataset(
        "json",
        data_files={"mydata": token_file}
    )["mydata"]

    for item in data:
        item = dict(item)

        item["id"] = str(item["id"]) + "_token"

        # remplacer la source par la phrase code-switchée
        item["translation"]["src_text"] = item["src_text"]

        # supprimer le champ inutile
        item.pop("src_text", None)

        combined_items.append(item)

    # =========================
    # LOAD ORIGINAL DATA
    # =========================

    original_data = load_dataset(
        "json",
        data_files={"mydata": original_file}
    )["mydata"]

    for item in original_data:
        combined_data.append(dict(item))

    # =========================
    # ADD CODE-SWITCHED DATA
    # =========================

    combined_data.extend(combined_items)

    # =========================
    # SAVE FINAL DATASET
    # =========================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            combined_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Dataset combiné sauvegardé :", OUTPUT_FILE)
    print("Nombre total d'exemples :", len(combined_data))