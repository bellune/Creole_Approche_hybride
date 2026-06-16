
from pathlib import Path


data_fairseq = "datasets/kreyol-mt-hat-eng/mt-fairseq"

for file in Path(data_fairseq).glob("*"):
    with open(file, encoding="utf-8") as f:
        print(file, sum(1 for _ in f))