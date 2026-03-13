from datasets import load_dataset
import config as cfg
# Charger le split complet
ds = load_dataset("jhu-clsp/kreyol-mt", "hat-eng")

print(ds)
ds.save_to_disk(cfg.save_path)

