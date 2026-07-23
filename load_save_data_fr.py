from datasets import load_dataset
import config as cfg
# Charger le split complet
ds = load_dataset("jhu-clsp/kreyol-mt", "hat-fra", download_mode="force_redownload")

print(ds)
ds.save_to_disk(cfg.save_path_fr)

