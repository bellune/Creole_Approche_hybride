from datasets import load_from_disk
import config as cfg

ds = load_from_disk(cfg.save_path)
print(ds)

train_ds = ds["train"]
val_ds   = ds["validation"]
test_ds  = ds["test"]

print(train_ds[0])
