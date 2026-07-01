import torch

# Désactive le fastpath MHA/Transformer incompatible avec Fairseq ancien + PyTorch récent
if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)

from fairseq_cli.generate import cli_main

if __name__ == "__main__":
    cli_main()
