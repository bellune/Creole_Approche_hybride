

import subprocess


def run_cmd(cmd: str):
    print(f"\nRunning:\n{cmd}\n")
    result = subprocess.run(
        cmd,
        shell=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}")
    
    
run_cmd("which fairseq-preprocess")
run_cmd("which fairseq-train")
run_cmd("which fairseq-generate")