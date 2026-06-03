from __future__ import annotations

import argparse

import torch

from prism.trainers.trainer import select_device


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Print PRISM runtime environment information.").parse_args()


def main() -> None:
    parse_args()
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"mps_available: {torch.backends.mps.is_available()}")
    print(f"selected_device: {select_device()}")


if __name__ == "__main__":
    main()
