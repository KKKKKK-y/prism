from __future__ import annotations

import argparse
import ast
import math
import struct
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


@dataclass
class StreamingStats:
    bins: int = 20_000
    low: float = 0.0
    high: float = 1.0
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    gt_03: int = 0
    gt_05: int = 0
    gt_07: int = 0
    hist: np.ndarray = field(default_factory=lambda: np.zeros(20_000, dtype=np.int64))

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        flat = values.ravel()
        self.count += int(flat.size)
        self.total += float(flat.sum())
        self.total_sq += float(np.square(flat).sum())
        self.min_value = min(self.min_value, float(flat.min()))
        self.max_value = max(self.max_value, float(flat.max()))
        self.gt_03 += int((flat > 0.3).sum())
        self.gt_05 += int((flat > 0.5).sum())
        self.gt_07 += int((flat > 0.7).sum())
        clipped = np.clip(flat, self.low, self.high)
        counts, _ = np.histogram(clipped, bins=self.bins, range=(self.low, self.high))
        self.hist += counts.astype(np.int64, copy=False)

    def percentile(self, q: float) -> float:
        if self.count <= 0:
            return float("nan")
        target = q / 100.0 * max(0, self.count - 1)
        cumsum = np.cumsum(self.hist)
        idx = int(np.searchsorted(cumsum, target, side="left"))
        idx = max(0, min(self.bins - 1, idx))
        return self.low + (idx + 0.5) * (self.high - self.low) / self.bins

    def summary(self) -> dict[str, float]:
        if self.count <= 0:
            return {key: float("nan") for key in ("min", "max", "mean", "std", "p50", "p75", "p90", "p95", "p99")}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": float(self.count),
            "min": self.min_value,
            "max": self.max_value,
            "mean": mean,
            "std": math.sqrt(variance),
            "p50": self.percentile(50),
            "p75": self.percentile(75),
            "p90": self.percentile(90),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "frac_gt_03": self.gt_03 / self.count,
            "frac_gt_05": self.gt_05 / self.count,
            "frac_gt_07": self.gt_07 / self.count,
        }


def parse_npy_header(handle: BinaryIO) -> tuple[np.dtype, tuple[int, ...], bool]:
    magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("NPZ member is not a .npy array")
    major, minor = struct.unpack("BB", handle.read(2))
    if major == 1:
        header_len = struct.unpack("<H", handle.read(2))[0]
    elif major in (2, 3):
        header_len = struct.unpack("<I", handle.read(4))[0]
    else:
        raise ValueError(f"Unsupported .npy version {major}.{minor}")
    header = handle.read(header_len)
    text = header.decode("latin1" if major < 3 else "utf-8")
    info = ast.literal_eval(text)
    dtype = np.dtype(info["descr"])
    shape = tuple(int(x) for x in info["shape"])
    fortran_order = bool(info["fortran_order"])
    if fortran_order:
        raise ValueError("Fortran-order arrays are not supported by this streaming inspector")
    return dtype, shape, fortran_order


def resolve_npz_paths(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.name.endswith("_train_hard.npz"):
        candidates = [
            path,
            path.with_name(path.name.replace("_train_hard.npz", "_val_hard.npz")),
            path.with_name(path.name.replace("_train_hard.npz", "_test_hard.npz")),
        ]
        return [candidate for candidate in candidates if candidate.exists()]
    return [path]


def inspect_member(npz_path: Path, member: str, horizon_stats: bool = False) -> tuple[dict[str, float], tuple[int, ...], list[dict[str, float]]]:
    global_stats = StreamingStats()
    horizon_summaries: list[StreamingStats] = []
    shape: tuple[int, ...] = ()
    with zipfile.ZipFile(npz_path, "r") as zf:
        with zf.open(member) as handle:
            dtype, shape, _ = parse_npy_header(handle)
            if horizon_stats:
                if len(shape) != 5:
                    raise ValueError(f"target array must be [N,H,1,H,W], got {shape}")
                horizon_summaries = [StreamingStats() for _ in range(shape[1])]
                sample_size = int(np.prod(shape[1:]))
                samples_per_block = max(1, min(64, 16 * 1024 * 1024 // max(1, sample_size * dtype.itemsize)))
                block_values = samples_per_block * sample_size
            else:
                block_values = max(1, 16 * 1024 * 1024 // dtype.itemsize)

            leftover = b""
            sample_values = int(np.prod(shape[1:])) if horizon_stats else 0
            while True:
                raw = handle.read(block_values * dtype.itemsize)
                if not raw:
                    break
                raw = leftover + raw
                usable = len(raw) // dtype.itemsize * dtype.itemsize
                data = np.frombuffer(raw[:usable], dtype=dtype)
                leftover = raw[usable:]
                if data.size == 0:
                    continue
                if horizon_stats:
                    sample_usable = data.size // sample_values * sample_values
                    if sample_usable <= 0:
                        leftover = raw[:usable] + leftover
                        continue
                    block = data[:sample_usable].reshape(-1, shape[1], shape[2], shape[3], shape[4])
                    global_stats.update(block)
                    for horizon_idx, stats in enumerate(horizon_summaries):
                        stats.update(block[:, horizon_idx])
                    remainder = data[sample_usable:].tobytes()
                    leftover = remainder + leftover
                else:
                    global_stats.update(data)

            if leftover:
                data = np.frombuffer(leftover, dtype=dtype)
                if data.size:
                    global_stats.update(data)

    return global_stats.summary(), shape, [stats.summary() for stats in horizon_summaries]


def format_summary(name: str, shape: tuple[int, ...], stats: dict[str, float]) -> list[str]:
    lines = [f"{name}_shape: {shape}"]
    for key in ("min", "max", "mean", "std", "p50", "p75", "p90", "p95", "p99", "frac_gt_03", "frac_gt_05", "frac_gt_07"):
        lines.append(f"{name}_{key}: {stats[key]:.8f}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect hard NPZ obs/target risk distribution without loading all arrays.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/results/stage5_5_hard_target_distribution.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_npz_paths(args.npz)
    if not paths:
        raise FileNotFoundError(f"No NPZ files found for {args.npz}")

    lines = ["PRISM Stage-5.5 Hard Target Distribution", "=========================================", ""]
    for npz_path in paths:
        print(f"Inspecting {npz_path}")
        lines.append(f"File: {npz_path}")
        obs_stats, obs_shape, _ = inspect_member(npz_path, "obs.npy", horizon_stats=False)
        target_stats, target_shape, horizon_stats = inspect_member(npz_path, "target.npy", horizon_stats=True)
        lines.extend(format_summary("obs", obs_shape, obs_stats))
        lines.extend(format_summary("target", target_shape, target_stats))
        lines.append("target_horizon_summary:")
        for idx, stats in enumerate(horizon_stats, start=1):
            lines.append(
                f"h{idx} target_mean={stats['mean']:.8f} target_p95={stats['p95']:.8f} "
                f"target_max={stats['max']:.8f} target_frac_gt_05={stats['frac_gt_05']:.8f}"
            )
        lines.append("")

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved hard target distribution summary to: {output}")


if __name__ == "__main__":
    main()
