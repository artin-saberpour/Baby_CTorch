#!/usr/bin/env python3
"""Compare babyCTorch and PyTorch DDPM trainability results."""

from __future__ import annotations

import csv
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
GATE_RATIO = 0.65


def read_losses(path: Path) -> list[float]:
    with path.open(newline="") as f:
        return [float(row["loss"]) for row in csv.DictReader(f)]


def loss_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("empty loss trace")
    window = min(100, max(1, len(values) // 4))
    initial = sum(values[:window]) / window
    final = sum(values[-window:]) / window
    return initial, final, final / initial


def read_samples(path: Path) -> list[list[float]]:
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return [[float(row[f"p{i}"]) for i in range(16)] for row in rows]


def prototypes() -> list[list[float]]:
    result: list[list[float]] = []
    for col in (1, 2):
        image = [-1.0] * 16
        for row in range(4):
            image[row * 4 + col] = 1.0
        result.append(image)
    for row in (1, 2):
        image = [-1.0] * 16
        for col in range(4):
            image[row * 4 + col] = 1.0
        result.append(image)
    return result


def nearest_prototype_mse(samples: list[list[float]]) -> float:
    refs = prototypes()
    per_sample = []
    for sample in samples:
        best = min(
            sum((a - b) ** 2 for a, b in zip(sample, ref)) / 16.0
            for ref in refs
        )
        per_sample.append(best)
    return sum(per_sample) / len(per_sample)


def maybe_plot(baby: list[float], torch_losses: list[float]) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "loss_plot=SKIPPED (matplotlib unavailable)"

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(baby, label="babyCTorch", linewidth=1.2)
    plt.plot(torch_losses, label="PyTorch", linewidth=1.2, alpha=0.8)
    plt.xlabel("training step")
    plt.ylabel("noise-prediction MSE")
    plt.title("Tiny DDPM trainability")
    plt.legend()
    plt.tight_layout()
    out = RESULTS / "loss_comparison.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return f"loss_plot={out.name}"


def main() -> int:
    baby_loss_path = RESULTS / "babyctorch_loss.csv"
    torch_loss_path = RESULTS / "pytorch_loss.csv"
    baby_samples_path = RESULTS / "babyctorch_samples.csv"
    torch_samples_path = RESULTS / "pytorch_samples.csv"

    required = [
        baby_loss_path,
        torch_loss_path,
        baby_samples_path,
        torch_samples_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing result files:")
        for path in missing:
            print(f"  {path}")
        return 2

    baby = read_losses(baby_loss_path)
    torch_losses = read_losses(torch_loss_path)
    b0, b1, br = loss_stats(baby)
    t0, t1, tr = loss_stats(torch_losses)

    baby_finite = all(math.isfinite(v) for v in baby)
    torch_finite = all(math.isfinite(v) for v in torch_losses)
    baby_pass = baby_finite and br <= GATE_RATIO
    torch_pass = torch_finite and tr <= GATE_RATIO

    baby_sample_mse = nearest_prototype_mse(read_samples(baby_samples_path))
    torch_sample_mse = nearest_prototype_mse(read_samples(torch_samples_path))

    lines = [
        "babyCTorch generative trainability verification",
        "================================================",
        f"gate: final/initial mean loss <= {GATE_RATIO:.2f}",
        "",
        f"babyCTorch initial mean loss : {b0:.6f}",
        f"babyCTorch final mean loss   : {b1:.6f}",
        f"babyCTorch final/initial     : {br:.3f}",
        f"babyCTorch finite losses     : {'PASS' if baby_finite else 'FAIL'}",
        f"babyCTorch trainability      : {'PASS' if baby_pass else 'FAIL'}",
        f"babyCTorch sample proto MSE  : {baby_sample_mse:.4f}",
        "",
        f"PyTorch initial mean loss    : {t0:.6f}",
        f"PyTorch final mean loss      : {t1:.6f}",
        f"PyTorch final/initial        : {tr:.3f}",
        f"PyTorch finite losses        : {'PASS' if torch_finite else 'FAIL'}",
        f"PyTorch trainability         : {'PASS' if torch_pass else 'FAIL'}",
        f"PyTorch sample proto MSE     : {torch_sample_mse:.4f}",
        "",
        "Sample prototype MSE is diagnostic, not a hard gate. The primary",
        "claim in this milestone is end-to-end trainability of the same DDPM",
        "objective in babyCTorch and PyTorch.",
    ]
    lines.append(maybe_plot(baby, torch_losses))
    lines.append("")
    lines.append(
        "OVERALL=" + ("PASS" if baby_pass and torch_pass else "FAIL")
    )

    report = "\n".join(lines) + "\n"
    (RESULTS / "verification.txt").write_text(report)
    print(report, end="")
    return 0 if baby_pass and torch_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
