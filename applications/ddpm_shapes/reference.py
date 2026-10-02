#!/usr/bin/env python3
"""PyTorch oracle for the babyCTorch tiny-image DDPM trainability test."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

IMAGE_SIDE = 4
IMAGE_DIM = 16
INPUT_DIM = 17
HIDDEN_DIM = 32
DIFF_STEPS = 8
NUM_SAMPLES = 8


def make_prototypes() -> torch.Tensor:
    images = []
    for col in (1, 2):
        image = -torch.ones(IMAGE_SIDE, IMAGE_SIDE)
        image[:, col] = 1.0
        images.append(image.flatten())
    for row in (1, 2):
        image = -torch.ones(IMAGE_SIDE, IMAGE_SIDE)
        image[row, :] = 1.0
        images.append(image.flatten())
    return torch.stack(images)


def make_schedule() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = torch.linspace(1.0e-4, 0.15, DIFF_STEPS)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def make_model() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM, IMAGE_DIM),
    )
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def train(args: argparse.Namespace):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    prototypes = make_prototypes()
    beta, alpha, alpha_bar = make_schedule()
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    losses: list[float] = []
    for step in range(args.steps):
        x0 = prototypes[torch.randint(0, len(prototypes), (args.batch,))]
        t = torch.randint(0, DIFF_STEPS, (args.batch,))
        eps = torch.randn(args.batch, IMAGE_DIM)
        abar = alpha_bar[t].unsqueeze(1)
        xt = torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * eps
        t_norm = 2.0 * t.float() / (DIFF_STEPS - 1) - 1.0
        denoiser_input = torch.cat((xt, t_norm.unsqueeze(1)), dim=1)

        pred = model(denoiser_input)
        loss = torch.mean((pred - eps) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

        if step == 0 or (step + 1) % 500 == 0 or step + 1 == args.steps:
            print(f"[PyTorch] step {step + 1}/{args.steps} loss={losses[-1]:.6f}")

    return model, losses, (beta, alpha, alpha_bar)


def sample(
    model: nn.Module,
    schedule: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int,
) -> torch.Tensor:
    beta, alpha, alpha_bar = schedule
    torch.manual_seed(seed + 999)
    x = torch.randn(NUM_SAMPLES, IMAGE_DIM)

    with torch.no_grad():
        for t in reversed(range(DIFF_STEPS)):
            t_norm = torch.full(
                (NUM_SAMPLES, 1), 2.0 * t / (DIFF_STEPS - 1) - 1.0
            )
            eps_pred = model(torch.cat((x, t_norm), dim=1))
            mean = (
                x - beta[t] / torch.sqrt(1.0 - alpha_bar[t]) * eps_pred
            ) / torch.sqrt(alpha[t])
            if t > 0:
                x = mean + torch.sqrt(beta[t]) * torch.randn_like(x)
            else:
                x = mean
    return x


def mean_window(values: list[float], from_end: bool = False) -> float:
    window = min(100, max(1, len(values) // 4))
    chunk = values[-window:] if from_end else values[:window]
    return sum(chunk) / len(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.10)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, losses, schedule = train(args)
    generated = sample(model, schedule, args.seed)

    with (args.out_dir / "pytorch_loss.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("step", "loss"))
        writer.writerows(enumerate(losses))

    with (args.out_dir / "pytorch_samples.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample"] + [f"p{i}" for i in range(IMAGE_DIM)])
        for i, row in enumerate(generated.tolist()):
            writer.writerow([i] + row)

    initial = mean_window(losses)
    final = mean_window(losses, from_end=True)
    ratio = final / initial
    gate = ratio <= 0.65
    summary = (
        "PyTorch DDPM reference\n"
        "domain=4x4 bar images\n"
        "architecture=17-32-32-16 tanh MLP\n"
        f"diffusion_steps={DIFF_STEPS}\n"
        "optimizer=SGD\n"
        f"learning_rate={args.lr:.6f}\n"
        f"training_steps={args.steps}\n"
        f"batch_size={args.batch}\n"
        f"initial_mean_loss={initial:.9f}\n"
        f"final_mean_loss={final:.9f}\n"
        f"final_over_initial={ratio:.9f}\n"
        f"trainability_gate={'PASS' if gate else 'FAIL'}\n"
    )
    (args.out_dir / "pytorch_summary.txt").write_text(summary)
    print(
        f"[PyTorch] initial={initial:.6f} final={final:.6f} "
        f"ratio={ratio:.3f} gate={'PASS' if gate else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
