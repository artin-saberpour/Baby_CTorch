# Tiny image DDPM trainability test

This is the first generative application for babyCTorch. It is intentionally small enough to debug at the framework level while still exercising a genuine diffusion training and sampling pipeline.

## What it learns

The data distribution contains four 4x4 image modes: a vertical bar in either center column and a horizontal bar in either center row. Pixels are represented as -1 or +1.

The forward diffusion process uses 8 timesteps. A babyCTorch MLP receives the noisy 16-pixel image plus one normalized timestep coordinate and predicts the Gaussian noise that produced the corrupted image.

Architecture:

```text
17 -> Linear(32) -> tanh -> Linear(32) -> tanh -> Linear(16)
```

The application uses babyCTorch `Linear`, `tanh`, `MSE`, autograd, and `SGD`. Sampling starts from Gaussian noise and runs the learned reverse diffusion chain.

## Why start here instead of MNIST

A Transformer would currently require several additional framework primitives such as indexed embedding lookup, axis-wise softmax, log/cross-entropy, reshape, and permutation. A convolutional MNIST DDPM would similarly add convolutional complexity before the existing autograd path has been stressed by a generative objective.

This experiment isolates the core question first: can babyCTorch train a modern generative objective end to end using its own tensor engine and backward functions?

Once this passes, the next milestone is the same diffusion objective on MNIST, followed by the Transformer primitives and an autoregressive image Transformer.

## Verification protocol

The runner trains two matched models:

1. babyCTorch C implementation
2. PyTorch reference implementation

Both use the same architecture, image distribution, DDPM schedule, SGD optimizer, learning rate, batch size, and number of training steps. Their random-number streams are not expected to match exactly.

The hard trainability gate is:

```text
mean(last 100 losses) / mean(first 100 losses) <= 0.65
```

All losses must also remain finite. Generated-sample distance to the four image prototypes is reported as a diagnostic, but is not used as a hard pass/fail criterion at this first milestone.

## Run

From the repository root on Windows, with CUDA `nvcc`, Visual Studio build tools, Python, and PyTorch available:

```powershell
powershell -ExecutionPolicy Bypass -File .\applications\ddpm_shapes\RUN_ALL.ps1
```

For a short build/smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File .\applications\ddpm_shapes\RUN_ALL.ps1 -Quick
```

`-Quick` is only a smoke test. Use the default 4000-step run for the trainability claim.

Outputs are written under `applications/ddpm_shapes/results/`, including loss traces, generated samples, summaries, a PyTorch comparison, and an optional loss plot when matplotlib is installed.

## Core issues exposed by this workload

The application required two framework-level correctness fixes rather than application workarounds:

- CPU matrix multiplication must keep autograd enabled when either operand requires gradients. This is required for the normal `data @ trainable_weight` case.
- MSE averaging must keep the graph connected when the averaging scalar is a constant.

The branch also corrects Linear initialization to symmetric Xavier/Glorot uniform weights with zero biases.
