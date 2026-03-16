# StyleGAN3 with a few tweaks

This is a fork of the original [StyleGAN3](https://github.com/NVlabs/stylegan3) codebase.

The goal of this repository is to continue developping the tooling for training and inferring StyleGAN3 for my own usage on macOS, Windows and Linux.

## Tweaks

- Fix training resume feature to restart fron last known kimg iteration count
- Apple Silicon support for image and video generation
- Add support for quantized palette images in dataset creation tool

## Carbon Tracking

Training runs are instrumented with [CodeCarbon](https://codecarbon.io/) to measure energy consumption and CO2 equivalent emissions. This runs automatically on rank 0 during training.

**Outputs per training run:**
- `emissions.csv` — detailed CodeCarbon report (flushed every tick)
- `energy/tick` (Wh) and `co2eq/tick` (g CO2eq) in the console status line
- `Emissions/energy_wh` and `Emissions/co2eq_g` curves in TensorBoard

To disable emission tracking, pass `--no-emissions` to the training script.

## Setup

This was tested on Python 3.13 only.

Recommended to use [uv](https://docs.astral.sh/uv/) for easy setup.

To run scripts use `uv run <script.py>`.