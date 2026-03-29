# autoresearch-mlx

Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time autonomous research loops controlled through `program.md`. This port keeps the same basic rules: one mutable `train.py`, one metric (`val_bpb`), a fixed 5-minute training budget, and keep-or-revert via git. It runs natively on Apple Silicon through [MLX](https://github.com/ml-explore/mlx), so there is no PyTorch or CUDA dependency.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# one-time data + tokenizer prep
uv run prepare.py

# run one 5-minute training experiment
uv run train.py
```

Then use the Claude Code plugin to run the autonomous loop (see below).

## Claude Code Plugin Usage

Install this plugin in Claude Code, then use the `/autoresearch` slash command:

```
/autoresearch [goal] [options]
```

### Examples

```bash
# Run with all defaults (tag=today's date, focus=all, budget=5min, unlimited iterations)
/autoresearch

# Set a specific optimization goal
/autoresearch beat 1.3 val_bpb

# Run exactly 3 iterations focusing on architecture
/autoresearch --iterations 3 --focus architecture

# Aggressive architectural search with memory limit
/autoresearch --focus architecture --aggressive --memory-limit 24

# Tune optimizer hyperparameters with a custom branch tag
/autoresearch --tag lr-sweep --focus optimizer --iterations 5

# Full example with all options
/autoresearch beat 1.25 val_bpb --tag march-run --iterations 10 --focus all --budget 5 --memory-limit 32
```

### Options

| Option | Default | Description |
|---|---|---|
| `goal` | *(none)* | Free-text optimization target, e.g. `beat 1.3 val_bpb` |
| `--tag <name>` | today's date | Git branch name: `autoresearch/<tag>` |
| `--iterations <N>` | unlimited | Stop after N experiments |
| `--focus <area>` | `all` | One of: `architecture`, `optimizer`, `efficiency`, `all` |
| `--budget <minutes>` | `5` | Training time budget per experiment |
| `--memory-limit <GB>` | no limit | Peak VRAM limit in GB |
| `--aggressive` | off | Allow large architectural changes |

### What happens

1. Creates a git branch `autoresearch/<tag>`
2. Establishes a baseline by running `train.py` on your hardware
3. Enters an autonomous loop: edit → train → evaluate → keep/revert
4. Logs every experiment to `results.tsv`
5. Keeps going until iteration limit or you interrupt (Ctrl+C)

### Prerequisites

Before first run, prepare the data:

```bash
cd autoresearch-mlx
uv sync
uv run prepare.py   # downloads data shards + tokenizer to ~/.cache/autoresearch/
```

## What matters

- `prepare.py` - data prep, tokenizer, dataloader, and evaluation. Treat as fixed.
- `train.py` - model, optimizer, and training loop. This is the file the agent edits.
- `program.md` - the autonomous experiment protocol.
- `results.tsv` - logged experiment history.

The loop is the same as upstream: edit `train.py`, run a fixed-budget experiment, read `val_bpb`, keep the change if it wins, revert if it loses, and repeat.

## Public baseline results

The public `results.tsv` captures the initial hardware-local walk from the default baseline down to `1.807902`:

| Commit | val_bpb | Status | Description |
|---|---:|---|---|
| `383abb4` | 2.667000 | keep | baseline (AdamW, default config) |
| `909dd59` | 2.588904 | keep | halve total batch size to `2^16` |
| `4161af3` | 2.533728 | keep | increase matrix LR to `0.04` |
| `5efc7aa` | 1.807902 | keep | reduce depth from `8` to `4` |

That result already shows the core Apple Silicon pattern: with a fixed 5-minute wall clock, smaller faster-training models can beat larger ones simply by fitting more optimizer steps into the budget.

## Longer Apple Silicon runs

Longer overnight runs on the working MLX port pushed much further. The long Mac Mini test is included here because it found a meaningfully different winner stack from the Max-class machines.

| Machine | Current best | Starting point | Repeated wins |
|---|---:|---:|---|
| M4 Max #1 | 1.294526 | 1.596971 | AdamW-only, low matrix LR, 3x MLP, no logit cap, moderate weight decay |
| M4 Max #2 | 1.330509 | 1.807902 | leaner batch, long anneal, SiLU, lower regularization, no logit cap |
| Mac Mini (long run) | 1.353329 | 1.922472 | Muon, sharper attention, smaller MLP, lower scalar LR |

The Mac Mini result matters because it did not just rediscover the same exact recipe. On smaller Apple Silicon hardware, the strongest changes leaned toward more aggressive step-efficiency wins. Later transfer tests showed some of those Mac Mini findings did not carry cleanly onto the Max baseline, which is exactly the kind of hardware-specific behavior this loop is useful for uncovering.

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon training with unified memory.
- **AdamW-only public path.** This public `train.py` keeps the default path simple. The long Mac Mini run above explored a Muon variant in the working port, but that branch is not exposed as a public default here.
- **Smaller eval token budget.** Reduced for faster iteration on Apple Silicon while keeping the same `evaluate_bpb` interface in `prepare.py`.
- **Roughly 6-7 minutes per experiment.** Expect 5 minutes of training plus compile and eval overhead.
- **MFU reporting is placeholder.** There is no Apple Silicon equivalent to the H100 FLOPs reference used upstream.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) - autoresearch and nanochat
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) - MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) - MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. See [LICENSE](LICENSE).
