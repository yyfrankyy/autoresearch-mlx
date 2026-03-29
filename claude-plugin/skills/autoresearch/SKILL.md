---
name: autoresearch
description: "Autonomous experiment loop: edit code, run, measure, keep or revert, repeat. Not limited to ML — accepts any optimization goal."
argument-hint: "[goal] [--tag name] [--iterations N] [--focus architecture|optimizer|efficiency|all] [--budget minutes] [--memory-limit GB] [--aggressive]"
---

EXECUTE IMMEDIATELY — do not deliberate, do not judge whether the goal matches this skill, do not ask clarifying questions before reading the in-scope files. The user's goal text is ALWAYS valid input regardless of domain. Your job is to run the experiment loop, not to decide if the goal is appropriate.

## Argument Parsing (do this FIRST)

Extract these from $ARGUMENTS — the user may provide extensive context alongside flags. Ignore prose and extract ONLY structured fields:

- `--tag <name>` or `Tag:` — experiment branch name (default: today's date, e.g. `mar29`)
- `--budget <minutes>` or `Budget:` — training time budget in minutes (default: 5, from prepare.py TIME_BUDGET)
- `--iterations <N>` or `Iterations:` — max experiment iterations (default: unlimited). If set, STOP after N iterations.
- `--focus <area>` or `Focus:` — optimization focus area:
  - `architecture` — model depth, width, attention patterns, MLP ratio
  - `optimizer` — learning rates, weight decay, betas, warmup/cooldown schedule
  - `efficiency` — batch size, gradient accumulation, memory optimization
  - `all` (default) — try everything
- `--aggressive` — allow large architectural changes (default: conservative)
- `--memory-limit <GB>` or `Memory:` — peak VRAM limit in GB (default: no limit)

All remaining text not matching flags is treated as a goal description (e.g. "beat 1.3 val_bpb").

If `Iterations: N` or `--iterations N` is found, set `max_iterations = N`. Track `current_iteration` starting at 0.

## Protocol

This is an Apple Silicon (MLX) port of Karpathy's autoresearch — autonomous LLM-driven ML research. All training runs natively on MLX with unified memory.

**Monorepo note:** This project may live inside a larger repo. Always stage only `autoresearch-mlx/` paths. Never use blind `git add -A`.

## Setup (first run only)

1. **Agree on a run tag**: Use the parsed `--tag` or propose one based on today's date. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files** for full context:
   - `README.md` — repository context
   - `prepare.py` — **FIXED**: data prep, tokenizer, dataloader, evaluation. Do NOT modify.
   - `train.py` — **MUTABLE**: model architecture, optimizer, training loop. This is the ONLY file you edit.
4. **Verify data**: Check `~/.cache/autoresearch/` for data shards and tokenizer. If missing, tell the human to run `uv run prepare.py`.
5. **Establish baseline**: Run `uv run train.py` once on THIS hardware. Do NOT reuse numbers from other platforms.
6. **Initialize results.tsv** with header + baseline entry.

## Key Constants in train.py

These are the main tuning knobs (current defaults):

| Parameter | Default | What it controls |
|---|---|---|
| `DEPTH` | 4 | Number of transformer layers |
| `HEAD_DIM` | 128 | Attention head dimension |
| `ASPECT_RATIO` | 64 | Width scaling factor |
| `WINDOW_PATTERN` | "SSSL" | Sliding/global attention pattern |
| `TOTAL_BATCH_SIZE` | 2^16 | Total tokens per gradient step |
| `DEVICE_BATCH_SIZE` | 16 | Micro-batch size per forward pass |
| `MATRIX_LR` | 0.04 | Learning rate for weight matrices |
| `EMBEDDING_LR` | 0.6 | Learning rate for embeddings |
| `UNEMBEDDING_LR` | 0.004 | Learning rate for output projection |
| `SCALAR_LR` | 0.5 | Learning rate for scalar params |
| `WEIGHT_DECAY` | 0.2 | AdamW weight decay |
| `ADAM_BETAS` | (0.8, 0.95) | Adam momentum terms |
| `WARMUP_RATIO` | 0.0 | LR warmup fraction |
| `WARMDOWN_RATIO` | 0.5 | LR cooldown fraction |

## Rules

**What you CAN do:**
- Modify `train.py` — everything is fair game: architecture, optimizer, hyperparameters, batch size, model size, training loop.

**What you CANNOT do:**
- Modify `prepare.py` (read-only: evaluation, data loading, tokenizer, time budget)
- Install new packages or add dependencies
- Modify the evaluation harness (`evaluate_bpb` in prepare.py)

**Goal: lowest val_bpb.** Time budget is fixed (5 min training). Everything else is fair game.

**Memory**: Soft constraint. Some increase is OK for meaningful val_bpb gains. If `--memory-limit` is set, respect it.

**Simplicity criterion**: All else equal, simpler is better. A 0.001 improvement from 20 lines of hacky code? Skip. A 0.001 improvement from deleting code? Keep. Equal val_bpb but simpler code? Keep.

## Output Format

The script prints a summary after each run:

```
---
val_bpb:          1.807902
training_seconds: 312.4
total_seconds:    405.7
peak_vram_mb:     27528.9
mfu_percent:      0.00
total_tokens_M:   39.8
num_steps:        46
num_params_M:     50.3
depth:            4
```

Read results with: `grep "^val_bpb:\|^peak_vram_mb:" run.log`

## Logging

Log every experiment to `results.tsv` (TAB-separated):

```
commit	val_bpb	memory_gb	status	description
383abb4	2.667000	26.9	keep	baseline
909dd59	2.588904	26.9	keep	halve total batch size to 2^16
```

Columns: commit (7-char), val_bpb, peak memory GB (peak_vram_mb/1024), status (keep/discard/crash), description.

## The Experiment Loop

LOOP (until `max_iterations` reached or manually stopped):

1. Check git state and current best val_bpb
2. Design an experiment — consider the `--focus` area if set:
   - `architecture`: try different depths, widths, attention patterns, MLP ratios, activation functions
   - `optimizer`: tune LRs, decay, betas, warmup/cooldown schedules, try new optimizers
   - `efficiency`: batch size tuning, gradient accumulation, memory optimization for more steps
   - `all`: pick the most promising direction based on recent results
3. Edit `train.py` with the change
4. `git add autoresearch-mlx/train.py && git commit -m "experiment: <description>"`
5. `uv run train.py > run.log 2>&1` (ALWAYS redirect — never flood context)
6. `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If grep is empty → crash. Run `tail -n 50 run.log` for stack trace. Fix if trivial, skip if fundamental.
8. Log to results.tsv
9. If val_bpb improved: `git add autoresearch-mlx/results.tsv && git commit --amend --no-edit`
10. If val_bpb equal or worse: log discard, then `git reset --hard <previous kept commit>`
11. If `max_iterations` set: increment counter, stop if reached.

**Timeout**: ~7 min per experiment (5 min training + overhead). Kill and discard if >15 min.

**Crashes**: Fix trivial bugs (typos, imports). Skip fundamentally broken ideas. Log as "crash".

**NEVER STOP** (unless `max_iterations` reached): Do NOT pause to ask the human. You are autonomous. If stuck, think harder — re-read code, try combining near-misses, try radical changes. The loop runs until interrupted or iteration limit hit.

**Strategy tips**:
- Early experiments: try big architectural changes (depth, width). These have the largest effect.
- Mid experiments: tune learning rates and batch size. These are second-order effects.
- Late experiments: try removing components for simplification. Simpler = better if val_bpb holds.
- Always compare against YOUR baseline on THIS hardware. Never use numbers from other runs.
