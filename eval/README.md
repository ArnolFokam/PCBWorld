# `eval/` — KiCad PCB evaluation

`eval.metrics.evaluate_one` is the canonical DRC scorer for a routed `.kicad_pcb`: it scores the board against
its source `.kicad_pro` design rules and returns a metric dict. `eval.evaluator.Evaluator` drives both evaluation
modes — rollout eval (`Evaluator.run`) and post-hoc board scoring (`Evaluator.score_boards`) — feeding the same
`eval.metrics.EvalSummary` into the sinks (logger / CSV / JSON). The staged CLI is `python -m eval.pipeline`, fronted
by [scripts/eval.py](../scripts/eval.py); setup [docs/QUICKSTART.md](../docs/QUICKSTART.md), numbers [docs/METRICS.md](../docs/METRICS.md).

## Module map

| File | Role |
|---|---|
| `metrics.py` | Scoring kernel and metric source of truth: `evaluate_one` / `compute_metrics` (disk board + `.pro`) and `compute_metrics_inline`, the non-destructive live-env entry (u₀ from the env's reset-time capture) that both branch wrappers expose as their `eval_inline_drc` `env_method` hook. Also defines `EvalSummary`, the sink-agnostic summary. |
| `evaluator.py` | `Evaluator.run()` (rollout, mode A) · `Evaluator.score_boards()` (post-hoc, mode B) + the CSV/JSON sinks (`export_csv` / `export_json` / `emit_csv_artifacts`). |
| `pipeline.py` | The `python -m eval.pipeline` 3-stage CLI (`main`) and the post-hoc DRC stage `eval_kicad_pcb()` over saved artifacts. |
| `aggregation.py` | Stage 3: `aggregate_boards` (post-hoc, from `per_rollout.csv`) and `aggregate_inline` (in-memory, training validation). |
| `eval_utils.py` | Stdlib-only CSV / schema / flattening helpers, plus the runtime metric-semantics kernel: `runtime_metrics_from_info()` (+ `success_from` / `clean_pass_from`) is the single definition of the live-env per-rollout derivations (success, clean_pass, ratsnest_reduction, …) — every producer calls it instead of computing its own. |
| `args.py` | Shared argparse builders for the LLM-env eval entry points; the flag list derives from the config schema. |
| `rollout/rl.py` | Stage-1 RL rollout **driver** (`eval_transformer`): packs `(board, rollout)` jobs into waves that fill all `n_envs` slots (`plan_job_schedule`, boards small-first to cut straggler wait), drives `methods/rl_agent/rollout/transformer.py`, flushes per-rollout rows. This is what the CLI injects into Stage 1. |
| `rollout/rule_based.py` | Stage-1 rule-based router rollout (FreeRouting / KRT / OrthoRoute). |

Same pipeline, other packages: LLM Stage-1 producers in `methods/llm_agent/rollout/`; board loaders (`BoardSpec`)
in `methods/_shared/board_loader.py`; metric-logging sinks in `methods/_shared/logger.py`; policy/env-kwargs
builders in `methods/rl_agent/models/loader.py`; the DRC worker pool (`SubprocEvalPool`) in `pcb_world/vec/subproc_pool.py`.

## Metric columns

Produced per routed board by `evaluate_one` / `compute_metrics`, written to `per_rollout.csv`:

| metric | meaning |
|---|---|
| `success` | every net's pads form one connectivity group — equivalent to an empty ratsnest when no dangling copper is present |
| `routability` | `Σᵢ(Gᵢ(0) − Gᵢ(t)) / Σᵢ(Gᵢ(0) − 1)` over pad groups `Gᵢ` (connectivity clusters holding ≥1 pad): exactly 0 on the bare board, exactly 1 fully connected; dangling copper holds no pad so it cannot enter the metric, and a lower-bound violation raises instead of emitting a negative score. Filled by the DRC stage **only** — the rollout stage leaves the column NaN; the env-side per-step proxy is `ratsnest_reduction` (signed) |
| `track_count`, `via_count`, `wirelength_mm` | engine reward snapshot (`run_drc=True`) |
| `drv_errors_only_count` · `drv_errors_and_promoted_count` | ERROR-severity violations · ERROR + the 3 promoted warnings (codes 12/13/37, below) |
| `drv_violations` | per-violation records: severity (int) + label, error_code, error_type, x_mm, y_mm, layer, net_names, `is_error`, `is_promoted` |
| `final_potential` · `initial_potential` · `potential_gain` | Φ(routed) · Φ(bare board, all tracks/vias deleted) · their difference. Both ends score with `run_drc=True`, so the gain never mixes a DRC-free initial with a DRC-included final. Board-dependent Φ terms resolve through the training env's own [`PotentialReward.bind_board`](../pcb_world/core/reward.py) (per-reset group = the same bare-board pad groups as the routability baseline), so offline Φ == training Φ — pinned by [tests/test_reward_parity.py](../tests/test_reward_parity.py). `initial_potential` is `None` on the inline path |
| `phi_components` · `phi_weights` | the five base Φ terms plus their `total` (penalties signed negative; ladder / clean-completion terms are **not** itemized, so `total` ≠ `final_potential` under ladder rules) · the reward config's weights as resolved on this board, enough to re-derive Φ |
| `extras.per_net` · `extras.board_meta` | per-net wirelength / track / via / unrouted-edges · bbox, net count, copper layer count |

Which violations count, and how the DRC term is shaped, is the reward config's business (`configs/reward/`, knob
`drc_severity_mode`); `--reward-config NAME` scores under another rule, warning when it differs from the ckpt's own.

## Source `.pro` matching

`evaluate_one(routed_pcb, pro_path)` takes the project path directly; through the pipeline it is resolved per board
by `eval.eval_utils.resolve_pro_path`, in order: the co-located `<stem>.kicad_pro`; for the cell grammar, the longest
co-located `<board_id>.kicad_pro` whose stem prefixes the file; then, under `--boards-dir`, `<stem>.kicad_pro` with
and without the staging suffix `_rollout_<idx>`.

## CLI — `python -m eval.pipeline`

Three stages: **(1) rollout** a live policy over a board set, **(2) post-hoc DRC** over the saved `.kicad_pcb` artifacts, **(3) aggregate**.

```bash
python -m eval.pipeline --ckpt <ckpt> --boards-dir <dir> --seed 42 --n-rollouts 5 --n-envs 1
python -m eval.pipeline --skip-rollout --output-dir <rollout-dir> --stages eval,aggregate   # score + aggregate only
```

Every flag is in `--help` and every default in the config schema; only these carry a constraint:

* `--ckpt`, one of `--boards-dir` / `--boards-list`, `--seed`, `--n-rollouts` — required unless `--skip-rollout`.
* `--skip-rollout` needs `--output-dir` pointing at a dir that already holds a `per_rollout.csv` (or a `boards/`
  dir of routed `.kicad_pcb` to reconstruct the rows from) — this is how the rule-based and LLM producers are
  scored through the same path.
* `--stages rollout,eval,aggregate` is the positive selector (aliases `drc`=eval, `agg`=aggregate); it overrides
  `--skip-rollout` / `--skip-drc` / `--skip-aggregate`.
* `--rollout-mode serial` requires `--n-envs 1`; `--n-envs` is also the Stage-2 DRC worker count.
* `--inline-drc on` scores on the live engine and skips Stage 2 — serial only, and incompatible with `--skip-rollout`.
* `--save-artifacts off` leaves nothing for Stage 2 to score.
* `--env-drc` / `--check-angle`, when omitted, inherit the ckpt's `emit_drc_tokens` / `corner_mode`.

## Library API — post-hoc board scoring

```python
eval_kicad_pcb(rollout_dir, n_workers=8, boards_dir=src)           # eval.pipeline — Stage 2 over a rollout dir
Evaluator.score_boards([(routed_pcb, pro_path), ...], parallel=8)  # eval.evaluator — pairs -> EvalSummary
evaluate_one(routed_pcb, pro_path, reward_config_name="...")       # eval.metrics — one board, the kernel
```
`eval_kicad_pcb` merges its DRC columns back into the dir's `per_rollout.csv`; `parallel>1` fans out over `SubprocEvalPool`.

## Output tree

A pipeline run writes into `--output-dir`:

```
<output-dir>/
├── boards/                    # routed rollouts, unified cell grammar (when --save-artifacts on):
│                              #   <board_id>_<cell>_s<SS>_r<RR>.kicad_pcb (+ .kicad_pro/.kicad_prl)
├── per_rollout.csv            # one row per rollout, DRC columns filled by Stage 2
├── per_boards_ckpts.csv       # Stage 3: one row per board x ckpt seed
├── per_boards_overall.csv     # Stage 3: one row per board
├── per_boards_summary.csv     # Stage 3: one model-level row
├── manifest.json              # env_kwargs + resolved args
└── eval_overall_summary.json  # overall summary + stage wall times
```

Stage 1 saves each rollout under a temporary `artifacts/` staging dir, then flattens it into `boards/` under
the cell grammar (`flatten_rollout_artifacts`; `<cell>` = the output dir basename, `s<SS>` = the ckpt's
training seed), rewriting the `artifact_path` column to match. Stage 3 parses exactly that grammar from
`per_rollout.csv` and **fails loudly** (`SystemExit`) when no row matches — it never exits 0 with nothing written.
`per_rollout.csv` is flushed incrementally during the rollout and updated in place by the post-hoc DRC stage,
keyed by the saved board filename (unique per rollout across ckpt seeds).

## Reading the DRV breakdown

`drv_breakdown.errors_only_by_type` and `drv_breakdown.errors_and_promoted_by_type` each list
`{severity, error_code, error_type, count}` rows, where `severity` is the human label (`ERROR` / `WARNING`)
and the integer KiCad value stays in `drv_violations[i].severity`. The 3 promoted warnings:

| code | error_type | meaning |
|---|---|---|
| 12 | via_dangling | a via with no track ending on either copper layer |
| 13 | track_dangling | a track segment with one end floating |
| 37 | net_conflict | a track/via on a net that disagrees with the connectivity (potential short) |

## Caveats

* **The source `.pro` is mandatory.** Without it the rules would fall back to KiCad's compile-time defaults
  (more permissive — `Track width` / `Via diameter` / `Hole size` undercounted), so `KiCadEngine` raises
  `RuntimeError` instead of scoring.
* **Singleton C++ engine.** One `RLRouter` per process, so serial scoring is sequential; parallel scoring
  fans out across `forkserver` subprocesses (`SubprocEvalPool`, `--n-envs N` / `parallel=N`).
* **`final_potential` ≠ the training reward at convergence.** The training reward is the per-step potential
  delta plus step penalty, not Φ itself; `final_potential` is Φ of the finished board under the config's weights.
