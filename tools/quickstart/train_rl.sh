#!/usr/bin/env bash
# Quick start: train the paper's RL policy on synthetic 2-layer boards and evaluate it.
#
#   bash tools/quickstart/train_rl.sh [--data-dir DIR] [--boards N] [--iterations N] [--seed N] [--gpu N] [--paper] [--smoke]
#
#   --data-dir DIR    boards live in DIR/{train,val,test}; generated there when missing
#                     (default $PCBWORLD_DATA_ROOT/synthetic/synth_2L_v2)
#   --boards N        training boards to generate (default 200; val and test get N/10 each)
#   --iterations N    PPO iterations (default 5)
#   --seed N          training seed (default 42)
#   --gpu N           CUDA device (default 0)
#   --paper           the paper run: 10 000/128/1 000 boards, the shipped d2a.json split, 300 iterations
#   --smoke           one tiny optimizer step, a plumbing check only
# Steps, each command echoed: tools/datagen/synthetic_generator/generate_2layer_v2.sh (+ a split
# file DIR/split.json listing the boards it made) -> experiments/kdd/table1_rl/train_policy.sh
# --method ppo_per_step -> var/outputs/quickstart/rl/ppo_per_step/checkpoints/policy_last.pt ->
# eval/pipeline.py on the test boards (rollout, post-hoc DRC, aggregation) ->
# var/results/quickstart/rl/per_boards_{ckpts,overall,summary}.csv
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_DIR=""; BOARDS=200; ITERATIONS=5; SEED=42; GPU=0; PAPER=0; SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --boards) BOARDS="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --paper) PAPER=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
if [ -z "$DATA_DIR" ]; then
  : "${PCBWORLD_DATA_ROOT:?pass --data-dir, or export PCBWORLD_DATA_ROOT=<your dataset root> (e.g. \$PWD/var/datasets)}"
  DATA_DIR="$PCBWORLD_DATA_ROOT/synthetic/synth_2L_v2"
fi
run() { echo "+ $*" >&2; "$@"; }

OUT=var/outputs/quickstart/rl
RESULTS=var/results/quickstart/rl
if [ "$PAPER" = 1 ]; then
  TRAIN_N=10000; VAL_N=128; TEST_N=1000; ITERATIONS=300
  SPLIT_ARGS=()   # the trainer's default: experiments/kdd/configs/datasets/d2a.json
else
  TRAIN_N="$BOARDS"; VAL_N=$(( BOARDS / 10 )); TEST_N=$(( BOARDS / 10 ))
  SPLIT="$DATA_DIR/split.json"; SPLIT_ARGS=(--split-json "$SPLIT")
fi

# 1. boards
if [ -d "$DATA_DIR/train" ]; then
  echo "[quickstart] boards: $DATA_DIR/train exists, not regenerating" >&2
else
  SHARDS="${SHARDS:-8}"; [ $((TRAIN_N % SHARDS)) -eq 0 ] || SHARDS=1
  run env TRAIN_N="$TRAIN_N" VAL_N="$VAL_N" TEST_N="$TEST_N" SHARDS="$SHARDS" \
    TRAIN_DIR="$DATA_DIR/train" VAL_DIR="$DATA_DIR/val" TEST_DIR="$DATA_DIR/test" \
    bash tools/datagen/synthetic_generator/generate_2layer_v2.sh
fi
if [ "$PAPER" = 0 ]; then
  run python - "$DATA_DIR" "$SPLIT" <<'PY'
import json, pathlib, sys
root, out = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
ids = {s: sorted(p.stem for p in (root / s).glob("board_*.kicad_pcb")) for s in ("train", "val", "test")}
json.dump({"easy": ids, "dataset_dirs": {s: str(root / s) for s in ids}}, open(out, "w"), indent=1)
print(f"split {out}: " + ", ".join(f"{s}={len(v)}" for s, v in ids.items()))
PY
fi

# 2. train
SMOKE_ARGS=(); [ "$SMOKE" = 1 ] && SMOKE_ARGS=(--smoke)
run env NO_WANDB=1 bash experiments/kdd/table1_rl/train_policy.sh --method ppo_per_step --seed "$SEED" --gpu "$GPU" \
  --iterations "$ITERATIONS" --output-root "$OUT" "${SPLIT_ARGS[@]}" "${SMOKE_ARGS[@]}"
# The last checkpoint, not policy_best.pt: the trainer writes a best only after an inline eval
# (every TABLE1_EVAL_EVERY iterations, experiments/kdd/table1_rl/cases.sh), which a few-iteration
# run never reaches.
CKPT="$OUT/ppo_per_step/checkpoints/policy_last.pt"
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }

# 3. evaluate
run python -u eval/pipeline.py --ckpt "$CKPT" --boards-dir "$DATA_DIR/test" --output-dir "$RESULTS" \
  --seed 5600 --n-rollouts 5 --n-envs 8 --rollout-mode parallel --selection-method posthoc_drc_aware --check-angle 45

echo "[quickstart] checkpoint: $CKPT"
echo "[quickstart] results: $RESULTS/per_boards_{ckpts,overall,summary}.csv"
[ "$PAPER" = 1 ] || echo "[quickstart] split: $SPLIT"
