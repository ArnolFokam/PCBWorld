#!/usr/bin/env bash
# Quick start: route a board with an LLM tool-calling agent.
#
#   bash tools/quickstart/run_llm.sh [--board PATH] [--provider google|together|anthropic|openai] [--model NAME]
#
# Two runs of methods/llm_agent/rollout/pcbworld.py, every command echoed:
#   1. --mode fixed: no API key. Prints the exact prompt the agent receives, executes one action
#      and shows the resulting state.
#   2. --mode api: routes the board end to end, one chat completion per step, with the first
#      provider whose key is in the environment (any of them; free tiers work):
#        GEMINI_API_KEY / GOOGLE_API_KEY -> google     (default model gemini-2.0-flash)
#        TOGETHER_API_KEY                -> together   (--model picks the model)
#        ANTHROPIC_API_KEY               -> anthropic  (claude-sonnet-4-20250514)
#        OPENAI_API_KEY                  -> openai     (gpt-4o; OPENAI_BASE_URL = any compatible endpoint)
#      Skipped, with a message, when no key is set.
#   --board PATH   the .kicad_pcb to route (default tests/fixtures/simple_routing_board.kicad_pcb)
#   --provider P   force a provider instead of detecting one from the keys
#   --model NAME   provider model (default: the provider's own)
set -euo pipefail
cd "$(dirname "$0")/../.."

BOARD=tests/fixtures/simple_routing_board.kicad_pcb; PROVIDER=""; MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --board) BOARD="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

run() { echo "+ $*" >&2; "$@"; }
ROLLOUT=methods/llm_agent/rollout/pcbworld.py

run python "$ROLLOUT" --mode fixed --board_path "$BOARD" --env_num 1

if [ -z "$PROVIDER" ]; then
  if   [ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]; then PROVIDER=google
  elif [ -n "${TOGETHER_API_KEY:-}" ];                  then PROVIDER=together
  elif [ -n "${ANTHROPIC_API_KEY:-}" ];                 then PROVIDER=anthropic
  elif [ -n "${OPENAI_API_KEY:-}" ];                    then PROVIDER=openai
  fi
fi
if [ -z "$PROVIDER" ]; then
  echo "[quickstart] no LLM API key in the environment (GEMINI/GOOGLE/TOGETHER/ANTHROPIC/OPENAI_API_KEY) — skipping the API rollout"
  exit 0
fi
if [ "$PROVIDER" = together ] && [ -z "$MODEL" ]; then
  echo "[quickstart] together has no default model: pass --model <together model id>" >&2; exit 2
fi
MODEL_ARGS=(); [ -n "$MODEL" ] && MODEL_ARGS=(--api_model "$MODEL")
run python "$ROLLOUT" --mode api --api_provider "$PROVIDER" "${MODEL_ARGS[@]}" \
  --board_path "$BOARD" --env_num 1 --max_steps 40 --rollout_episodes 1 --silent
