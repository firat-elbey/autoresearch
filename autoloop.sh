#!/usr/bin/env bash
# Headless autoresearch: drives the experiment loop from program.md using
# `claude -p` (non-interactive mode). Each experiment runs in a fresh claude
# invocation with a small context; state carries over via git history and
# results.tsv, exactly as program.md prescribes.
#
# Usage:
#   ./autoloop.sh [num_experiments]      # default 5
#
# Environment:
#   CLAUDE_MODEL      claude model to drive the loop (default: haiku — cheap,
#                     good enough to test the loop; use sonnet/opus for real runs)
#   AUTORESEARCH_TAG  run tag, becomes branch autoresearch/<tag> (default: today)
#
# Requires the claude CLI (https://claude.com/claude-code) authenticated on
# this machine, and a working `uv run train.py` (see README quickstart first).
#
# WARNING: runs claude with --dangerously-skip-permissions so it can edit
# train.py, commit, and launch training runs unattended. Run it in this repo
# on a machine/checkout you're comfortable letting an agent loose on.

set -euo pipefail
cd "$(dirname "$0")"

N="${1:-5}"
MODEL="${CLAUDE_MODEL:-haiku}"
TAG="${AUTORESEARCH_TAG:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
BRANCH="autoresearch/${TAG}"

# --- One-time setup (program.md's Setup section, done here since there is no
# --- human in the loop): experiment branch + results.tsv header.
if git rev-parse --verify --quiet "$BRANCH" >/dev/null; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH"
fi
[ -f results.tsv ] || printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n' > results.tsv

read -r -d '' PROMPT <<'EOF' || true
You are the autonomous researcher described in program.md. Read program.md,
README.md, prepare.py and train.py for context. Setup is already done: you are
on the experiment branch and results.tsv exists with a header row. Read
results.tsv and `git log --oneline -20` to see what has already been tried.

Now run exactly ONE experiment iteration:
1. If results.tsv has no data rows yet, this is the baseline: leave train.py
   unmodified. Otherwise, modify train.py with ONE new experimental idea that
   has not been tried yet.
2. git commit train.py (even for the baseline, commit only if there are changes).
3. Run: uv run train.py > run.log 2>&1
4. Read the result: grep "^val_bpb:\|^peak_vram_mb:" run.log
   If empty, the run crashed — check tail -n 50 run.log; fix trivial bugs and
   re-run once or twice, otherwise record it as a crash.
5. Append ONE row to results.tsv (tab-separated: short commit hash, val_bpb,
   memory_gb, status keep/discard/crash, short description). Never commit
   results.tsv.
6. Keep or revert: if val_bpb improved on the best previous "keep" row (or this
   is the baseline), keep the commit. Otherwise revert your experiment with:
   git reset --hard HEAD~1

Then STOP. Do not start another experiment — the outer loop handles that.
EOF

echo "Branch: $BRANCH | model: $MODEL | experiments: $N"
for i in $(seq 1 "$N"); do
  echo "=== experiment $i/$N ($(date '+%H:%M:%S')) ==="
  claude -p --model "$MODEL" --dangerously-skip-permissions "$PROMPT" \
    || echo "claude exited non-zero on experiment $i; continuing"
done

echo "Done. Results:"
column -t -s$'\t' results.tsv
