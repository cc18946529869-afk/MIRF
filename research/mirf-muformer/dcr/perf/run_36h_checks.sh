#!/usr/bin/env bash
# Reproduce the gated A diagnostics, categorical B checks and statistical audit.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "Usage: bash $0 PREP_DIR IQTREE_BINARY NEW_OUTPUT_DIR START_EPOCH" >&2
    exit 2
fi
prep=$(realpath "$1")
iqtree=$(realpath "$2")
logs=$(realpath -m "$3")
session_start=$4
[[ "$session_start" =~ ^[0-9]+$ ]]
mkdir "$logs"
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE
ulimit -c 0

stage_deadline=$((session_start + 24*3600))
timed() {
    local path=$1 nominal_limit=$2 current remaining
    shift 2
    current=$(date -u +%s)
    remaining=$((stage_deadline - current - 10))
    if ((remaining <= 0)); then
        echo "Stage deadline reached; command not submitted" >&2
        return 124
    fi
    if ((nominal_limit > remaining)); then
        nominal_limit=$remaining
    fi
    /usr/bin/time -v -o "$path" timeout --signal=TERM --kill-after=5s "$nominal_limit" "$@"
}

event() {
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$logs/timeline.log"
}

event "A pre-fit exact gate started"
timed "$logs/a-gate.time" 1200 python3 -m perf.cluster_gate --prep "$prep" \
    --scratch "$logs/private" --ledger "$logs/a-gate.json" --threads 36 | tee "$logs/a-gate.stdout"
python3 - "$logs/a-gate.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    gate = json.load(handle)
if any(x["eligible"] for x in gate["granularities"].values()):
    raise SystemExit("At least one partition passed. This diagnostic driver stops for its separate fitting stage.")
PY
event "A stopped at its preregistered gate; B diagnostic stage started"

stage_deadline=$(($(date -u +%s) + 8*3600))
if ((stage_deadline > session_start + 32*3600)); then
    stage_deadline=$((session_start + 32*3600))
fi
timed "$logs/kernel.time" 180 python3 -m perf.check_categorical_tree --check > "$logs/kernel.json"
timed "$logs/fixture.time" 180 python3 -m perf.check_categorical_tree --fixture-dir "$logs/fixture" > "$logs/iq-reference.json"
(
    cd "$logs"
    "$iqtree" --version > iq-version.txt
    sha256sum "$iqtree" >> iq-version.txt
    for model in F81 MK; do
        status=0
        timed "iq-$model.time" 180 "$iqtree" -s fixture/categories.phy -st MORPH \
            -m "$model" -fs fixture/profiles.txt -te fixture/toy.tree -blfix -nt 1 \
            --prefix "iq-$model" > "iq-$model.stdout" 2>&1 || status=$?
        printf '%s %s\n' "$model" "$status" >> iq-exits.txt
        test "$status" = 2
        grep -Fq 'Frequencies must be strictly positive and smaller than 1' "iq-$model.stdout"
    done
)
for n in 1000 5000 10000; do
    timed "$logs/scale-$n.time" 600 python3 -m perf.check_categorical_tree --scale "$n" > "$logs/scale-$n.json"
done
event "B numerical checks and IQ-TREE incompatibility probes completed; audit started"

stage_deadline=$(($(date -u +%s) + 4*3600))
if ((stage_deadline > session_start + 36*3600)); then
    stage_deadline=$((session_start + 36*3600))
fi
timed "$logs/a-audit.time" 600 python3 -m perf.audit_cluster_gate "$logs/a-gate.json" \
    "$logs/private" "$prep" > "$logs/a-audit.json"
timed "$logs/median-sequential.time" 180 python3 -m perf.median_sequential > "$logs/median-sequential.json"
event "All authorized diagnostic checks complete"
