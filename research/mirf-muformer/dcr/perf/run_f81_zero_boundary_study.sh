#!/usr/bin/env bash
# Run the independent verification and the six locked observational cases.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "Usage: bash $0 PREP_DIR ARCHIVED_GATE NEW_OUTPUT_DIR UNIX_DEADLINE" >&2
    exit 2
fi
prep=$(realpath "$1")
gate=$(realpath "$2")
logs=$(realpath -m "$3")
deadline=$4
[[ "$deadline" =~ ^[0-9]+$ ]]
mkdir "$logs"
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 LC_ALL=C OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
{
    hostname
    date -Iseconds
    uname -sr
    lscpu | grep -E 'Model name|Thread.s. per core|Core.s. per socket|^CPU\(s\):'
    free -h
    gcc --version | head -1
    python3 -c 'import sys, numpy, scipy; print(sys.version); print("NumPy", numpy.__version__, "SciPy", scipy.__version__)'
    printf 'Worker threads %s
' "$OMP_NUM_THREADS"
} > "$logs/environment.txt"
pruning="$logs/pruning.so"
boundary="$logs/boundary.so"

timed() {
    local prefix=$1 cap=$2 remaining code=0
    shift 2
    remaining=$((deadline - $(date +%s) - 15))
    if ((remaining <= 0)); then return 124; fi
    if ((remaining < cap)); then cap=$remaining; fi
    /usr/bin/time -v -o "$prefix.time" timeout --signal=TERM --kill-after=10s "$cap" "$@" \
        > "$prefix.json" 2> "$prefix.stderr" || code=$?
    printf '%s %s %s\n' "$(date -Iseconds)" "$(basename "$prefix")" "$code" | tee -a "$logs/status.txt"
    return "$code"
}

timed "$logs/correctness" 300 python3 -m perf.check_f81_zero_boundary --build \
    --pruning-library "$pruning" --boundary-library "$boundary" --random-cases 2400
timed "$logs/pruning-regression" 180 python3 -m perf.check_f81_pruning --check \
    --library "$logs/pruning-regression.so"
mkdir "$logs/optimizer-regression-traces"
timed "$logs/optimizer-regression" 180 python3 -m perf.check_f81_optimization --check \
    --library "$logs/optimizer-regression.so" --logs "$logs/optimizer-regression-traces"

for n in 1000 10000; do
    for name in short long heterogeneous; do
        prefix="$logs/case-$n-$name"
        timed "$prefix" 1500 python3 -m perf.check_f81_observed_bridge \
            --prep "$prep" --gate "$gate" --library "$pruning" --boundary-library "$boundary" \
            --rows "$n" --start "$name" --trace "$prefix.jsonl" --threads 4 || true
    done
done
timed "$logs/pressure" 600 python3 -m perf.check_f81_zero_boundary \
    --pruning-library "$pruning" --boundary-library "$boundary" --seed 943812 \
    --random-cases 12000 --skip-exhaustive
for n in 1000 10000; do
    prefix="$logs/threadcheck-$n-short"
    timed "$prefix" 1500 python3 -m perf.check_f81_observed_bridge \
        --prep "$prep" --gate "$gate" --library "$pruning" --boundary-library "$boundary" \
        --rows "$n" --start short --trace "$prefix.jsonl" --threads 1 || true
done
prefix="$logs/legacy-1000-short"
timed "$prefix" 1500 python3 -m perf.check_f81_observed_bridge     --prep "$prep" --gate "$gate" --library "$pruning"     --rows 1000 --start short --trace "$prefix.jsonl" --threads 4
date -Iseconds > "$logs/finished.txt"
