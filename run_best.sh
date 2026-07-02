#!/usr/bin/env bash

set -uo pipefail

usage()
{
    cat <<'EOF'
Usage: ./run_best.sh [--gpu a100|h100] [--results-dir DIR] KERNEL PRECISION SIZE TIMESTEP

Run the binary selected by a previous benchmark_generated.sh run.

Arguments:
  KERNEL       Kernel name, for example j2d5pt or star3d1r
  PRECISION    float or double
  SIZE         Compute size along every spatial dimension
  TIMESTEP     Total number of time steps

Options:
  --gpu GPU          Override automatic A100/H100 detection
  --results-dir DIR  Benchmark result directory (default: results)
  -h, --help         Show this help

The selected binary is executed five times (-n 5). CUDA_VISIBLE_DEVICES is
inherited unchanged.
EOF
}

die()
{
    echo "Error: $*" >&2
    exit 1
}

gpu=""
results_root="results"
arguments=()

while (($#)); do
    case "$1" in
        --gpu)
            (($# >= 2)) || die "--gpu requires an argument"
            gpu="$2"
            shift 2
            ;;
        --results-dir)
            (($# >= 2)) || die "--results-dir requires an argument"
            results_root="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            arguments+=("$1")
            shift
            ;;
    esac
done

((${#arguments[@]} == 4)) || {
    usage >&2
    exit 1
}

kernel="${arguments[0]}"
precision="${arguments[1]}"
size="${arguments[2]}"
timestep="${arguments[3]}"

[[ "$precision" == float || "$precision" == double ]] ||
    die "PRECISION must be float or double"
[[ "$size" =~ ^[1-9][0-9]*$ ]] || die "SIZE must be a positive integer"
[[ "$timestep" =~ ^[1-9][0-9]*$ ]] || die "TIMESTEP must be a positive integer"

if [[ -z "$gpu" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 ||
        die "nvidia-smi was not found; specify --gpu a100 or --gpu h100"

    visible_device="${CUDA_VISIBLE_DEVICES:-0}"
    visible_device="${visible_device%%,*}"
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$visible_device" 2>/dev/null | head -n 1)"
    case "${gpu_name^^}" in
        *A100*) gpu="a100" ;;
        *H100*) gpu="h100" ;;
        *)
            die "could not identify visible GPU '$gpu_name' as A100 or H100; specify --gpu"
            ;;
    esac
fi

case "$gpu" in
    a100|h100) ;;
    *) die "unsupported GPU '$gpu' (expected a100 or h100)" ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1

best_csv="$results_root/$gpu/best_results.csv"
[[ -f "$best_csv" ]] ||
    die "best results not found: $best_csv (run benchmark_generated.sh first)"

row="$(awk -F, -v kernel="$kernel" -v precision="$precision" '
    NR > 1 && $1 == kernel && $2 == precision && $10 == "success" {
        print
        exit
    }
' "$best_csv")"
[[ -n "$row" ]] ||
    die "no successful best result for kernel '$kernel' with precision '$precision' in $best_csv"

IFS=, read -r selected_kernel selected_precision bs1 bs2 bt sl regnum gflops ms status log <<< "$row"

if [[ -n "$bs2" ]]; then
    config="${selected_kernel}-${bs1}x${bs2}-${bt}-${sl}"
else
    config="${selected_kernel}-${bs1}-${bt}-${sl}"
fi
binary="$results_root/$gpu/bin/$selected_precision/reg$regnum/$config"
[[ -x "$binary" ]] || die "selected binary is missing or not executable: $binary"

echo "GPU: $gpu"
echo "Selected: kernel=$selected_kernel precision=$selected_precision bS1=$bs1 bS2=${bs2:-N/A} bT=$bt sl=$sl REGNUM=$regnum"
echo "Benchmark result: $gflops GFLOPS, $ms ms"
echo "Running: size=$size timestep=$timestep repetitions=5"

exec "$binary" -s "$size" -t "$timestep" -n 5
