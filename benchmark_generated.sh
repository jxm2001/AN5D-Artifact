#!/usr/bin/env bash

set -uo pipefail

usage()
{
    cat <<'EOF'
Usage: ./benchmark_generated.sh --gpu a100|h100 [OPTIONS] [KERNEL ...]

Compile and benchmark every generated configuration for all kernels, or only
the kernels named as positional arguments.

Options:
  --gpu GPU          Target GPU: a100 (sm_80) or h100 (sm_90), required
  --jobs N           Maximum parallel nvcc processes (default: nproc)
  --compile-only     Compile without checking for a GPU or running binaries
  --results-dir DIR  Output directory (default: results)
  -h, --help         Show this help

Examples:
  ./benchmark_generated.sh --gpu a100
  ./benchmark_generated.sh --gpu h100 --jobs 4 j2d5pt star3d1r
  ./benchmark_generated.sh --gpu a100 --compile-only j2d5pt
EOF
}

die()
{
    echo "Error: $*" >&2
    exit 1
}

gpu=""
jobs="$(nproc 2>/dev/null || echo 1)"
compile_only=0
results_root="results"
kernels=()

while (($#)); do
    case "$1" in
        --gpu)
            (($# >= 2)) || die "--gpu requires an argument"
            gpu="$2"
            shift 2
            ;;
        --jobs)
            (($# >= 2)) || die "--jobs requires an argument"
            jobs="$2"
            shift 2
            ;;
        --compile-only)
            compile_only=1
            shift
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
            kernels+=("$1")
            shift
            ;;
    esac
done

case "$gpu" in
    a100) arch="80" ;;
    h100) arch="90" ;;
    "") die "--gpu is required" ;;
    *) die "unsupported GPU '$gpu' (expected a100 or h100)" ;;
esac

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"

command -v nvcc >/dev/null 2>&1 || die "nvcc was not found in PATH"
if ((compile_only == 0)); then
    command -v nvidia-smi >/dev/null 2>&1 ||
        die "nvidia-smi was not found; use --compile-only on a machine without a GPU"
    nvidia-smi -L >/dev/null 2>&1 ||
        die "no accessible NVIDIA GPU was found; use --compile-only to skip execution"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir" || exit 1

available_kernels=()
while IFS= read -r source; do
    config="${source##*/}"
    config="${config%_host.cu}"
    if [[ "$config" =~ ^(.+)-[0-9]+(x[0-9]+)?-[0-9]+-[0-9]+$ ]]; then
        available_kernels+=("${BASH_REMATCH[1]}")
    fi
done < <(find compiled/float -maxdepth 1 -type f -name '*_host.cu' | sort)
mapfile -t available_kernels < <(printf '%s\n' "${available_kernels[@]}" | sort -u)

if ((${#kernels[@]} == 0)); then
    kernels=("${available_kernels[@]}")
else
    for kernel in "${kernels[@]}"; do
        found=0
        for available in "${available_kernels[@]}"; do
            [[ "$kernel" == "$available" ]] && found=1 && break
        done
        ((found == 1)) || die "unknown kernel '$kernel'; available kernels: ${available_kernels[*]}"
    done
fi

selected()
{
    local candidate=$1 kernel
    for kernel in "${kernels[@]}"; do
        [[ "$candidate" == "$kernel" ]] && return 0
    done
    return 1
}

result_dir="$results_root/$gpu"
bin_dir="$result_dir/bin"
log_dir="$result_dir/logs"
mkdir -p "$bin_dir" "$log_dir"

task_file="$(mktemp)"
trap 'rm -f "$task_file"' EXIT

for precision in float double; do
    while IFS= read -r host; do
        config="${host##*/}"
        config="${config%_host.cu}"
        [[ "$config" =~ ^(.+)-([0-9]+)(x([0-9]+))?-([0-9]+)-([0-9]+)$ ]] ||
            die "cannot parse generated configuration '$config'"
        kernel="${BASH_REMATCH[1]}"
        selected "$kernel" || continue
        bs1="${BASH_REMATCH[2]}"
        bs2="${BASH_REMATCH[4]:-}"
        bt="${BASH_REMATCH[5]}"
        sl="${BASH_REMATCH[6]}"
        kernel_source="compiled/$precision/${config}_kernel.cu"
        [[ -f "$kernel_source" ]] || die "missing kernel source: $kernel_source"

        for regnum in 0 32 64 96; do
            printf '%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
                "$kernel" "$precision" "$config" "$bs1" "$bs2" "$bt" "$sl" "$regnum" "$host" \
                >> "$task_file"
        done
    done < <(find "compiled/$precision" -maxdepth 1 -type f -name '*_host.cu' | sort)
done

[[ -s "$task_file" ]] || die "no generated configurations matched the requested kernels"

compile_one()
{
    local precision=$1 config=$2 regnum=$3 host=$4
    local output="$bin_dir/$precision/reg$regnum/$config"
    local log="$log_dir/$precision/reg$regnum/${config}.compile.log"
    local status="${output}.compile_status"
    local regopt=()

    mkdir -p "$(dirname "$output")" "$(dirname "$log")"
    ((regnum > 0)) && regopt=("--maxrregcount=$regnum")

    if nvcc -D "SB_TYPE=$precision" \
        "-gencode=arch=compute_${arch},code=sm_${arch}" \
        --use_fast_math -Xptxas -v -Xcompiler -fopenmp -O3 -I. \
        "${regopt[@]}" "$host" "compiled/$precision/${config}_kernel.cu" \
        -o "$output" >"$log" 2>&1; then
        printf 'success\n' > "$status"
    else
        printf 'compile_failed\n' > "$status"
    fi
}

echo "Compiling $(wc -l < "$task_file") configurations for $gpu (sm_$arch), up to $jobs at a time..."
while IFS='|' read -r kernel precision config bs1 bs2 bt sl regnum host; do
    compile_one "$precision" "$config" "$regnum" "$host" &
    while (($(jobs -pr | wc -l) >= jobs)); do
        wait -n || true
    done
done < "$task_file"
wait

all_csv="$result_dir/all_results.csv"
best_csv="$result_dir/best_results.csv"
echo 'kernel,precision,bS1,bS2,bT,sl,regnum,gflops,ms,status,log' > "$all_csv"

total_tasks="$(wc -l < "$task_file")"
current_task=0
while IFS='|' read -r kernel precision config bs1 bs2 bt sl regnum host; do
    ((current_task += 1))
    output="$bin_dir/$precision/reg$regnum/$config"
    compile_log="$log_dir/$precision/reg$regnum/${config}.compile.log"
    run_log="$log_dir/$precision/reg$regnum/${config}.run.log"
    status_file="${output}.compile_status"
    status="$(cat "$status_file" 2>/dev/null || echo compile_failed)"
    gflops=""
    ms=""
    result_log="$compile_log"

    if [[ "$status" == success ]]; then
        if ((compile_only == 1)); then
            status="compile_only"
        else
            echo "[$current_task/$total_tasks] Running $kernel ($precision, $config, REGNUM=$regnum)"
            size=512
            [[ "$kernel" == *2d* ]] && size=16384
            if "$output" -s "$size" -t 1000 -n 5 >"$run_log" 2>&1; then
                average="$(awk '/^Average:/ {print; exit}' "$run_log")"
                gflops="$(awk '/^Average:/ {print $2; exit}' "$run_log")"
                ms="$(awk '/^Average:/ {gsub(/,/, "", $4); print $4; exit}' "$run_log")"
                if [[ -n "$average" && "$gflops" =~ ^[0-9]+([.][0-9]+)?$ &&
                      "$ms" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                    status="success"
                else
                    status="parse_failed"
                    gflops=""
                    ms=""
                fi
            else
                status="run_failed"
            fi
            result_log="$run_log"
        fi
    elif ((compile_only == 0)); then
        echo "[$current_task/$total_tasks] Skipping $kernel ($precision, $config, REGNUM=$regnum): compilation failed"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$kernel" "$precision" "$bs1" "$bs2" "$bt" "$sl" "$regnum" \
        "$gflops" "$ms" "$status" "$result_log" >> "$all_csv"
done < "$task_file"

echo 'kernel,precision,bS1,bS2,bT,sl,regnum,gflops,ms,status,log' > "$best_csv"
awk -F, '
    NR == 1 || $10 != "success" { next }
    {
        key = $1 SUBSEP $2
        if (!(key in best) || ($8 + 0) > best[key] ||
            (($8 + 0) == best[key] && ($7 + 0) < reg[key]) ||
            (($8 + 0) == best[key] && ($7 + 0) == reg[key] && $0 < row[key])) {
            best[key] = $8 + 0
            reg[key] = $7 + 0
            row[key] = $0
        }
    }
    END {
        for (key in row)
            print row[key]
    }
' "$all_csv" | sort -t, -k1,1 -k2,2 >> "$best_csv"

if ((compile_only == 1)); then
    echo "Compilation complete. Results: $all_csv"
else
    echo "Benchmark complete. All results: $all_csv"
    echo "Best results: $best_csv"
fi
