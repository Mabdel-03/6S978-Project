#!/bin/bash
# Resubmit rendered job dirs (by name) through serve_and_run.sbatch; an existing job resumes and re-runs
# infrastructure failures (default DNAH_RESUME_FILTER in the sbatch). Updates logs/sweep_jobs.tsv.
#   bash scripts/resubmit.sh exp_a_shared-c2of8 [more job names...]
# Partitions / exclusions: DNAH_PARTITIONS, DNAH_EXCLUDE (default: node3904, recovering from the sandbox leak).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${DNAH_ROOT:-$(sed -n 's/^root: *//p' configs/paths.yaml | sed 's/ *#.*//')}"
EXC="${DNAH_EXCLUDE-node3904}"
for name in "$@"; do
  d="$ROOT/jobs/$name"; [ -d "$d" ] || { echo "no job dir $d"; exit 2; }
  case "$name" in
    sft_teacher*) P="${DNAH_PARTITIONS:-ou_bcs_low,pi_tpoggio,ou_bcs_normal}"; R=(--gres=gpu:2 -c 28 --mem=160G --time=12:00:00) ;;
    *)            P="${DNAH_PARTITIONS:-ou_bcs_normal,pi_tpoggio,ou_bcs_low}"; R=(--gres=gpu:1 -c 28 --mem=96G --time=20:00:00) ;;
  esac
  id=$(sbatch --parsable -J "dnah-$name" -p "$P" ${EXC:+--exclude=$EXC} "${R[@]}" -o "logs/slurm-%x-%j.out" slurm/serve_and_run.sbatch "$d")
  if grep -q -P "\t$name\$" logs/sweep_jobs.tsv; then sed -i -E "s/^[0-9]+\t$name\$/$id\t$name/" logs/sweep_jobs.tsv; else printf "%s\t%s\n" "$id" "$name" >> logs/sweep_jobs.tsv; fi
  echo "$name -> $id"
done
