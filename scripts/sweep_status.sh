#!/bin/bash
# Status of the submitted sweep / Stage-3 allocations listed in logs/sweep_jobs.tsv (jobid<TAB>job dir name).
#   bash scripts/sweep_status.sh            # table: state, node, finished/total trials
cd "$(dirname "$0")/.."
ROOT="${DNAH_ROOT:-$(sed -n 's/^root: *//p' configs/paths.yaml | sed 's/ *#.*//')}"
J="$ROOT/jobs"
printf "%-10s %-28s %-12s %-12s %s\n" JOBID JOB STATE NODE TRIALS
while IFS=$'\t' read -r id name; do
  st=$(sacct -n -X -j "$id" -o State%14 2>/dev/null | head -1 | tr -d ' ')
  node=$(sacct -n -X -j "$id" -o NodeList%14 2>/dev/null | head -1 | tr -d ' ')
  total=$(python3 -c "import yaml,sys;c=yaml.safe_load(open('$J/$name/harbor_config.yaml'));print(sum(len(d.get('task_names') or []) for d in c['datasets'])*len(c['agents'])*c['n_attempts'])" 2>/dev/null)
  done_=$(ls "$J/$name"/*/result.json 2>/dev/null | wc -l)
  printf "%-10s %-28s %-12s %-12s %s/%s\n" "$id" "$name" "$st" "$node" "$done_" "$total"
done < logs/sweep_jobs.tsv
