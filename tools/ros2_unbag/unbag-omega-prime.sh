#!/usr/bin/env bash
set -euo pipefail

# Config via env vars
OP_DATA=${OP_DATA:-/data}
OP_OUT=${OP_OUT:-/out}
OP_TOPIC=${OP_TOPIC:-}
OP_FORMAT=${OP_FORMAT:-omega-prime/MCAP}
OP_ROUTINE=${OP_ROUTINE:-/opt/routines/object_list_to_omega_prime.py}
OP_GLOB=${OP_GLOB:-"*.mcap *.db3"}
OP_EXTRA_ARGS=${OP_EXTRA_ARGS:-}

if [[ -n "${OP_DEBUG:-}" ]]; then
  set -x
fi

mkdir -p "${OP_OUT}"

# Discover bags
shopt -s nullglob
declare -a BAGS=()
for pat in ${OP_GLOB}; do
  for f in "${OP_DATA}"/${pat}; do
    BAGS+=("${f}")
  done
done

if (( ${#BAGS[@]} == 0 )); then
  echo "[unbag-omega-prime] No bag files found in ${OP_DATA} for patterns: ${OP_GLOB}" >&2
  echo "[unbag-omega-prime] Mount your data directory to ${OP_DATA} (e.g., -v /path/to/bags:/data)." >&2
  exit 1
fi

if [[ -z "${OP_TOPIC}" ]]; then
  echo "[unbag-omega-prime] OP_TOPIC is not set. Please set the ObjectList topic, e.g.:" >&2
  echo "  -e OP_TOPIC=/perception/object_list" >&2
  exit 2
fi

echo "[unbag-omega-prime] Processing ${#BAGS[@]} bag(s) from ${OP_DATA} -> ${OP_OUT}"

for bag in "${BAGS[@]}"; do
  echo "[unbag-omega-prime] Exporting: ${bag} (topic: ${OP_TOPIC})"
  ros2 unbag "${bag}" \
    --export "${OP_TOPIC}:${OP_FORMAT}" \
    --use-routine "${OP_ROUTINE}" \
    --output "${OP_OUT}" ${OP_EXTRA_ARGS}
done

echo "[unbag-omega-prime] Done. Outputs in ${OP_OUT}"
