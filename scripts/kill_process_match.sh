#!/usr/bin/env bash

# Find processes whose command name contains user-supplied substrings
# and kill them with SIGKILL unless dry-run mode is enabled.

set -u

DRY_RUN=0
MATCH_CSV=""

usage() {
  cat <<'EOF'
Usage: kill_process_match.sh [OPTIONS]

Options:
  -m, --match     Comma-delimited substrings to match in process names (required)
  -n, --dry-run   Show matching processes without killing them
  -h, --help      Show this help message

Example:
  kill_process_match.sh --match abc,rf,80211 --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--match)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      MATCH_CSV="$2"
      shift 2
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MATCH_CSV" ]]; then
  echo "--match is required." >&2
  usage >&2
  exit 1
fi

IFS=',' read -r -a raw_patterns <<< "$MATCH_CSV"
patterns=()
for p in "${raw_patterns[@]}"; do
  # Trim leading/trailing spaces and normalize to lowercase.
  trimmed="${p#${p%%[![:space:]]*}}"
  trimmed="${trimmed%${trimmed##*[![:space:]]}}"
  if [[ -n "$trimmed" ]]; then
    patterns+=("${trimmed,,}")
  fi
done

if [[ ${#patterns[@]} -eq 0 ]]; then
  echo "--match must contain at least one non-empty token." >&2
  usage >&2
  exit 1
fi

matches=()

while read -r pid comm; do
  name="${comm,,}"
  for pat in "${patterns[@]}"; do
    if [[ "$name" == *"$pat"* ]]; then
      matches+=("$pid:$comm")
      break
    fi
  done
done < <(ps -eo pid=,comm=)

if [[ ${#matches[@]} -eq 0 ]]; then
  echo "No matching processes found."
  exit 0
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run mode: these processes would be killed with -9:"
  for entry in "${matches[@]}"; do
    pid="${entry%%:*}"
    comm="${entry#*:}"
    echo "  PID $pid  NAME $comm"
  done
  exit 0
fi

echo "Killing matching processes with -9:"
for entry in "${matches[@]}"; do
  pid="${entry%%:*}"
  comm="${entry#*:}"
  if kill -9 "$pid" 2>/dev/null; then
    echo "  Killed PID $pid  NAME $comm"
  else
    echo "  Failed PID $pid  NAME $comm" >&2
  fi
done
