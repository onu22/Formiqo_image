#!/usr/bin/env bash
# Print Formiqo harness gate and sprint status.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GATES="$ROOT/harness/gates"
SPRINTS="$ROOT/harness/sprints"

echo "=== Formiqo Harness Status ==="
echo

echo "Gates:"
for g in G1-architecture G2-parity G3-security G4-ship; do
  f="$GATES/${g}.md"
  if [[ -f "$f" ]]; then
    status=$(grep -m1 '^\*\*Status:\*\*' "$f" 2>/dev/null | sed 's/^\*\*Status:\*\* //' || echo "unknown")
    echo "  $g: $status"
  fi
done

echo
if [[ -L "$SPRINTS/CURRENT" ]]; then
  current=$(readlink "$SPRINTS/CURRENT")
  echo "Active sprint: $current"
  sprint_file="$SPRINTS/$current"
  if [[ -f "$sprint_file" ]]; then
    goal=$(grep -m1 '^goal:' "$sprint_file" 2>/dev/null | sed 's/^goal: //' || true)
    [[ -n "$goal" ]] && echo "  Goal: $goal"
    todo=$(grep -c '| TODO |' "$sprint_file" 2>/dev/null || echo 0)
    blocked=$(grep -c '| BLOCKED |' "$sprint_file" 2>/dev/null || echo 0)
    done=$(grep -c '| DONE |' "$sprint_file" 2>/dev/null || echo 0)
    echo "  Tasks: DONE=$done TODO=$todo BLOCKED=$blocked"
  fi
else
  echo "Active sprint: (no CURRENT symlink)"
fi

echo
echo "Docs: harness/README.md"
