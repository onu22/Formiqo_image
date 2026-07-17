#!/usr/bin/env bash
# Compute the next autonomous harness action for /run-mvp.
# Continues past G4 into post-MVP work (E5, then stretch E7).
# Usage: ./scripts/harness-next.sh [--json]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GATES="$ROOT/harness/gates"
SPRINTS="$ROOT/harness/sprints"
JSON=false
[[ "${1:-}" == "--json" ]] && JSON=true

gate_status() {
  local id="$1"
  local f="$GATES/${id}.md"
  if [[ ! -f "$f" ]]; then
    echo "PENDING"
    return
  fi
  local raw
  raw=$(grep -m1 '^\*\*Status:\*\*' "$f" 2>/dev/null | sed 's/^\*\*Status:\*\* //' || echo "PENDING")
  if [[ "$raw" == *"QA APPROVED"* ]]; then
    echo "QA APPROVED"
  elif [[ "$raw" == *"APPROVED WITH CONDITIONS"* ]] || [[ "$raw" == *"APPROVED"* ]]; then
    echo "APPROVED"
  elif [[ "$raw" == *"BLOCKED"* ]]; then
    echo "BLOCKED"
  else
    echo "PENDING"
  fi
}

gate_open() {
  local st="$1"
  [[ "$st" == "APPROVED" ]] || [[ "$st" == "QA APPROVED" ]]
}

sprint_file() {
  if [[ -L "$SPRINTS/CURRENT" ]]; then
    local target
    target=$(readlink "$SPRINTS/CURRENT")
    echo "$SPRINTS/$target"
  elif [[ -f "$SPRINTS/sprint-001.md" ]]; then
    echo "$SPRINTS/sprint-001.md"
  else
    ls "$SPRINTS"/sprint-*.md 2>/dev/null | tail -1
  fi
}

sprint_todo_count() {
  local sf
  sf=$(sprint_file)
  [[ -f "$sf" ]] || { echo 0; return; }
  grep -c '| TODO |' "$sf" 2>/dev/null || true
}

epic_todo() {
  local epic="$1"
  local sf
  sf=$(sprint_file)
  [[ -f "$sf" ]] || return 1
  grep -E "\\| T[0-9]+ \\| ${epic} \\|" "$sf" 2>/dev/null | grep -q '| TODO |'
}

epic_present() {
  local epic="$1"
  local sf
  sf=$(sprint_file)
  [[ -f "$sf" ]] || return 1
  grep -qE "\\| T[0-9]+ \\| ${epic} \\|" "$sf" 2>/dev/null
}

epic_done() {
  local epic="$1"
  local sf
  sf=$(sprint_file)
  [[ -f "$sf" ]] || return 1
  epic_present "$epic" || return 1
  local todo
  todo=$(grep -E "\\| T[0-9]+ \\| ${epic} \\|" "$sf" 2>/dev/null | grep -c '| TODO |' || true)
  local blocked
  blocked=$(grep -E "\\| T[0-9]+ \\| ${epic} \\|" "$sf" 2>/dev/null | grep -c '| BLOCKED |' || true)
  local inprog
  inprog=$(grep -E "\\| T[0-9]+ \\| ${epic} \\|" "$sf" 2>/dev/null | grep -cE '\| (IN_PROGRESS|IN_REVIEW) \|' || true)
  [[ "$todo" -eq 0 ]] && [[ "$blocked" -eq 0 ]] && [[ "$inprog" -eq 0 ]]
}

G1=$(gate_status G1-architecture)
G2=$(gate_status G2-parity)
G3=$(gate_status G3-security)
G4=$(gate_status G4-ship)

ACTION=""
TARGET=""
AGENT=""
REASON=""
PARALLEL="no"
EXTRA=""

# Hard gate blocks
if [[ "$G1" == "BLOCKED" ]]; then
  ACTION="blocked"
  TARGET="G1"
  REASON="G1 architecture gate BLOCKED — human decision required"
  AGENT="solution-architect"
elif [[ "$G2" == "BLOCKED" ]]; then
  ACTION="blocked"
  TARGET="G2"
  REASON="G2 parity gate BLOCKED"
  AGENT="qa-specialist"
elif [[ "$G3" == "BLOCKED" ]]; then
  ACTION="blocked"
  TARGET="G3"
  REASON="G3 security gate BLOCKED"
  AGENT="security-reviewer"
elif [[ "$G4" == "BLOCKED" ]]; then
  ACTION="blocked"
  TARGET="G4"
  REASON="G4 ship gate BLOCKED"
  AGENT="qa-specialist"
fi

# --- Pre-G4 MVP pipeline ---
if [[ -z "$ACTION" ]] && [[ "$G4" != "QA APPROVED" ]]; then
  if ! gate_open "$G1"; then
    ACTION="gate"
    TARGET="G1"
    AGENT="solution-architect"
    REASON="G1 must be signed before E1+ implementation"
  elif epic_todo "E1" 2>/dev/null; then
    ACTION="epic"
    TARGET="E1"
    AGENT="backend-developer"
    REASON="E1 sprint tasks still TODO"
  elif epic_todo "E3" 2>/dev/null; then
    ACTION="epic"
    TARGET="E3"
    AGENT="backend-developer"
    REASON="E3 API tasks still TODO"
  elif epic_todo "E2" 2>/dev/null; then
    ACTION="epic"
    TARGET="E2"
    AGENT="backend-developer"
    REASON="E2 stamping parity — critical path to G2 and E5"
    if epic_todo "E4" 2>/dev/null || epic_todo "E6" 2>/dev/null; then
      PARALLEL="yes"
      EXTRA="E4,E6"
    fi
  elif ! gate_open "$G2" && epic_done "E2"; then
    ACTION="gate"
    TARGET="G2"
    AGENT="qa-specialist"
    REASON="E2 complete — run parity golden tests (G2)"
  elif epic_todo "E4" 2>/dev/null; then
    ACTION="epic"
    TARGET="E4"
    AGENT="llm-engineer"
    REASON="E4 grounding accuracy"
    if epic_todo "E6" 2>/dev/null; then
      PARALLEL="yes"
      EXTRA="E6"
    fi
  elif epic_todo "E6" 2>/dev/null; then
    if gate_open "$G3"; then
      ACTION="epic"
      TARGET="E6"
      AGENT="frontend-developer"
      REASON="E6 review UI — G3 approved for live API"
    else
      ACTION="gate"
      TARGET="G3"
      AGENT="security-reviewer"
      REASON="E6 live integration blocked until G3"
    fi
  elif epic_todo "E5" 2>/dev/null; then
    if gate_open "$G2" && epic_done "E4"; then
      ACTION="epic"
      TARGET="E5"
      AGENT="llm-engineer"
      REASON="E5 QA refine loop — G2 and E4 prerequisites met"
    elif ! gate_open "$G2"; then
      ACTION="gate"
      TARGET="G2"
      AGENT="qa-specialist"
      REASON="E5 blocked until G2 QA APPROVED"
    else
      ACTION="epic"
      TARGET="E4"
      AGENT="llm-engineer"
      REASON="E5 blocked until E4 complete"
    fi
  elif ! gate_open "$G4" && epic_done "E6"; then
    ACTION="gate"
    TARGET="G4"
    AGENT="qa-specialist"
    REASON="E6 complete — ship review (G4)"
  else
    TODO_COUNT=$(sprint_todo_count)
    if [[ "$TODO_COUNT" -eq 0 ]]; then
      ACTION="sprint-plan"
      TARGET="sprint-next"
      AGENT="product-manager"
      REASON="Sprint backlog empty — plan next sprint or close MVP gaps"
    else
      ACTION="blocked"
      TARGET="unknown"
      REASON="No automatic next action — run ./scripts/harness-status.sh and inspect sprint"
      AGENT="product-manager"
    fi
  fi
fi

# --- Post-G4: M4 (E5) then stretch M5 (E7) — keeps automation running ---
if [[ -z "$ACTION" ]] && [[ "$G4" == "QA APPROVED" ]]; then
  if epic_todo "E5" 2>/dev/null; then
    ACTION="epic"
    TARGET="E5"
    AGENT="llm-engineer"
    REASON="Post-MVP M4 — E5 vision QA refine loop"
  elif epic_todo "E7" 2>/dev/null; then
    ACTION="epic"
    TARGET="E7"
    AGENT="llm-engineer"
    REASON="Post-MVP M5 stretch — E7 template memory"
  elif ! epic_present "E5" 2>/dev/null; then
    ACTION="sprint-plan"
    TARGET="sprint-002"
    AGENT="product-manager"
    REASON="G4 shipped — plan post-MVP sprint with E5 (required) and E7 (stretch) TODO rows"
  elif epic_done "E5" && epic_present "E7" && ! epic_done "E7"; then
    ACTION="blocked"
    TARGET="E7"
    REASON="E7 present but not TODO/DONE — resolve IN_PROGRESS or BLOCKED rows"
    AGENT="product-manager"
  elif epic_done "E5" && (! epic_present "E7" 2>/dev/null || epic_done "E7"); then
    ACTION="complete"
    TARGET="post-MVP"
    AGENT="product-manager"
    REASON="G4 shipped; E5 done$([ epic_present E7 ] && echo '; E7 done' || echo '; E7 not queued') — post-MVP complete"
  else
    TODO_COUNT=$(sprint_todo_count)
    if [[ "$TODO_COUNT" -gt 0 ]]; then
      ACTION="blocked"
      TARGET="sprint"
      REASON="Post-MVP sprint has non-E5/E7 TODO rows — inspect CURRENT sprint"
      AGENT="product-manager"
    else
      ACTION="blocked"
      TARGET="E5"
      REASON="E5 present but not done and no TODO — check IN_PROGRESS/BLOCKED rows"
      AGENT="product-manager"
    fi
  fi
fi

# Fix REASON line for complete (avoid nested command issues in echo)
if [[ "$ACTION" == "complete" ]] && [[ "$TARGET" == "post-MVP" ]]; then
  if epic_present "E7" 2>/dev/null && epic_done "E7"; then
    REASON="G4 shipped; E5 and E7 complete — post-MVP done"
  else
    REASON="G4 shipped; E5 done — post-MVP complete (E7 not queued)"
  fi
fi

if $JSON; then
  printf '{"action":"%s","target":"%s","agent":"%s","reason":"%s","parallel":"%s","extra":"%s","gates":{"G1":"%s","G2":"%s","G3":"%s","G4":"%s"}}\n' \
    "$ACTION" "$TARGET" "$AGENT" "$REASON" "$PARALLEL" "$EXTRA" "$G1" "$G2" "$G3" "$G4"
else
  echo "=== Harness Next Action (/run-mvp) ==="
  echo
  echo "Gates: G1=$G1  G2=$G2  G3=$G3  G4=$G4"
  echo
  echo "ACTION=$ACTION"
  echo "TARGET=$TARGET"
  echo "AGENT=$AGENT"
  echo "PARALLEL=$PARALLEL"
  [[ -n "$EXTRA" ]] && echo "ALSO_CONSIDER=$EXTRA"
  echo "REASON=$REASON"
  echo
  echo "Run: /run-mvp"
fi
