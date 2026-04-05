#!/usr/bin/env bash
set -euo pipefail

# ── deploy.sh ────────────────────────────────────────────────────────────
# Memory Pro System — full local deployment with verification.
#
# Ensures every module is on the target version, all Python modules compile,
# the memory server starts cleanly, the Gateway reloads the plugin, and
# every critical HTTP endpoint returns a healthy response.
#
# Usage:
#   ./scripts/deploy.sh                  # deploy current HEAD as-is
#   ./scripts/deploy.sh 0.0.12           # bump version → deploy
#   ./scripts/deploy.sh 0.0.12 --push    # bump + deploy + git push
#   ./scripts/deploy.sh --dry-run        # show what would happen, touch nothing
#   ./scripts/deploy.sh --status         # quick health report, no changes
# ─────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
PKG="$ROOT/plugin/package.json"
PORT=18790
GW_PORT=18789
LAUNCHD_LABEL="ai.openclaw.gateway"
PLIST="$HOME/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"
LOG_FILE="$ROOT/deploy.log"

# ── Colors ───────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; CYN='\033[0;36m'; RST='\033[0m'

_ok()   { printf "  ${GRN}✓${RST} %s\n" "$*"; }
_warn() { printf "  ${YLW}⚠${RST} %s\n" "$*"; }
_fail() { printf "  ${RED}✗${RST} %s\n" "$*"; }
_info() { printf "  ${CYN}→${RST} %s\n" "$*"; }
_hdr()  { printf "\n${CYN}── %s ──${RST}\n" "$*"; }

ERRORS=0
_assert() {
  if eval "$1"; then _ok "$2"; else _fail "$2 — $3"; ERRORS=$((ERRORS + 1)); fi
}

# ── Parse args ───────────────────────────────────────────────────────────
NEW_VERSION=""
DO_PUSH=false
DRY_RUN=false
STATUS_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --push)    DO_PUSH=true ;;
    --dry-run) DRY_RUN=true ;;
    --status)  STATUS_ONLY=true ;;
    -h|--help)
      echo "Usage: deploy.sh [VERSION] [--push] [--dry-run] [--status]"
      exit 0 ;;
    *)
      if [[ "$arg" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        NEW_VERSION="$arg"
      else
        echo "Unknown arg: $arg" >&2; exit 1
      fi ;;
  esac
done

current_version() {
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "$PKG"
}
VERSION="${NEW_VERSION:-$(current_version)}"

# ── All Python modules that the server loads (direct + transitive) ───────
CORE_PYTHON_MODULES=(
  memory_server
  memory_hub
  context_composer
  llm_client
  reranker
  shared_embedder
  memory_security
  bm25
)
PACKAGE_MODULES=(
  memora
  msa
  chronos
  second_brain
  skill_registry
)
# Specific submodules with significant logic
SUBMODULES=(
  memora.vectorstore
  memora.digest
  memora.config
  msa.system
  msa.router
  msa.memory_bank
  chronos.learner
  chronos.consolidator
  chronos.system
  second_brain.collision
  second_brain.inference
  second_brain.knowledge_graph
  second_brain.relation_extractor
  second_brain.bridge
  second_brain.config
  second_brain.tracker
  second_brain.skill_proposer
  skill_registry.registry
)

# ── Version files that must match ────────────────────────────────────────
VERSION_FILES=(
  "plugin/package.json"
  "plugin/openclaw.plugin.json"
  "plugin/skills/openclaw-memory-pro/SKILL.md"
)

# ── HTTP endpoints to probe after deploy ─────────────────────────────────
HEALTH_ENDPOINTS=(
  "GET  /health           (no-auth)"
  "GET  /status           (auth)"
  "POST /recall           (auth)"
  "GET  /skills/stats     (auth)"
)

# ════════════════════════════════════════════════════════════════════════
# --status mode: quick health report
# ════════════════════════════════════════════════════════════════════════
if $STATUS_ONLY; then
  _hdr "Memory Pro System — Status Report"

  _info "Version in package.json: $(current_version)"

  # Server process
  SRV_PID=$(lsof -t -i :"$PORT" 2>/dev/null || true)
  if [[ -n "$SRV_PID" ]]; then
    _ok "Memory server running (PID $SRV_PID, port $PORT)"
  else
    _fail "Memory server NOT running on port $PORT"
  fi

  # Gateway process
  GW_PID=$(pgrep -f 'openclaw-gateway' 2>/dev/null || true)
  if [[ -n "$GW_PID" ]]; then
    _ok "Gateway running (PID $GW_PID)"
  else
    _fail "Gateway NOT running"
  fi

  # Health endpoint
  HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "000")
  if [[ "$HEALTH" == "200" ]]; then
    _ok "/health → 200"
  else
    _fail "/health → $HEALTH"
  fi

  # Auth token
  AUTH_TOKEN=$(cd "$ROOT" && python3 -c "from memory_security import load_auth_token; print(load_auth_token() or '')" 2>/dev/null || true)
  if [[ -n "$AUTH_TOKEN" ]]; then
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $AUTH_TOKEN" "http://127.0.0.1:$PORT/status" 2>/dev/null || echo "000")
    if [[ "$STATUS_CODE" == "200" ]]; then
      _ok "/status → 200 (auth OK)"
    else
      _fail "/status → $STATUS_CODE (auth problem)"
    fi
  else
    _warn "No auth token found — skipping auth endpoints"
  fi

  # Gateway log version
  GW_LOG="$HOME/.openclaw/logs/gateway.log"
  if [[ -f "$GW_LOG" ]]; then
    LOADED_VER=$(grep -o 'Memory Pro plugin v[0-9.]*' "$GW_LOG" | tail -1 || true)
    if [[ -n "$LOADED_VER" ]]; then
      _info "Gateway last loaded: $LOADED_VER"
    fi
  fi

  # LLM_PROVIDER
  if [[ -n "$SRV_PID" ]]; then
    LLM_P=$(ps -p "$SRV_PID" -E 2>/dev/null | tr ' ' '\n' | grep 'LLM_PROVIDER=' || echo "not set")
    _info "Memory server LLM_PROVIDER: $LLM_P"
  fi

  echo ""
  exit 0
fi

# ════════════════════════════════════════════════════════════════════════
# Main deployment flow
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Memory Pro System — Deploy v$VERSION"
if $DRY_RUN; then
echo "║  MODE: DRY RUN (no changes will be made)            "
fi
echo "╚══════════════════════════════════════════════════════╝"

# ── Step 1: Version bump ─────────────────────────────────────────────
_hdr "Step 1/7 — Version Sync"

if [[ -n "$NEW_VERSION" ]]; then
  if $DRY_RUN; then
    _info "Would bump version to $NEW_VERSION via bump-version.sh"
  else
    bash "$SCRIPTS/bump-version.sh" "$NEW_VERSION"
  fi
else
  _info "No version arg — keeping current $(current_version)"
fi

# Verify version consistency across all files
for vf in "${VERSION_FILES[@]}"; do
  FPATH="$ROOT/$vf"
  if [[ ! -f "$FPATH" ]]; then
    _fail "Missing: $vf"
    ERRORS=$((ERRORS + 1))
    continue
  fi
  case "$vf" in
    *.json)
      FILE_VER=$(python3 -c "import json; print(json.load(open('$FPATH'))['version'])")
      ;;
    *.md)
      FILE_VER=$(grep -m1 '^version:' "$FPATH" | awk '{print $2}')
      ;;
  esac
  if [[ "$FILE_VER" == "$VERSION" ]]; then
    _ok "$vf → $FILE_VER"
  else
    _fail "$vf → $FILE_VER (expected $VERSION)"
    ERRORS=$((ERRORS + 1))
  fi
done

# Check index.ts reads version dynamically
if grep -q 'PKG.version' "$ROOT/plugin/index.ts" 2>/dev/null; then
  _ok "plugin/index.ts reads version from package.json (dynamic)"
else
  _fail "plugin/index.ts may have hardcoded version!"
  ERRORS=$((ERRORS + 1))
fi

# ── Step 2: Git state ────────────────────────────────────────────────
_hdr "Step 2/7 — Git State"

cd "$ROOT"
GIT_DIRTY=$(git status --porcelain 2>/dev/null | grep -v '??' | wc -l | tr -d ' ')
GIT_UNTRACKED=$(git status --porcelain 2>/dev/null | grep '^??' | wc -l | tr -d ' ')
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

_info "Branch: $GIT_BRANCH @ $GIT_SHA"

if [[ "$GIT_DIRTY" -gt 0 ]]; then
  _warn "$GIT_DIRTY uncommitted changes — consider committing first"
else
  _ok "Working tree clean"
fi
if [[ "$GIT_UNTRACKED" -gt 0 ]]; then
  _warn "$GIT_UNTRACKED untracked files"
fi

# ── Step 3: Python module compilation check ──────────────────────────
_hdr "Step 3/7 — Python Module Compilation ($(( ${#CORE_PYTHON_MODULES[@]} + ${#PACKAGE_MODULES[@]} + ${#SUBMODULES[@]} )) modules)"

cd "$ROOT"
_MODULE_ERRORS=0

for mod in "${CORE_PYTHON_MODULES[@]}"; do
  if python3 -c "import ast; ast.parse(open('${mod}.py').read())" 2>/dev/null; then
    _ok "$mod.py — syntax OK"
  else
    _fail "$mod.py — SYNTAX ERROR"
    _MODULE_ERRORS=$((_MODULE_ERRORS + 1))
  fi
done

for pkg in "${PACKAGE_MODULES[@]}"; do
  if [[ -d "$pkg" ]]; then
    PY_COUNT=$(find "$pkg" -name '*.py' | wc -l | tr -d ' ')
    FAIL_COUNT=0
    for pyf in $(find "$pkg" -name '*.py'); do
      if ! python3 -c "import ast; ast.parse(open('$pyf').read())" 2>/dev/null; then
        _fail "$pyf — SYNTAX ERROR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
      fi
    done
    if [[ "$FAIL_COUNT" -eq 0 ]]; then
      _ok "$pkg/ — all $PY_COUNT files OK"
    else
      _MODULE_ERRORS=$((_MODULE_ERRORS + FAIL_COUNT))
    fi
  else
    _fail "Package directory missing: $pkg/"
    _MODULE_ERRORS=$((_MODULE_ERRORS + 1))
  fi
done

# Runtime import test for critical submodules
_info "Runtime import check..."
IMPORT_RESULT=$(cd "$ROOT" && python3 -c "
import sys, importlib
errors = []
modules = [
$(printf '    \"%s\",\n' "${SUBMODULES[@]}")
]
for m in modules:
    try:
        importlib.import_module(m)
    except Exception as e:
        errors.append(f'{m}: {e}')
if errors:
    for e in errors:
        print(f'FAIL {e}')
    sys.exit(1)
else:
    print(f'OK {len(modules)} submodules imported successfully')
" 2>&1)

if [[ $? -eq 0 ]]; then
  _ok "$IMPORT_RESULT"
else
  echo "$IMPORT_RESULT" | while read -r line; do _fail "$line"; done
  _MODULE_ERRORS=$((_MODULE_ERRORS + 1))
fi

ERRORS=$((ERRORS + _MODULE_ERRORS))

if [[ "$_MODULE_ERRORS" -gt 0 ]]; then
  _fail "Module compilation failed — aborting deployment"
  if ! $DRY_RUN; then exit 1; fi
fi

# ── Step 4: Stop existing services ──────────────────────────────────
_hdr "Step 4/7 — Stop Services"

if $DRY_RUN; then
  _info "Would stop gateway and memory server"
else
  # Kill memory server on target port
  SRV_PID=$(lsof -t -i :"$PORT" 2>/dev/null || true)
  if [[ -n "$SRV_PID" ]]; then
    _info "Killing memory server PID $SRV_PID on port $PORT"
    kill "$SRV_PID" 2>/dev/null || true
    sleep 2
    # Force kill if still running
    if kill -0 "$SRV_PID" 2>/dev/null; then
      kill -9 "$SRV_PID" 2>/dev/null || true
      sleep 1
    fi
    _ok "Memory server stopped"
  else
    _info "No memory server on port $PORT"
  fi

  # Stop gateway service
  if launchctl print "gui/$(id -u)/$LAUNCHD_LABEL" &>/dev/null; then
    _info "Stopping gateway service..."
    launchctl bootout "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || true
    sleep 2
    _ok "Gateway service stopped"
  else
    GW_PID=$(pgrep -f 'openclaw-gateway' 2>/dev/null || true)
    if [[ -n "$GW_PID" ]]; then
      _info "Killing gateway PID $GW_PID"
      kill "$GW_PID" 2>/dev/null || true
      sleep 2
      _ok "Gateway process killed"
    else
      _info "Gateway not running"
    fi
  fi
fi

# ── Step 5: Prepare LaunchAgent plist with required env vars ─────────
_hdr "Step 5/7 — LaunchAgent Environment"

REQUIRED_ENV_VARS=(
  "LLM_PROVIDER=xai"
)

if $DRY_RUN; then
  _info "Would ensure LaunchAgent plist has: ${REQUIRED_ENV_VARS[*]}"
else
  # Regenerate plist via openclaw gateway install (gets latest paths).
  # install also bootstraps the service, so unload it right after
  # to apply our env patches before the real start.
  openclaw gateway install >/dev/null 2>&1 || true
  launchctl bootout "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || true
  sleep 1

  # Patch in required env vars that openclaw install doesn't know about
  for kv in "${REQUIRED_ENV_VARS[@]}"; do
    KEY="${kv%%=*}"
    VAL="${kv#*=}"
    if grep -q "<key>$KEY</key>" "$PLIST" 2>/dev/null; then
      _ok "Plist already has $KEY"
    else
      python3 -c "
import sys
plist = open(sys.argv[1]).read()
insert = f'    <key>{sys.argv[2]}</key>\n    <string>{sys.argv[3]}</string>\n    </dict>'
plist = plist.replace('    </dict>\n  </dict>', insert + '\n  </dict>')
open(sys.argv[1], 'w').write(plist)
print(f'  Injected {sys.argv[2]}={sys.argv[3]}')
" "$PLIST" "$KEY" "$VAL"
      _ok "Plist patched: $KEY=$VAL"
    fi
  done

  # Kill any server the brief install may have spawned
  STALE_PID=$(lsof -t -i :"$PORT" 2>/dev/null || true)
  if [[ -n "$STALE_PID" ]]; then
    kill "$STALE_PID" 2>/dev/null || true
    sleep 1
  fi
fi

# ── Step 6: Start services ──────────────────────────────────────────
_hdr "Step 6/7 — Start Services"

if $DRY_RUN; then
  _info "Would bootstrap gateway via launchd"
  _info "Gateway autoStart would spawn memory_server.py on port $PORT"
else
  _info "Bootstrapping gateway with patched plist..."
  launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || true
  _ok "Gateway service started"

  # Wait for memory server to be ready
  _info "Waiting for memory server on port $PORT..."
  DEADLINE=$((SECONDS + 90))
  SERVER_READY=false
  while [[ $SECONDS -lt $DEADLINE ]]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "000")
    if [[ "$HTTP_CODE" == "200" ]]; then
      SERVER_READY=true
      break
    fi
    sleep 2
  done

  if $SERVER_READY; then
    _ok "Memory server ready"
  else
    _fail "Memory server did not become ready within 90s"
    ERRORS=$((ERRORS + 1))
  fi
fi

# ── Step 7: End-to-end verification ─────────────────────────────────
_hdr "Step 7/7 — End-to-End Verification"

if $DRY_RUN; then
  _info "Would test: health, status (auth), recall, skills/stats, collide"
  _info "Would verify Gateway log shows v$VERSION"
  _info "Would verify LLM_PROVIDER env"
else
  AUTH_TOKEN=$(cd "$ROOT" && python3 -c "from memory_security import load_auth_token; print(load_auth_token() or '')" 2>/dev/null || true)

  # 7a. /health (no auth)
  HEALTH_BODY=$(curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null || echo '{}')
  HEALTH_STATUS=$(echo "$HEALTH_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
  _assert "[[ '$HEALTH_STATUS' == 'ok' ]]" "/health → status=ok" "got '$HEALTH_STATUS'"

  # 7b. /status (auth)
  if [[ -n "$AUTH_TOKEN" ]]; then
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $AUTH_TOKEN" "http://127.0.0.1:$PORT/status" 2>/dev/null || echo "000")
    _assert "[[ '$STATUS_CODE' == '200' ]]" "/status → 200 (auth OK)" "got $STATUS_CODE"
  else
    _warn "No auth token — skipping authed endpoints"
    ERRORS=$((ERRORS + 1))
  fi

  # 7c. /recall (auth, POST)
  if [[ -n "$AUTH_TOKEN" ]]; then
    RECALL_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
      -H "Authorization: Bearer $AUTH_TOKEN" -H "Content-Type: application/json" \
      -d '{"query":"deploy verification test","top_k":1}' \
      "http://127.0.0.1:$PORT/recall" 2>/dev/null || echo "000")
    _assert "[[ '$RECALL_CODE' == '200' ]]" "/recall → 200" "got $RECALL_CODE"
  fi

  # 7d. /skills/stats (auth)
  if [[ -n "$AUTH_TOKEN" ]]; then
    SKILLS_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
      -H "Authorization: Bearer $AUTH_TOKEN" \
      "http://127.0.0.1:$PORT/skills/stats" 2>/dev/null || echo "000")
    _assert "[[ '$SKILLS_CODE' == '200' ]]" "/skills/stats → 200" "got $SKILLS_CODE"
  fi

  # 7e. Gateway log confirms correct plugin version
  GW_LOG="$HOME/.openclaw/logs/gateway.log"
  if [[ -f "$GW_LOG" ]]; then
    LOADED_VER=$(grep -o "Memory Pro plugin v[0-9.]*" "$GW_LOG" | tail -1 | grep -o '[0-9].*' || true)
    _assert "[[ '$LOADED_VER' == '$VERSION' ]]" \
      "Gateway loaded plugin v$VERSION" \
      "got v$LOADED_VER"
  else
    _warn "Gateway log not found at $GW_LOG"
  fi

  # 7f. Memory server has correct LLM_PROVIDER
  SRV_PID=$(lsof -t -i :"$PORT" 2>/dev/null || true)
  if [[ -n "$SRV_PID" ]]; then
    LLM_P=$(ps -p "$SRV_PID" -E 2>/dev/null | tr ' ' '\n' | grep 'LLM_PROVIDER=' | head -1 || echo "NOT_SET")
    _assert "[[ '$LLM_P' == 'LLM_PROVIDER=xai' ]]" \
      "Memory server LLM_PROVIDER=xai" \
      "got $LLM_P"
  fi

  # 7g. server.pid consistency
  if [[ -n "$SRV_PID" ]]; then
    echo "$SRV_PID" > "$ROOT/server.pid"
    _ok "server.pid updated to $SRV_PID"
  fi

  # 7h. Subsystem counts (informational)
  if [[ -n "$AUTH_TOKEN" ]]; then
    STATUS_JSON=$(curl -s -H "Authorization: Bearer $AUTH_TOKEN" "http://127.0.0.1:$PORT/status" 2>/dev/null || echo '{}')
    MEMORA_N=$(echo "$STATUS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('systems',{}).get('memora',{}).get('entries',0))" 2>/dev/null || echo "?")
    MSA_N=$(echo "$STATUS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('systems',{}).get('msa',{}).get('document_count',0))" 2>/dev/null || echo "?")
    SKILLS_JSON=$(curl -s -H "Authorization: Bearer $AUTH_TOKEN" "http://127.0.0.1:$PORT/skills/stats" 2>/dev/null || echo '{}')
    SKILL_N=$(echo "$SKILLS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo "?")

    echo ""
    _info "Subsystem stats: Memora=$MEMORA_N entries, MSA=$MSA_N docs, Skills=$SKILL_N"
  fi
fi

# ── Step 8 (optional): Git push ──────────────────────────────────────
if [[ -n "$NEW_VERSION" ]] && $DO_PUSH && ! $DRY_RUN; then
  _hdr "Bonus — Git Push"
  cd "$ROOT"
  git add -A
  DIRTY_NOW=$(git status --porcelain | grep -v '??' | wc -l | tr -d ' ')
  if [[ "$DIRTY_NOW" -gt 0 ]]; then
    git commit -m "chore: deploy v$VERSION"
  fi
  if ! git tag -l "v$VERSION" | grep -q "v$VERSION"; then
    git tag -a "v$VERSION" -m "v$VERSION"
  fi
  git push origin "$GIT_BRANCH" && git push origin "v$VERSION" --force
  _ok "Pushed $GIT_BRANCH + tag v$VERSION"
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
if [[ "$ERRORS" -eq 0 ]]; then
  printf "║  ${GRN}Deploy v$VERSION — ALL CHECKS PASSED${RST}              \n"
else
  printf "║  ${RED}Deploy v$VERSION — $ERRORS ERROR(S) DETECTED${RST}             \n"
fi
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Append to deploy log
if ! $DRY_RUN; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] v$VERSION sha=$GIT_SHA errors=$ERRORS branch=$GIT_BRANCH" >> "$LOG_FILE"
fi

exit "$ERRORS"
