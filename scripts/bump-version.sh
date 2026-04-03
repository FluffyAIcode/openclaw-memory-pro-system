#!/usr/bin/env bash
set -euo pipefail

# ── bump-version.sh ─────────────────────────────────────────────────
# Single source of truth: plugin/package.json "version" field.
# This script reads or sets that version, then propagates it to every
# file that carries a version string.
#
# Usage:
#   ./scripts/bump-version.sh              # show current version
#   ./scripts/bump-version.sh 0.1.0        # set version to 0.1.0
#   ./scripts/bump-version.sh 0.1.0 --tag  # set + git commit + tag
# ─────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/plugin/package.json"

current_version() {
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "$PKG"
}

CURRENT=$(current_version)

if [[ $# -eq 0 ]]; then
  echo "Current version: $CURRENT"
  exit 0
fi

NEW_VERSION="$1"
DO_TAG="${2:-}"

if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
  echo "Error: invalid semver '$NEW_VERSION'" >&2
  exit 1
fi

if [[ "$NEW_VERSION" == "$CURRENT" ]]; then
  echo "Version is already $CURRENT — nothing to do."
  exit 0
fi

echo "Bumping version: $CURRENT → $NEW_VERSION"

# ── 1. plugin/package.json (single source of truth) ─────────────────
python3 -c "
import json, sys
path = sys.argv[1]
ver  = sys.argv[2]
data = json.load(open(path))
data['version'] = ver
json.dump(data, open(path, 'w'), indent=2, ensure_ascii=False)
print('  ✓ ' + path)
" "$PKG" "$NEW_VERSION"

# ── 2. plugin/openclaw.plugin.json ──────────────────────────────────
MANIFEST="$ROOT/plugin/openclaw.plugin.json"
if [[ -f "$MANIFEST" ]]; then
  python3 -c "
import json, sys
path = sys.argv[1]
ver  = sys.argv[2]
data = json.load(open(path))
data['version'] = ver
json.dump(data, open(path, 'w'), indent=2, ensure_ascii=False)
print('  ✓ ' + path)
" "$MANIFEST" "$NEW_VERSION"
fi

# ── 3. SKILL.md frontmatter (all copies) ────────────────────────────
update_skill_md() {
  local f="$1"
  if [[ -f "$f" ]]; then
    python3 -c "
import re, sys
path = sys.argv[1]
ver  = sys.argv[2]
text = open(path).read()
text = re.sub(r'(?m)^version:\s*\S+', 'version: ' + ver, text, count=1)
open(path, 'w').write(text)
print('  ✓ ' + path)
" "$f" "$NEW_VERSION"
  fi
}

update_skill_md "$ROOT/plugin/skills/openclaw-memory-pro/SKILL.md"
update_skill_md "$ROOT/skills/openclaw-memory-pro/SKILL.md"

# ── 4. README.md — update "Current Architecture" heading if present ─
# (no-op if the pattern isn't found; avoids breaking anything)

echo ""
echo "Done. All files now at v$NEW_VERSION."
echo ""
echo "Verify with:  grep -rn '\"version\"' plugin/package.json plugin/openclaw.plugin.json"

# ── 5. Optional: git commit + tag ───────────────────────────────────
if [[ "$DO_TAG" == "--tag" ]]; then
  echo ""
  echo "Creating git commit and tag..."
  cd "$ROOT"
  git add \
    plugin/package.json \
    plugin/openclaw.plugin.json \
    plugin/index.ts \
    plugin/skills/openclaw-memory-pro/SKILL.md \
    skills/openclaw-memory-pro/SKILL.md \
    scripts/bump-version.sh \
    2>/dev/null || true
  git commit -m "chore: bump version to v$NEW_VERSION"
  git tag -a "v$NEW_VERSION" -m "v$NEW_VERSION"
  echo "  ✓ Committed and tagged v$NEW_VERSION"
  echo "  → Push with: git push && git push --tags"
fi
