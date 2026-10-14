#!/usr/bin/env bash
# Remove downloaded models and/or the Python environment. Nothing else on the machine is touched.
# Usage: ./uninstall.sh [--profile NAME | --all-models] [--venv] [--all] [--yes]
#   --profile NAME   remove only that profile's models (keeps models shared with other profiles)
#   --all-models     remove every downloaded model
#   --venv           remove the Python environment (.venv)
#   --all            everything above
#   --yes            do not ask for confirmation
#   --dry-run        show what would be removed and how much space it frees, remove nothing
set -euo pipefail; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
MODELS_DIR="${AITD_MODELS_DIR:-}"; [[ -z "$MODELS_DIR" && -f .models-dir ]] && MODELS_DIR="$(cat .models-dir)"; MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
PROFILE=""; ALLM=0; VENV=0; YES=0; DRY=0
while [[ $# -gt 0 ]]; do case "$1" in
  --profile) PROFILE="$2"; shift 2;; --profile=*) PROFILE="${1#--profile=}"; shift;; --all-models) ALLM=1; shift;;
  --venv) VENV=1; shift;; --all) ALLM=1; VENV=1; shift;; --yes|-y) YES=1; shift;; --dry-run) DRY=1; shift;; -h|--help) sed -n '2,10p' "$0"; exit 0;;
  *) echo "unknown option $1" >&2; exit 2;; esac; done
[[ -n "$PROFILE" || $ALLM -eq 1 || $VENV -eq 1 ]] || { sed -n '2,10p' "$0"; exit 2; }
size() { du -sh "$1" 2>/dev/null | cut -f1; }
plan=()
if [[ $ALLM -eq 1 ]]; then [[ -d "$MODELS_DIR" ]] && plan+=("all models in $MODELS_DIR ($(size "$MODELS_DIR"))")
elif [[ -n "$PROFILE" ]]; then plan+=("models used only by profile '$PROFILE' in $MODELS_DIR"); fi
[[ $VENV -eq 1 && -d .venv ]] && plan+=("Python environment .venv ($(size .venv))")
[[ ${#plan[@]} -gt 0 ]] || { echo "nothing to remove"; exit 0; }
printf 'Will remove:\n'; printf '  - %s\n' "${plan[@]}"
if [[ $DRY -eq 1 ]]; then echo "(dry run: nothing removed)"; [[ -n "$PROFILE" && $ALLM -eq 0 ]] || exit 0; fi
if [[ $YES -ne 1 && $DRY -ne 1 ]]; then read -r -p "Proceed? [y/N] " a; [[ "$a" =~ ^[Yy]$ ]] || { echo "cancelled"; exit 0; }; fi
if [[ $ALLM -eq 1 ]]; then rm -rf "$MODELS_DIR"; echo "removed $MODELS_DIR"
elif [[ -n "$PROFILE" ]]; then
  [[ -x .venv/bin/python ]] || { echo "needs .venv to remove a single profile cleanly; use --all-models instead" >&2; exit 1; }
  PROFILE="$PROFILE" HF_HOME="$MODELS_DIR" DRY="$DRY" .venv/bin/python - <<'PY'
import os, json, pathlib, shutil
profiles = json.load(open("profiles.json")); name = os.environ["PROFILE"]; hub = pathlib.Path(os.environ["HF_HOME"]) / "hub"
def repos(p): return {p["observer"], p["performer"], *[m for m, _ in p["perplexity_models"]]}
mine = repos(profiles[name]); others = set().union(*(repos(p) for n, p in profiles.items() if n != name)) if len(profiles) > 1 else set()
victims = mine - others; kept = mine & others
def repo_dir(r): return hub / ("models--" + r.replace("/", "--"))
def blobs_of(r):  # real files behind a repo's snapshot symlinks (new shared-blob layout) plus its own blobs folder
    out = set()
    for f in repo_dir(r).rglob("*"):
        if f.is_file() or f.is_symlink():
            t = f.resolve()
            if t.exists(): out.add(t)
    return out
keep_blobs = set().union(*(blobs_of(r) for r in others if repo_dir(r).exists())) if others else set()
to_delete = set()
for r in victims:
    if repo_dir(r).exists(): to_delete |= blobs_of(r) - keep_blobs
size = sum(f.stat().st_size for f in to_delete if f.is_file())
gb = size / 1e9
if os.environ.get("DRY") == "1": print(f"would remove {', '.join(sorted(victims)) or 'nothing'} ({gb:.1f} GB)")
else:
    for f in to_delete:
        if f.is_file(): f.unlink()
    for r in victims:
        if repo_dir(r).exists(): shutil.rmtree(repo_dir(r))
    print(f"removed {', '.join(sorted(victims)) or 'nothing'} ({gb:.1f} GB freed)")
if kept: print(f"kept {', '.join(sorted(kept))}: also used by another profile")
PY
fi
if [[ $VENV -eq 1 && $DRY -ne 1 ]]; then rm -rf .venv; echo "removed .venv"; fi
[[ $DRY -eq 1 ]] || echo "done. Re-run ./install.sh to reinstall."
