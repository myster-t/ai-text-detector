#!/usr/bin/env bash
# ai-text-detector installer. Assumes nothing: no Homebrew, no Python, no Hugging Face CLI.
# Usage: ./install.sh [--older-models | --profile NAME] [--models-dir PATH]
#   --older-models  download the original 2023 set (Falcon-7B pair + Qwen2.5/GPT-2, ~33 GB) instead of the default
#                   default set (OLMo-3-7B pair + Qwen3.5 small models + classifier, ~36 GB). --profile NAME picks any
#                   entry in profiles.json; --profile smollm3 is the light set (~13 GB).
#   --models-dir  keep the models somewhere else (an external drive, a shared folder). Remembered in .models-dir.
# Installs uv (a standalone Python manager), a Python 3.12 runtime, the virtualenv, and downloads the models
# into ./models so the checkout is self-contained. Re-running is safe; finished steps are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PROFILE=olmo3; MODELS_DIR=""
while [[ $# -gt 0 ]]; do case "$1" in
  --older-models) PROFILE=original; shift;;
  --profile) PROFILE="$2"; shift 2;; --profile=*) PROFILE="${1#--profile=}"; shift;;
  --models-dir) MODELS_DIR="$2"; shift 2;; --models-dir=*) MODELS_DIR="${1#--models-dir=}"; shift;;
  -h|--help) sed -n '2,6p' "$0"; exit 0;; *) fail "unknown option $1 (use --older-models, --profile NAME, --models-dir PATH)";; esac; done
# where the models live: --models-dir > AITD_MODELS_DIR > ./.models-dir file > ./models
if [[ -n "$MODELS_DIR" ]]; then mkdir -p "$MODELS_DIR"; MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"; printf '%s\n' "$MODELS_DIR" > .models-dir
elif [[ -n "${AITD_MODELS_DIR:-}" ]]; then MODELS_DIR="$AITD_MODELS_DIR"
elif [[ -f .models-dir ]]; then MODELS_DIR="$(cat .models-dir)"
else MODELS_DIR="$ROOT/models"; fi
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in Darwin|Linux) ;; *) fail "This installer supports macOS and Linux. On Windows use WSL2." ;; esac
command -v curl >/dev/null || fail "curl is required (it ships with macOS and nearly every Linux)."

# ---- resources -------------------------------------------------------------------------------------------
if [[ "$(uname -s)" == Darwin ]]; then
  ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 )); free_gb=$(df -g "$MODELS_DIR" | awk 'NR==2{print $4}')
else
  ram_gb=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1048576 )); free_gb=$(df -BG "$MODELS_DIR" | awk 'NR==2{gsub("G","",$4); print $4}')
fi
case "$PROFILE" in qwen35) NEED=50;; olmo3) NEED=45;; smollm3) NEED=25;; hybrid) NEED=45;; *) NEED=40;; esac
say "System check: ${ram_gb} GB RAM, ${free_gb} GB free at ${MODELS_DIR} (profile ${PROFILE} needs about ${NEED} GB)"
(( free_gb >= NEED )) || fail "Need about ${NEED} GB free for the ${PROFILE} models; ${free_gb} GB available at ${MODELS_DIR}. Use --models-dir to put them on another disk."
(( ram_gb >= 32 )) || printf 'warning: the Falcon-7B pair needs ~29 GB of memory during a run; with %s GB this will be slow or fail.\n' "$ram_gb"

# ---- uv (installs its own Python; no Homebrew, no sudo) --------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  say "Installing uv (standalone, into ~/.local/bin)"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  command -v uv >/dev/null || fail "uv did not install; see https://docs.astral.sh/uv/getting-started/installation/"
fi
say "Python 3.12 runtime"; uv python install 3.12 --quiet

# ---- virtualenv + packages ------------------------------------------------------------------------------
say "Python packages (PyTorch, Transformers)"
[[ -d .venv ]] || uv venv --python 3.12 --quiet .venv
uv pip install --python .venv/bin/python --quiet -r requirements.txt

# ---- models (into ./models, so the folder is portable) --------------------------------------------------
say "Models for profile ${PROFILE} into ${MODELS_DIR} (skipped when already present)"
export HF_HOME="$MODELS_DIR"; mkdir -p "$HF_HOME"
PROFILE="$PROFILE" .venv/bin/python - <<'PY'
import os, json
from huggingface_hub import snapshot_download
P = json.load(open("profiles.json"))[os.environ["PROFILE"]]
patterns = ["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.jinja"]
MODELS = {m: patterns for m in [P["observer"], P["performer"]] + [m for m, _ in P["perplexity_models"]]}
for repo, patterns in MODELS.items():
    try:
        snapshot_download(repo, allow_patterns=patterns, local_files_only=True); print(f"  {repo}: present"); continue
    except Exception: pass
    print(f"  {repo}: downloading"); snapshot_download(repo, allow_patterns=patterns)
PY

# ---- self-test ------------------------------------------------------------------------------------------
say "Self-test on the bundled controls (loads all models once; a few minutes)"
./detect --selftest --profile "$PROFILE"
say "Remove models later with ./uninstall.sh"
say "Done. Score a file with:   ./detect path/to/text.txt$( [[ "$PROFILE" == original ]] && echo " --older-models" || { [[ "$PROFILE" != olmo3 ]] && echo " --profile $PROFILE"; } )      (reports land in ./reports)"
