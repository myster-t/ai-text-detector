#!/usr/bin/env bash
# ai-text-detector installer. Assumes nothing: no Homebrew, no Python, no Hugging Face CLI.
# Installs uv (a standalone Python manager), a Python 3.12 runtime, the virtualenv, and downloads the models
# into ./models so the checkout is self-contained. Re-running is safe; finished steps are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in Darwin|Linux) ;; *) fail "This installer supports macOS and Linux. On Windows use WSL2." ;; esac
command -v curl >/dev/null || fail "curl is required (it ships with macOS and nearly every Linux)."

# ---- resources -------------------------------------------------------------------------------------------
if [[ "$(uname -s)" == Darwin ]]; then
  ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 )); free_gb=$(df -g "$ROOT" | awk 'NR==2{print $4}')
else
  ram_gb=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1048576 )); free_gb=$(df -BG "$ROOT" | awk 'NR==2{gsub("G","",$4); print $4}')
fi
say "System check: ${ram_gb} GB RAM, ${free_gb} GB free disk"
(( free_gb >= 40 )) || fail "Need about 40 GB free for the models; ${free_gb} GB available."
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
say "Models (about 33 GB on first install; skipped when already present)"
export HF_HOME="$ROOT/models"; mkdir -p "$HF_HOME"
.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
MODELS = {
    "tiiuae/falcon-7b":          ["*.safetensors", "*.json", "tokenizer*"],
    "tiiuae/falcon-7b-instruct": ["*.safetensors", "*.json", "tokenizer*"],
    "Qwen/Qwen2.5-1.5B":         ["*.safetensors", "*.json", "*.txt"],
    "openai-community/gpt2":     ["*.safetensors", "*.json", "*.txt"],
}
for repo, patterns in MODELS.items():
    try:
        snapshot_download(repo, allow_patterns=patterns, local_files_only=True); print(f"  {repo}: present"); continue
    except Exception: pass
    print(f"  {repo}: downloading"); snapshot_download(repo, allow_patterns=patterns)
PY

# ---- self-test ------------------------------------------------------------------------------------------
say "Self-test on the bundled controls (loads all models once; a few minutes)"
./detect --selftest
say "Done. Score a file with:   ./detect path/to/text.txt      (reports land in ./reports)"
