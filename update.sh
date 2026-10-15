#!/usr/bin/env bash
# Update this folder to the latest version of ai-text-detector without re-downloading the zip.
# Usage: ./update.sh [--no-selftest]
# Works for a folder that came from GitHub's "Download ZIP" (no git needed) and for a git clone.
# Keeps your models, Python environment, settings (.models-dir, .profile), reports and any control texts you added.
# After copying the new files it re-runs ./install.sh, which only installs packages and models that are missing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
REPO="myster-t/ai-text-detector"; BRANCH="main"; SELFTEST=1
for a in "$@"; do case "$a" in --no-selftest) SELFTEST=0;; -h|--help) sed -n '2,6p' "$0"; exit 0;; *) echo "unknown option $a" >&2; exit 2;; esac; done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
command -v curl >/dev/null || fail "curl is required"; command -v tar >/dev/null || fail "tar is required"

if [[ -d .git ]] && command -v git >/dev/null; then
  say "Git checkout detected: pulling $BRANCH"; git pull --ff-only origin "$BRANCH" || fail "git pull failed; resolve it manually"
else
  latest="$(curl -fsSL "https://api.github.com/repos/$REPO/commits/$BRANCH" 2>/dev/null | sed -n 's/^  "sha": "\([0-9a-f]*\)".*/\1/p' | head -1 || true)"
  current="$(cat .version 2>/dev/null || true)"
  if [[ -n "$latest" && "$latest" == "$current" ]]; then say "Already up to date (${latest:0:7})"; exit 0; fi
  say "Downloading the latest files from github.com/$REPO${latest:+ (${latest:0:7})}"
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/aitd-update-XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
  curl -fsSL "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" -o "$tmp/src.tgz" || fail "download failed (no network?)"
  tar -xzf "$tmp/src.tgz" -C "$tmp" || fail "could not unpack the download"
  src="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -1)"
  [[ -f "$src/aitd/report.py" && -f "$src/install.sh" ]] || fail "unexpected archive layout"
  say "Updating files (models, .venv, .models-dir, .profile, reports and your own controls are left alone)"
  # copy everything from the archive over this folder; nothing of yours is deleted
  ( cd "$src" && tar -cf - . ) | tar -xf - -C "$ROOT"
  [[ -n "$latest" ]] && printf '%s\n' "$latest" > .version
  # files that older versions kept at the root and that now live in aitd/ or scripts/
  for legacy in report.py binoculars.py perplexity.py style.py unicode_check.py features.py lmfeat.py charts.py slop.py classifier.py update-style-rules.sh; do
    [[ -f "$ROOT/$legacy" && -f "$ROOT/aitd/${legacy%.sh}.py" || -f "$ROOT/scripts/$legacy" ]] && rm -f "$ROOT/$legacy"
  done
  chmod +x detect install.sh uninstall.sh update.sh scripts/update-style-rules.sh 2>/dev/null || true
fi

say "Checking packages and models for the installed profile"
PROFILE_ARG=(); [[ -f .profile ]] && PROFILE_ARG=(--profile "$(cat .profile)")
if [[ $SELFTEST -eq 1 ]]; then ./install.sh "${PROFILE_ARG[@]}"; else SKIP_SELFTEST=1 ./install.sh "${PROFILE_ARG[@]}"; fi
say "Update complete"
