#!/usr/bin/env bash
# Refresh the vendored avoid-ai-writing detector (MIT, Conor Bronsdon) to the current upstream main.
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")"
D=third_party/avoid-ai-writing; B=https://raw.githubusercontent.com/conorbronsdon/avoid-ai-writing/main
SHA=$(curl -s https://api.github.com/repos/conorbronsdon/avoid-ai-writing/commits/main | python3 -c "import sys,json; print(json.load(sys.stdin)['sha'])")
VER=$(curl -sL $B/package.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','?'))")
curl -sL $B/detector/patterns.js -o $D/patterns.js; curl -sL $B/LICENSE -o $D/LICENSE
sed -i.bak -E "s/version [^,]+, commit [0-9a-f]+, on [0-9-]+/version $VER, commit $SHA, on $(date +%F)/" $D/NOTICE.md && rm -f $D/NOTICE.md.bak
echo "avoid-ai-writing detector updated to $VER ($SHA)"
