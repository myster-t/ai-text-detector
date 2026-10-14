# avoid-ai-writing detector (vendored)

`patterns.js` is the detector engine from https://github.com/conorbronsdon/avoid-ai-writing by Conor Bronsdon, MIT License
(see LICENSE in this folder). Vendored unmodified from version 3.37.0, commit 2dc826f7b9aea762f5557e56ba88a3e515bd2fed, on 2026-10-14.

It is executed in-process by the V8 engine bundled in the `mini-racer` Python package, so no Node.js install is needed. Refresh with
`./update-style-rules.sh`, which fetches the current file and records the new version and commit here.
