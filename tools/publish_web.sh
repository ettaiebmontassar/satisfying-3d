#!/usr/bin/env bash
# Publishes web/ as https://howmanycuts.github.io (repo howmanycuts/howmanycuts.github.io, branch main).
# The site repo holds one commit with a neutral author: each publish replaces it.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
cp -a "$here/web/." "$tmp/"
cd "$tmp"
git init -q && git symbolic-ref HEAD refs/heads/main
git add -A
git -c user.name="How Many Cuts" -c user.email="howmanycuts@users.noreply.github.com" commit -q -m "Publish the game"
git push -q --force git@github.com:howmanycuts/howmanycuts.github.io.git main
echo "published: https://howmanycuts.github.io"
