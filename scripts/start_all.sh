#!/usr/bin/env bash
# scripts/start_all.sh — Start all Anti-gravity IDE services
set -euo pipefail

echo "→ Installing systemd user units..."
UNIT_DIR="${HOME}/.config/systemd/user"
mkdir -p "${UNIT_DIR}"

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for f in "${WORKTREE}/systemd/"*.{service,socket,timer}; do
    [[ -f "$f" ]] || continue
    unit="$(basename "$f")"
    dst="${UNIT_DIR}/${unit}"
    if [[ ! -f "$dst" ]] || ! diff -q "$f" "$dst" >/dev/null 2>&1; then
        cp "$f" "$dst"
        echo "  installed ${unit}"
    fi
done

systemctl --user daemon-reload

echo "→ Starting bus stream socket (always-on)..."
systemctl --user enable --now loci-bus-stream.socket
systemctl --user restart loci-bus-stream.service 2>/dev/null || true

echo "→ Starting Arc token offset service..."
systemctl --user enable --now arc-token-offset.service

echo ""
echo "Services:"
systemctl --user status loci-bus-stream.socket  --no-pager -l | head -4
systemctl --user status arc-token-offset.service --no-pager -l | head -4
echo ""
echo "IDE config: ${WORKTREE}/antigravity.json"
echo "Open in Anti-gravity IDE:  code --open-with antigravity ."
