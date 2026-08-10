#!/usr/bin/env bash
# Run the Adamawa Fulfulde Streamlit demo in a detached tmux session, so it
# survives a closed terminal and its logs stay readable.
#
#   scripts/demo_tmux.sh           start the app on localhost only
#   scripts/demo_tmux.sh tunnel    also open a PUBLIC Cloudflare Quick Tunnel
#   tmux attach -t fub-demo        watch it (detach again with Ctrl-b then d)
#   tmux kill-session -t fub-demo  stop everything
set -euo pipefail

SESSION=fub-demo
PORT=${PORT:-8501}
THREADS=${FUB_THREADS:-4}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$REPO_ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' is already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already serving something else:" >&2
  lsof -iTCP:"$PORT" -sTCP:LISTEN >&2
  echo "Stop it first (pkill -f 'streamlit run') or set PORT to a free port." >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n app
tmux send-keys -t "$SESSION:app" \
  "cd '$REPO_ROOT' && FUB_THREADS=$THREADS .venv/bin/streamlit run scripts/demo_streamlit_adamawa.py --server.address 127.0.0.1 --server.port $PORT --server.headless true" C-m

# The pane keeps a live shell even if Streamlit exits, so poll the health
# endpoint with a deadline rather than waiting on the session.
echo "Waiting for Streamlit on 127.0.0.1:$PORT ..."
for _ in $(seq 60); do
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/_stcore/health"; then
    echo "App is live at http://127.0.0.1:$PORT"
    break
  fi
  sleep 1
done

if ! curl -sf -o /dev/null "http://127.0.0.1:$PORT/_stcore/health"; then
  echo "Streamlit did not come up within 60s. Its output:" >&2
  tmux capture-pane -p -t "$SESSION:app" | tail -20 >&2
  exit 1
fi

if [ "${1:-}" = "tunnel" ]; then
  tmux new-window -t "$SESSION" -n tunnel
  tmux send-keys -t "$SESSION:tunnel" "cloudflared tunnel --url http://localhost:$PORT" C-m
  echo
  echo "Opening a public tunnel. The URL is reachable by anyone who has it, with no login."
  for _ in $(seq 30); do
    # -J joins wrapped lines: the hostname is long enough to split across rows.
    url=$(tmux capture-pane -p -J -S -200 -t "$SESSION:tunnel" | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1 || true)
    [ -n "$url" ] && break
    sleep 1
  done
  if [ -n "${url:-}" ]; then
    echo "Public URL: $url"
  else
    echo "Tunnel URL not printed yet. Read it with:"
    echo "  tmux capture-pane -p -t $SESSION:tunnel | grep trycloudflare"
  fi
fi

echo
echo "attach:  tmux attach -t $SESSION"
echo "stop:    tmux kill-session -t $SESSION"
