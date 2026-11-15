#!/usr/bin/env bash
# Chef Mystico — one-command setup and launch.
set -euo pipefail
cd "$(dirname "$0")"

# Use the project's own virtualenv unless PYTHON says otherwise. Homebrew and
# recent Debian Pythons refuse `pip install` outside one.
if [ -n "${PYTHON:-}" ]; then
  PY=$PYTHON
else
  if [ ! -x .venv/bin/python ]; then
    echo "▸ creating .venv"
    python3 -m venv .venv
  fi
  PY=.venv/bin/python
fi
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}

echo "▸ checking dependencies"
if ! $PY -c "import fastapi, torch, sklearn, sqlalchemy, pandas" 2>/dev/null; then
  echo "  installing from requirements.txt …"
  $PY -m pip install -r requirements.txt
fi

echo "▸ validating the dish catalogue"
$PY -m ml.validate_catalog >/dev/null || { $PY -m ml.validate_catalog; exit 1; }

if [ ! -f data/artifacts/model.pt ]; then
  echo "▸ no trained model found — building the corpus and training now"
  echo "  (drop the Kaggle recipes.csv into data/ first if you want real recipe text)"
  $PY -m ml.prepare_data
  $PY -m ml.train
else
  echo "▸ trained model found at data/artifacts/model.pt"
fi

echo "▸ serving on http://$HOST:$PORT   (API docs at /docs, the decision tree at /tree)"
exec $PY -m uvicorn app.main:app --host "$HOST" --port "$PORT" "${@}"
