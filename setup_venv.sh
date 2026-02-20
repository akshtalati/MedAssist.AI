#!/usr/bin/env bash
# Create virtual environment and install MedAssist.AI dependencies.
# Run from project root: ./setup_venv.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-venv}"
if [[ -n "$VIRTUAL_ENV" ]]; then
  echo "Already inside a virtualenv: $VIRTUAL_ENV"
  pip install -r requirements.txt
  exit 0
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi
echo "Activating $VENV_DIR ..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install -r requirements.txt
echo "Done. Activate with: source $VENV_DIR/bin/activate"
