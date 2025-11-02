#!/usr/bin/env bash
set -euo pipefail

# Minimal setup script (Linux/macOS). For Windows, use PowerShell equivalent.
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt || true
