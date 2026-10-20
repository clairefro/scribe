#!/usr/bin/env bash
# Builds a standalone app at dist/scribe/scribe (must be run on the target OS).
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[build]"

# Bundle the whisper model so the binary works offline: ./build.sh small
MODEL="${1:-}"
ADD_DATA=()
if [ -n "$MODEL" ]; then
  if [ ! -d "bundled_models/$MODEL" ]; then
    python - "$MODEL" <<'EOF'
import sys
from faster_whisper.utils import download_model
download_model(sys.argv[1], output_dir=f"bundled_models/{sys.argv[1]}")
EOF
  fi
  ADD_DATA=(--add-data "bundled_models:bundled_models")
fi

# Pre-fetch diarization models into the cache for bundling.
python -c "from scribe.cli import fetch_diarization_models as f; f()"

pyinstaller --noconfirm --onedir --name scribe \
  --collect-all faster_whisper --collect-all ctranslate2 \
  --collect-all sherpa_onnx --collect-all av \
  --add-data "$HOME/.cache/scribe/models:models" \
  ${ADD_DATA[@]+"${ADD_DATA[@]}"} \
  scribe/cli.py
echo "built dist/scribe"
