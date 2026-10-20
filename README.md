# scribe

Local audio transcription CLI using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with optional speaker diarization (pyannote segmentation + WeSpeaker embeddings via sherpa-onnx). Runs fully offline once models are downloaded.

## Usage

```
scribe talk.m4a -o talk.txt
scribe meeting.wav -o meeting.txt --diarize --num-speakers 3 --timestamps
scribe interview.m4a -o interview.md --diarize   # 2 speakers by default; .md gives markdown
```

| Option                                     | Description                                                                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `-o, --output`                             | Output path; `.md`/`.markdown` gives markdown (default: stdout)                             |
| `-m, --model`                              | Whisper model name or local path (default: `small`)                                         |
| `-l, --language`                           | Language code, e.g. `en` (default: auto-detect)                                             |
| `--device`                                 | `auto`, `cpu`, or `cuda`                                                                    |
| `--compute-type`                           | e.g. `int8`, `float16`                                                                      |
| `--threads`                                | CPU threads (default: 4)                                                                    |
| `-d, --diarize`                            | Label speakers (assumes 2 speakers unless `--num-speakers` is set)                          |
| `--num-speakers`                           | Speaker count for `--diarize` (default: 2; `0` auto-detects)                                |
| `--embedding-model`                        | `wespeaker-resnet34` (default), `campplus-en`, `titanet-small`, `titanet-large`             |
| `--cluster-threshold`                      | With `--num-speakers 0`: lower finds more speakers (default: 0.5)                           |
| `--min-turn`                               | Merge speaker turns shorter than N seconds into the neighbouring speaker (default: 0 = off) |
| `--min-duration-on` / `--min-duration-off` | Diarizer segment filtering/merging in seconds (defaults: 0.3 / 0.5)                         |
| `--word-shift`                             | Shift word times later (seconds) when matching speaker turns; fixes first words going to the previous speaker (default: 0.15) |
| `--timestamps`                             | Prefix lines with timestamps                                                                |

### Diarization tips

- Defaults are tuned for two-person calls. For other counts pass `--num-speakers N`, or `0` to auto-detect.
- Clean audio matters most: headsets or separate mics, no speakerphone, little crosstalk. Separate per-participant recordings beat any diarization model.
- Speakers flip back and forth: enable smoothing with `--min-turn 1` (up to 3-4 for heavier smoothing) or raise `--min-duration-off` (1.0).
- Smoothing is off by default; if it drops real short replies, lower `--min-turn`.
- Try `--embedding-model campplus-en` or `titanet-large` and compare on the same file.
- Each model downloads on first use; the built binary bundles only the default.

Supported input: any audio format FFmpeg can decode, including m4a, mp3, wav, ogg, flac, aac, opus, wma and webm (video files such as mp4 work too; only the audio track is used). The extension doesn't matter, only the actual encoding.

Relative paths resolve against the current directory. Models are cached in `~/.cache/scribe/models` (Windows: `%USERPROFILE%\.cache\scribe\models`).

## Requirements

Approximate figures, not benchmarked:

- 64-bit OS: macOS (Apple Silicon or Intel), Linux (x86_64/arm64), or Windows 10+.
- RAM: 4 GB minimum for `small`/`base` with `--compute-type int8`; 8 GB+ recommended; `medium`/`large-v3` need roughly 4-10 GB. Diarization adds a few hundred MB.
- Disk: ~1 GB for `small` plus diarization models; ~3 GB free to build the binary.
- CPU is enough; a larger model or longer audio just takes longer. NVIDIA GPU (CUDA 12) is optional on Linux/Windows.
- Network is needed once to download models (unless bundled in a built binary).

## Install (Python 3.10+)

Whisper models download from Hugging Face on first use, so the first run needs internet.

```
pipx install .          # global `scribe` command
```

Or in a venv (`scribe` is only available while it's active):

```
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

Apple Silicon runs on CPU only. For NVIDIA GPUs on Linux/Windows, install CUDA 12 + cuDNN 9 and pass `--device cuda`.

## Build a standalone binary

Build on each target OS/arch (no cross-compiling). Output: the folder `dist/scribe/` (executable: `dist/scribe/scribe`). Keep the folder intact; the executable needs its `_internal/` directory.

### macOS/Linux

```
chmod +x build.sh       # first time only
./build.sh small        # macOS/Linux; bundles the "small" Whisper model
ln -sf "$PWD/dist/scribe/scribe" ~/.local/bin/scribe
```

### Windows (PowerShell)

```
py -3 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[build]"
python -c "from faster_whisper.utils import download_model; download_model('small', output_dir='bundled_models/small')"
python -c "from scribe.cli import fetch_diarization_models as f; f()"
pyinstaller --noconfirm --onedir --name scribe --collect-all faster_whisper --collect-all ctranslate2 --collect-all sherpa_onnx --collect-all av --add-data "$env:USERPROFILE\.cache\scribe\models;models" --add-data "bundled_models;bundled_models" scribe/cli.py
```

Notes:

- The first build is slow because it downloads the Whisper and diarization models; later builds reuse the cached models.
- macOS Gatekeeper block: `xattr -dr com.apple.quarantine dist/scribe`.
- Linux: build on an older distro (e.g. Ubuntu 22.04) for glibc compatibility.
- The folder is several hundred MB with models bundled.
- Diarization uses an English speaker-embedding model; accuracy may drop for other languages.
