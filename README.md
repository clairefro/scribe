# scribe

Local audio transcription CLI using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with optional speaker diarization (pyannote segmentation + WeSpeaker embeddings via sherpa-onnx).

> **Internet is required for the first run only.**
> The first time you run `scribe`, it downloads its models and needs internet access:
>
> - **Transcription:** the Whisper model (`small` by default, about 500 MB) is downloaded from **Hugging Face**.
> - **Speaker labels (`--diarize`):** the diarization models are downloaded from **GitHub**.
>
> After that, `scribe` is **100% local and works fully offline**: your audio never leaves your machine, and nothing else is downloaded. Models are cached in `~/.cache/huggingface` (Whisper) and `~/.cache/scribe/models` (diarization).
>
> To prepare a machine ahead of time, run it once while online, including `--diarize` if you'll use it, on any short audio file. A built standalone binary with bundled models needs no internet at all (see [Build a standalone binary](#build-a-standalone-binary)).
>
> When online, the Whisper library may briefly check Hugging Face for model updates on each run (no audio is sent). To forbid all network access, run with `HF_HUB_OFFLINE=1 scribe ...`.

## Usage

```
scribe talk.m4a -o talk.txt

# diarize assumes 2 speakers by default; .md gives markdown
scribe interview.m4a -o interview.md --diarize

scribe meeting.wav -o meeting.txt --diarize --num-speakers 3 --timestamps

```

| Option                                     | Description                                                                                                                   |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `-o, --output`                             | Output path; `.md`/`.markdown` gives markdown (default: stdout)                                                               |
| `-m, --model`                              | Whisper model name or local path (default: `small`)                                                                           |
| `-l, --language`                           | Language code, e.g. `en` (default: auto-detect)                                                                               |
| `--device`                                 | `auto`, `cpu`, or `cuda`                                                                                                      |
| `--compute-type`                           | e.g. `int8`, `float16`                                                                                                        |
| `--threads`                                | CPU threads (default: 4)                                                                                                      |
| `-d, --diarize`                            | Label speakers (assumes 2 speakers unless `--num-speakers` is set)                                                            |
| `--num-speakers`                           | Speaker count for `--diarize` (default: 2; `0` auto-detects)                                                                  |
| `--embedding-model`                        | `wespeaker-resnet34` (default), `campplus-en`, `titanet-small`, `titanet-large`                                               |
| `--cluster-threshold`                      | With `--num-speakers 0`: lower finds more speakers (default: 0.5)                                                             |
| `--min-turn`                               | Merge speaker turns shorter than N seconds into the neighbouring speaker (default: 0 = off)                                   |
| `--min-duration-on` / `--min-duration-off` | Diarizer segment filtering/merging in seconds (defaults: 0.3 / 0.5)                                                           |
| `--word-shift`                             | Shift word times later (seconds) when matching speaker turns; fixes first words going to the previous speaker (default: 0.15) |
| `--timestamps`                             | Prefix lines with timestamps                                                                                                  |

### Technical terms and acronyms

Whisper guesses acronyms phonetically (e.g. "GSoC" becomes "g soccer"). A built-in list of common programming terms (Recurse Center, GSoC, Kubernetes, PostgreSQL, gRPC, etc.; see `DEFAULT_TERMS` in `scribe/cli.py`) is always used. Add your own, which take priority over the defaults, or disable the built-ins with `--no-default-terms`:

```
scribe interview.m4a -o out.md --diarize --terms "GSOC, gRPC"
scribe interview.m4a -o out.md --terms-file terms.txt    # one term per line
```

Spell terms exactly as you want them written. Keep your own list short and focused on words Whisper actually gets wrong: long lists, or common words like "Rust" or "Swift", can make it insert those words by mistake. Whisper only uses roughly the first 220 tokens, so if the list is too long, built-in terms are dropped to make room for yours and a `glossary:` line says how many fit. A larger Whisper model (`-m medium` or `-m large-v3`) also helps substantially but is slower and needs more RAM; the model downloads on first use.

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
- Internet is needed for the first run only, to download models (Hugging Face for Whisper, GitHub for diarization). After that it is fully offline. A built binary with bundled models needs none.

## Install (Python 3.10-3.14)

First check your Python: `python3 --version`. It must be **3.10 to 3.14**. Python 3.15 isn't supported yet because `ctranslate2` and `sherpa-onnx` don't ship wheels for it, and installing on it fails with a confusing `no matching distributions available ... ctranslate2` error. If yours is 3.15 (or older than 3.10), install 3.14 (`brew install python@3.14` on macOS) and use it explicitly as below.

```
git clone <repo-url> scribe && cd scribe
pipx install --python python3.14 .       # global `scribe` command (drop --python if python3 is 3.10-3.14)
```

Or in a venv (`scribe` is only available while it's active). Use the versioned `python3.14`, not plain `python3`:

```
python3.14 -m venv .venv && source .venv/bin/activate    # Windows: py -3.14 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e .
```

With [uv](https://docs.astral.sh/uv/) you don't need to install Python yourself: `uv tool install --python 3.14 .`

**If install fails:** make sure the right venv is active (`which pip` should point into your project's `.venv`) and run `pip --version` to see which Python pip uses. Don't run `pip install` inside another project's venv.

The first run needs internet to download models (see the note at the top). After that it works offline.

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
