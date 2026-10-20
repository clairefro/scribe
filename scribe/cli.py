import argparse
import sys
import tarfile
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

BUNDLE = Path(getattr(sys, "_MEIPASS", ""))
CACHE = Path.home() / ".cache" / "scribe" / "models"
if (BUNDLE / "models").is_dir():
    CACHE = BUNDLE / "models"
SEG_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
EMBEDDING_MODELS = {
    "wespeaker-resnet34": "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "campplus-en": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    "titanet-small": "nemo_en_titanet_small.onnx",
    "titanet-large": "nemo_en_titanet_large.onnx",
}
DEFAULT_EMBEDDING = "wespeaker-resnet34"
PARAGRAPH_GAP = 2.0  # seconds of silence that starts a new paragraph
PARAGRAPH_CHARS = 500  # paragraphs break at the next sentence end once this long
SEG_PATH = CACHE / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"


def log(msg):
    print(msg, file=sys.stderr)


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def fetch_diarization_models(embedding=DEFAULT_EMBEDDING):
    if not SEG_PATH.exists():
        archive = CACHE / "seg.tar.bz2"
        _download(SEG_URL, archive)
        with tarfile.open(archive) as t:
            t.extractall(CACHE, filter="data")
        archive.unlink()
    filename = EMBEDDING_MODELS[embedding]
    if not (CACHE / filename).exists():
        _download(EMB_BASE_URL + filename, CACHE / filename)
    return CACHE / filename


def diarize(samples, num_speakers, threads, embedding=DEFAULT_EMBEDDING, threshold=0.5, min_on=0.3, min_off=0.5):
    import sherpa_onnx

    emb_path = fetch_diarization_models(embedding)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG_PATH)
            ),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_path), num_threads=threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=threshold
        ),
        min_duration_on=min_on,
        min_duration_off=min_off,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    with spinner("diarizing") as state:

        def on_progress(done, total, *_):
            state["fraction"] = done / max(total, 1)
            return 0

        result = sd.process(samples, callback=on_progress).sort_by_start_time()
    turns = [(r.start, r.end, r.speaker) for r in result]
    log(f"diarization: {len(turns)} turns, {len({t[2] for t in turns})} speakers")
    if not num_speakers and len({t[2] for t in turns}) < 2:
        log(f"hint: one speaker found; pass --num-speakers N or lower --cluster-threshold (now {threshold})")
    return turns


def speaker_for(word, turns, shift=0.0):
    """Speaker whose turn contains the word's midpoint (shifted later by `shift` s), else the nearest turn within 0.5 s."""
    t = (word.start + word.end) / 2 + shift
    inside = [x for x in turns if x[0] <= t <= x[1]]
    if inside:
        return max(inside, key=lambda x: min(word.end, x[1]) - max(word.start, x[0]))[2]
    gap, spk = min(((max(s - t, t - e), spk) for s, e, spk in turns), default=(None, None))
    return spk if gap is not None and gap <= 0.5 else None


def coalesce(blocks):
    out = []
    for b in blocks:
        if out and out[-1]["label"] == b["label"]:
            out[-1]["text"] += b["text"]
            out[-1]["end"] = b["end"]
        else:
            out.append(b)
    return out


def smooth(blocks, min_turn):
    """Relabel turns shorter than min_turn seconds to the longer adjacent speaker."""
    blocks = coalesce(blocks)
    while len(blocks) > 1:
        # Never absorb a speaker's only turns, or a speaker with short turns vanishes.
        keep = {b["label"] for b in blocks if b["end"] - b["start"] >= min_turn}
        short = [i for i, b in enumerate(blocks) if b["end"] - b["start"] < min_turn and b["label"] in keep]
        if not short:
            break
        i = min(short, key=lambda i: blocks[i]["end"] - blocks[i]["start"])
        neighbors = [blocks[j] for j in (i - 1, i + 1) if 0 <= j < len(blocks)]
        blocks[i]["label"] = max(neighbors, key=lambda n: n["end"] - n["start"])["label"]
        blocks = coalesce(blocks)
    return blocks


def ts(t):
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@contextmanager
def spinner(label):
    """Animated stderr status line; set state["fraction"] (0-1) to show percent and ETA."""
    state = {"fraction": 0.0, "done": False}
    started = time.monotonic()

    def spin():
        frames = "|/-\\"
        i = 0
        while not state["done"]:
            elapsed = time.monotonic() - started
            f = min(max(state["fraction"], 0.0), 1.0)
            extra = f"  {int(f * 100)}%" if f > 0 else ""
            if f > 0.02:
                extra += f"  ETA {ts(elapsed / f * (1 - f))}"
            print(f"\r{label} {frames[i % 4]} {ts(elapsed)}{extra}\033[K", end="", file=sys.stderr, flush=True)
            i += 1
            time.sleep(0.2)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        state["done"] = True
        thread.join()
        print(f"\r{label}: done\033[K", file=sys.stderr)


def main():
    # Frozen builds re-launch themselves as `scribe -B -S -I -c <code>` for multiprocessing helpers.
    if len(sys.argv) > 2 and "-c" in sys.argv[1:]:
        i = sys.argv.index("-c")
        if all(a.startswith("-") for a in sys.argv[1:i]) and i + 1 < len(sys.argv):
            exec(sys.argv[i + 1])
            sys.exit(0)

    p = argparse.ArgumentParser(
        prog="scribe",
        description="Local audio transcription.",
        epilog="examples:\n"
        "  scribe talk.m4a -o talk.txt\n"
        "  scribe meeting.wav -o meeting.txt --diarize\n"
        "  scribe meeting.wav -o meeting.md --diarize --num-speakers 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("audio", type=Path, help="input audio file (m4a, wav, mp3, ogg, ...)")
    p.add_argument("-o", "--output", type=Path, help="output transcript path; a .md suffix gives markdown output (default: stdout)")
    p.add_argument("-m", "--model", default="small", help="faster-whisper model name or local path (default: small)")
    p.add_argument("-l", "--language", help="language code, e.g. en (default: auto-detect)")
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    p.add_argument("--compute-type", default="auto", help="e.g. int8, float16")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("-d", "--diarize", action="store_true", help="label speakers (assumes 2 speakers unless --num-speakers is set)")
    p.add_argument("--num-speakers", type=int, default=2, help="speaker count for diarization; 0 auto-detects via --cluster-threshold (default: 2)")
    p.add_argument("--embedding-model", choices=list(EMBEDDING_MODELS), default=DEFAULT_EMBEDDING, help="speaker embedding model (default: %(default)s)")
    p.add_argument("--cluster-threshold", type=float, default=0.5, help="used when --num-speakers is 0; lower finds more speakers (default: 0.5)")
    p.add_argument("--min-turn", type=float, default=0.0, help="merge speaker turns shorter than this many seconds into the neighbouring speaker; 0 disables (default: 0)")
    p.add_argument("--min-duration-on", type=float, default=0.3, help="drop diarizer speech segments shorter than this many seconds (default: 0.3)")
    p.add_argument("--min-duration-off", type=float, default=0.5, help="merge same-speaker segments separated by less than this many seconds (default: 0.5)")
    p.add_argument("--word-shift", type=float, default=0.15, help="seconds to shift word times later when matching speaker turns; fixes first words of a turn going to the previous speaker (default: 0.15)")
    p.add_argument("--timestamps", action="store_true", help="prefix lines with timestamps")
    args = p.parse_args()
    markdown = bool(args.output) and args.output.suffix.lower() in (".md", ".markdown")

    if not args.audio.is_file():
        p.error(f"file not found: {args.audio}")

    from faster_whisper import WhisperModel, decode_audio

    log("decoding audio")
    samples = decode_audio(str(args.audio), sampling_rate=16000)

    bundled = BUNDLE / "bundled_models" / args.model
    model_id = str(bundled) if bundled.is_dir() else args.model
    log(f"loading model '{args.model}'")
    model = WhisperModel(
        model_id,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.threads,
    )
    segments, info = model.transcribe(
        samples, language=args.language, vad_filter=True, word_timestamps=args.diarize
    )
    log(f"language: {info.language}")
    collected = []
    with spinner("transcribing") as state:
        for seg in segments:
            collected.append(seg)
            state["fraction"] = seg.end / max(info.duration, 1e-6)
    segments = collected

    turns = []
    if args.diarize:
        turns = diarize(samples, args.num_speakers, args.threads, args.embedding_model, args.cluster_threshold, args.min_duration_on, args.min_duration_off)

    blocks = []  # (label, start, text)
    if args.diarize:
        # Label each word so speaker changes inside a Whisper segment are kept.
        prev_label = None
        for seg in segments:
            for w in seg.words or []:
                spk = speaker_for(w, turns, args.word_shift)
                label = f"SPEAKER_{spk:02d}" if spk is not None else (prev_label or "UNKNOWN")
                prev_label = label
                if blocks and blocks[-1]["label"] == label:
                    blocks[-1]["text"] += w.word
                    blocks[-1]["end"] = w.end
                else:
                    blocks.append({"label": label, "start": w.start, "end": w.end, "text": w.word})
        blocks = smooth(blocks, args.min_turn)
        for b in blocks:
            b["text"] = b["text"].strip()
        blocks = [b for b in blocks if b["text"]]
    else:
        # Whisper segments often end mid-sentence; join them until a pause or a long paragraph ends at a sentence.
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            prev = blocks[-1] if blocks else None
            if (
                prev
                and seg.start - prev["end"] < PARAGRAPH_GAP
                and not (len(prev["text"]) > PARAGRAPH_CHARS and prev["text"].endswith((".", "?", "!")))
            ):
                prev["text"] += " " + text
                prev["end"] = seg.end
            else:
                blocks.append({"label": None, "start": seg.start, "end": seg.end, "text": text})

    paragraphs = []
    for b in blocks:
        label, start, text = b["label"], b["start"], b["text"]
        stamp = ts(start) if args.timestamps else None
        if markdown:
            if label:
                head = f"**{label}**" + (f" `{stamp}`" if stamp else "")
                paragraphs.append(f"{head}\n\n{text}")
            else:
                paragraphs.append((f"`{stamp}` " if stamp else "") + text)
        else:
            prefix = f"[{stamp}] " if stamp else ""
            paragraphs.append(f"{prefix}{label + ': ' if label else ''}{text}")

    out = "\n\n".join(paragraphs).strip() + "\n"
    if args.output:
        args.output.write_text(out, encoding="utf-8")
        log(f"wrote {args.output}")
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
