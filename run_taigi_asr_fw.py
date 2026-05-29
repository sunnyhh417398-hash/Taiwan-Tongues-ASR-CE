"""Transcribe with faster-whisper + VAD (matches asr_core.py pipeline, CPU).

Uses the CTranslate2-converted NUTN-KWS Taiwanese model in ./taigi-ct2.
VAD filtering + anti-repetition settings greatly reduce the hallucination/
looping seen with the plain transformers pipeline on long audio.
"""
import sys
import time
import numpy as np
from faster_whisper import WhisperModel


def load_audio_16k_mono(path):
    import librosa
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return audio.astype(np.float32)


def fmt_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    audio_path = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "rec001_fw"

    print(f"[1/4] Loading audio: {audio_path}", flush=True)
    t0 = time.time()
    audio = load_audio_16k_mono(audio_path)
    print(f"      duration={len(audio)/16000/60:.1f} min ({time.time()-t0:.1f}s)", flush=True)

    print("[2/4] Loading CT2 model (./taigi-ct2, int8 CPU)", flush=True)
    t0 = time.time()
    model = WhisperModel("taigi-ct2", device="cpu", compute_type="int8", cpu_threads=4)
    print(f"      model ready ({time.time()-t0:.1f}s)", flush=True)

    print("[3/4] Transcribing with VAD...", flush=True)
    t0 = time.time()
    segments, info = model.transcribe(
        audio,
        language="zh",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    try:
        import opencc
        s2tw = opencc.OpenCC("s2tw")
        conv = s2tw.convert
    except Exception:
        conv = lambda x: x

    srt_lines = []
    full = []
    for i, seg in enumerate(segments, 1):
        txt = conv(seg.text.strip())
        full.append(txt)
        srt_lines.append(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{txt}\n")
        if i <= 8 or i % 20 == 0:
            print(f"   [{fmt_ts(seg.start)}] {txt}", flush=True)
    print(f"      done in {(time.time()-t0)/60:.1f} min, {len(full)} segments", flush=True)

    text = "".join(full)
    with open(f"{out_prefix}_asr.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(f"{out_prefix}.srt", "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"[4/4] Wrote {out_prefix}_asr.txt and {out_prefix}.srt", flush=True)


if __name__ == "__main__":
    main()
