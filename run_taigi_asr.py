"""Transcribe an audio file with a Taiwanese-finetuned Whisper model.

Uses NUTN-KWS/Whisper-Taiwanese-model-v0.5 (openai/whisper-large-v3-turbo
fine-tuned for Taiwanese ASR). Outputs Taiwanese Han characters (台語漢字).

This mirrors the post-processing in asr_core.py (簡->繁 via OpenCC) but works on
a single file and on CPU, since the project's bundled `models/` and CUDA are not
available in this environment.
"""
import sys
import time
import numpy as np


def load_audio_16k_mono(path):
    """Decode any audio file to 16kHz mono float32 without system ffmpeg.

    Tries soundfile/librosa first, then falls back to PyAV (bundled with
    faster-whisper) for mp3.
    """
    try:
        import librosa
        audio, _ = librosa.load(path, sr=16000, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"[load] librosa failed ({e}); falling back to PyAV", flush=True)

    import av
    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=16000)
    chunks = []
    for frame in container.decode(stream):
        for rs in resampler.resample(frame):
            chunks.append(rs.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(chunks).astype(np.float32)


def fmt_ts(seconds):
    if seconds is None:
        return "??:??:??"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    audio_path = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "rec001"
    model_id = "NUTN-KWS/Whisper-Taiwanese-model-v0.5"

    print(f"[1/4] Loading audio: {audio_path}", flush=True)
    t0 = time.time()
    audio = load_audio_16k_mono(audio_path)
    dur = len(audio) / 16000
    print(f"      duration={dur/60:.1f} min, samples={len(audio)} ({time.time()-t0:.1f}s)", flush=True)

    print(f"[2/4] Loading model: {model_id} (CPU)", flush=True)
    t0 = time.time()
    import torch
    from transformers import pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=-1,
        torch_dtype=torch.float32,
        chunk_length_s=30,
        stride_length_s=5,
    )
    print(f"      model ready ({time.time()-t0:.1f}s)", flush=True)

    print(f"[3/4] Transcribing (this is the slow part on CPU)...", flush=True)
    t0 = time.time()
    result = pipe(
        audio,
        return_timestamps=True,
        generate_kwargs={"language": "zh", "task": "transcribe"},
    )
    print(f"      done in {(time.time()-t0)/60:.1f} min", flush=True)

    raw_text = result["text"].strip()

    # 簡 -> 繁 (Taiwan) post-processing, matching asr_core.py
    try:
        import opencc
        s2tw = opencc.OpenCC("s2tw")
        text = s2tw.convert(raw_text)
    except Exception as e:
        print(f"[opencc] skipped ({e})", flush=True)
        text = raw_text

    # Plain transcript
    with open(f"{out_prefix}_asr.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # SRT with timestamps
    chunks = result.get("chunks") or []
    with open(f"{out_prefix}.srt", "w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks, 1):
            start, end = ch.get("timestamp", (None, None))
            seg = ch["text"].strip()
            try:
                seg = s2tw.convert(seg)
            except Exception:
                pass
            f.write(f"{i}\n{fmt_ts(start)} --> {fmt_ts(end)}\n{seg}\n\n")

    print(f"[4/4] Wrote {out_prefix}_asr.txt and {out_prefix}.srt", flush=True)
    print("\n===== TRANSCRIPT (preview) =====", flush=True)
    print(text[:1500], flush=True)


if __name__ == "__main__":
    main()
