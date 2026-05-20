#!/usr/bin/env python3
"""
Qwen-TTS Pipeline — emotion-controlled narration with 1.7B model.
Generates audio at natural speed, trims video to match.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np


PHASES = [
    {
        "name": "day",
        "text": "Bright sun, blue sky over gleaming glass towers.",
        "speaker": "serena",
        "instruct": "Warm, cheerful, bright, energetic, slightly excited.",
    },
    {
        "name": "dusk",
        "text": "Golden hour glow spreads across the skyline.",
        "speaker": "ryan",
        "instruct": "Calm, smooth, reflective, gentle, mellow tone.",
    },
    {
        "name": "night",
        "text": "Darkness falls, neon lights reveal a new city.",
        "speaker": "dylan",
        "instruct": "Deep, mysterious, quiet, atmospheric, whisper-like.",
    },
]


def get_video_info(video_path: str) -> tuple[int, float]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if count == 0:
        from PIL import Image
        img = Image.open(video_path)
        count = 0
        try:
            while True:
                count += 1
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    if fps <= 0 or fps > 120:
        fps = 8.0
    return count, fps


def main():
    parser = argparse.ArgumentParser(description="Qwen-TTS Narration Pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/final_with_audio.mp4")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_wav = str(Path(args.output).with_suffix(".wav"))

    num_frames, fps = get_video_info(args.video)
    video_duration = num_frames / fps
    print(f"Video: {num_frames} frames, {video_duration:.2f}s at {fps:.0f}fps")

    from qwen_tts import Qwen3TTSModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"  Speakers: {model.get_supported_speakers()}")

    # Synthesize each phase with emotion
    audio_segments = []
    for phase in PHASES:
        print(f"  [{phase['name']}] {phase['text']}")
        print(f"    instruct: {phase['instruct']}")
        wavs, sr = model.generate_custom_voice(
            text=phase["text"],
            speaker=phase["speaker"],
            language="English",
            instruct=phase["instruct"],
            non_streaming_mode=True,
        )
        audio = wavs[0] if isinstance(wavs, list) and len(wavs) > 0 else wavs
        audio = np.array(audio, dtype=np.float32)
        dur = len(audio) / sr
        print(f"    -> {dur:.2f}s")
        audio_segments.append(audio)

    combined = np.concatenate(audio_segments)
    total_dur = len(combined) / sr
    print(f"\nTotal audio: {total_dur:.2f}s @ {sr}Hz")

    # Trim leading/trailing silence
    threshold = 10 ** (-40 / 20) * np.max(np.abs(combined))
    mask = np.abs(combined) > threshold
    if np.any(mask):
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1])
        combined = combined[start:end]
        total_dur = len(combined) / sr
        print(f"After trim: {total_dur:.2f}s")

    # Save WAV (16-bit PCM for max compatibility)
    import soundfile as sf
    sf.write(output_wav, combined, sr, subtype='PCM_16')
    print(f"Saved WAV: {output_wav}")

    # Composite — create looped video via concat, then merge with audio
    import math, os
    loop_count = max(1, math.ceil(total_dur / video_duration))

    # Write concat file for looping
    concat_path = str(Path(args.video).with_suffix(".txt"))
    with open(concat_path, "w") as f:
        abs_video = os.path.abspath(args.video)
        for _ in range(loop_count):
            f.write(f"file '{abs_video}'\n")

    # First pass: create looped video
    looped_path = str(Path(args.video).parent / f"{Path(args.video).stem}_looped.mp4")
    concat_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-t", f"{total_dur:.2f}",
        looped_path,
    ]
    r1 = subprocess.run(concat_cmd, capture_output=True, text=True)
    os.remove(concat_path)
    if r1.returncode != 0:
        print(f"Concat loop failed:\n{r1.stderr}")
        return

    # Second pass: merge looped video with audio (AAC in MP4 = most compatible)
    cmd = [
        "ffmpeg", "-y",
        "-i", looped_path,
        "-i", output_wav,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        args.output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(looped_path)
    if result.returncode != 0:
        print(f"FFmpeg merge failed:\n{result.stderr}")
        return

    # Verify
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "csv=p=0", args.output],
        capture_output=True, text=True,
    )
    print(f"Saved: {args.output} ({info.stdout.strip()}s)")

    # Quick check audio level
    check = subprocess.run(
        ["ffmpeg", "-i", args.output, "-vn", "-acodec", "pcm_s16le",
         "-f", "wav", "-y", "/tmp/_tts_check.wav"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        y, _ = sf.read("/tmp/_tts_check.wav")
        rms = float(np.sqrt(np.mean(y**2)))
        max_amp = float(np.max(np.abs(y)))
        print(f"Audio check: rms={rms:.4f}, max={max_amp:.4f} "
              f"{'✓ AUDIBLE' if rms > 0.02 else '✗ TOO QUIET'}")
    else:
        print(f"Audio check skipped")


if __name__ == "__main__":
    main()
