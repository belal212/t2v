#!/usr/bin/env python3
"""
Qwen-TTS Pipeline — Bonus Audio Generation

Generates multi-voice, emotion-controlled, context-aware narration
for day-to-night videos. Synchronizes audio to video duration.

Features implemented:
  1. Multi-voice support (day / dusk / night profiles)
  2. Emotion-controlled synthesis (keyword parsing)
  3. Context-aware synthesis (pause tokens)
  4. Audio-video synchronization (time-stretch)

Usage:
    python tts/tts_pipeline.py \
        --video outputs/enhanced/test.gif \
        --prompt "cinematic time-lapse of a city skyline..." \
        --output outputs/final_with_audio.mp4
"""

import argparse
import re
import tempfile
from pathlib import Path

import numpy as np


# Feature 1: Multi-Voice Profiles
VOICE_PROFILES = {
    "narrator_day":   {"style": "warm",    "pitch": 1.05, "rate": 1.0,  "energy": 0.7},
    "narrator_dusk":  {"style": "calm",    "pitch": 0.95, "rate": 0.95, "energy": 0.5},
    "narrator_night": {"style": "mystery", "pitch": 0.85, "rate": 0.9,  "energy": 0.3},
}


def select_voice(frame_index: int, num_frames: int) -> dict:
    """Select voice profile based on temporal position."""
    t = frame_index / (num_frames - 1)
    if t < 0.35:
        return VOICE_PROFILES["narrator_day"]
    elif t < 0.65:
        return VOICE_PROFILES["narrator_dusk"]
    else:
        return VOICE_PROFILES["narrator_night"]


# Feature 2: Emotion-Controlled Synthesis
EMOTION_MAP = {
    "bustling":   {"energy": "high",  "pace": 1.15, "pitch_shift": 0.05},
    "calm":       {"energy": "low",   "pace": 0.90, "pitch_shift": -0.05},
    "night":      {"energy": "low",   "pace": 0.85, "pitch_shift": -0.10},
    "rush":       {"energy": "high",  "pace": 1.25, "pitch_shift": 0.10},
    "neon":       {"energy": "high",  "pace": 1.10, "pitch_shift": 0.0},
    "sunset":     {"energy": "mid",   "pace": 0.95, "pitch_shift": -0.03},
    "rain":       {"energy": "low",   "pace": 0.90, "pitch_shift": -0.08},
    "festival":   {"energy": "high",  "pace": 1.20, "pitch_shift": 0.08},
}


def parse_emotion(prompt_text: str) -> dict:
    """Parse emotional keywords from prompt."""
    prompt_lower = prompt_text.lower()
    params = {"energy": "neutral", "pace": 1.0, "pitch_shift": 0.0}
    for keyword, emo_params in EMOTION_MAP.items():
        if keyword in prompt_lower:
            params.update(emo_params)
    return params


# Feature 3: Context-Aware Synthesis
def insert_pause_tokens(text: str) -> str:
    """Insert pause markers for natural breathing."""
    text = text.replace(",", ", <|pause|>")
    text = text.replace(".", ". <|pause|><|pause|>")
    scene_markers = ["as the sun sets", "meanwhile", "gradually", "then"]
    for marker in scene_markers:
        text = text.replace(marker, f"<|pause|><|pause|> {marker}")
    return text


# Narration Script Generation
def generate_narration_script(prompt_text: str, num_frames: int) -> list[tuple[int, int, str]]:
    """Generate time-stamped narration segments aligned to frames."""
    segments = []

    # Phase 1: Day (frames 0-5)
    if any(w in prompt_text for w in ["skyline", "city"]):
        segments.append((0, 5, "The city awakens under a brilliant blue sky."))
    else:
        segments.append((0, 5, "Daylight bathes the scene in warm golden hues."))

    # Phase 2: Golden Hour (frames 5-8)
    segments.append((5, 8, "Shadows grow longer as the golden hour approaches."))

    # Phase 3: Dusk (frames 8-11)
    segments.append((8, 11, "The sky transitions through shades of orange and purple."))
    if "neon" in prompt_text:
        segments.append((9, 11, "Neon signs begin to flicker to life."))

    # Phase 4: Night (frames 11-15)
    if "lights" in prompt_text or "neon" in prompt_text:
        segments.append((11, 15, "The city transforms under a blanket of artificial lights."))
    else:
        segments.append((11, 15, "Night settles in, revealing a different world."))

    return segments


# Feature 4: Audio-Video Synchronization
def sync_audio_to_video(audio_path: str, video_duration_sec: float, output_path: str):
    """Time-stretch audio to match video duration."""
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("WARNING: librosa/soundfile not available, skipping sync")
        return

    y, sr = librosa.load(audio_path)
    audio_duration = len(y) / sr
    if audio_duration == 0:
        return
    stretch_rate = audio_duration / video_duration_sec
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
    sf.write(output_path, y_stretched, sr)


def get_frame_count(video_path: str) -> int:
    """Count frames in a video or GIF file."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    return count


def composite_audio_video(video_path: str, audio_path: str, output_path: str):
    """Merge audio and video using ffmpeg."""
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True)


class QwenTTSPipeline:
    """
    Qwen-TTS pipeline for narrated video generation.
    Falls back to placeholder if Qwen model is unavailable.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model.eval()
            print(f"Loaded Qwen-TTS: {self.model_name}")
        except Exception as e:
            print(f"WARNING: Could not load Qwen-TTS ({e}). Using placeholder mode.")
            self.model = None

    def synthesize(self, text: str, voice_profile: dict) -> np.ndarray:
        """Synthesize audio for a text segment. Returns audio array."""
        if self.model is None:
            # Placeholder: return silence
            sr = 22050
            duration_sec = len(text.split()) * 0.5  # rough estimate
            return np.zeros(int(sr * duration_sec), dtype=np.float32)

        # Qwen-TTS synthesize call (model-specific API)
        try:
            audio = self.model.synthesize(text, voice_profile=voice_profile)
            return audio
        except Exception as e:
            print(f"Synthesis failed: {e}")
            sr = 22050
            duration_sec = len(text.split()) * 0.5
            return np.zeros(int(sr * duration_sec), dtype=np.float32)

    def generate_narration(self, prompt_text: str, video_path: str, output_video: str):
        """Full pipeline: script → synthesize → sync → composite."""
        num_frames = get_frame_count(video_path)
        video_duration = num_frames / 8.0  # 8 FPS

        print(f"Video: {num_frames} frames, {video_duration:.2f}s")

        # Generate script
        segments = generate_narration_script(prompt_text, num_frames)
        emotion_profile = parse_emotion(prompt_text)

        # Synthesize per segment
        audio_segments = []
        for start_frame, end_frame, text in segments:
            text_with_pauses = insert_pause_tokens(text)
            voice = select_voice(start_frame, num_frames)
            merged_profile = {**voice, **emotion_profile}

            print(f"  [{start_frame}-{end_frame}] {text}")
            audio = self.synthesize(text_with_pauses, merged_profile)
            audio_segments.append(audio)

        # Concatenate audio
        combined_audio = np.concatenate(audio_segments) if audio_segments else np.array([])

        # Save temporary audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        if len(combined_audio) > 0:
            try:
                import soundfile as sf
                sf.write(tmp_path, combined_audio, 22050)
            except ImportError:
                print("WARNING: soundfile not available")
                return None

        # Sync to video duration
        synced_path = tmp_path.replace(".wav", "_synced.wav")
        sync_audio_to_video(tmp_path, video_duration, synced_path)

        # Composite
        composite_audio_video(video_path, synced_path, output_video)
        print(f"Final video with narration: {output_video}")
        return output_video


def main():
    parser = argparse.ArgumentParser(description="Qwen-TTS Narration Pipeline")
    parser.add_argument("--video", required=True, help="Input video/GIF path")
    parser.add_argument("--prompt", required=True, help="Original generation prompt")
    parser.add_argument("--output", default="outputs/final_with_audio.mp4")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    pipeline = QwenTTSPipeline(model_name=args.model)
    pipeline.generate_narration(args.prompt, args.video, args.output)


if __name__ == "__main__":
    main()
