# Bonus: Text-to-Audio — Qwen-TTS Integration

## Overview
Integrate Qwen-TTS as a post-processing module that generates synchronized, expressive narration for the generated day-to-night videos. This goes beyond basic TTS by implementing multiple advanced features.

## Model
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (or similar Qwen-based TTS)
- Neural TTS with voice profile conditioning
- Runs on CPU or GPU (~2GB VRAM if GPU)

---

## Required Advanced Features

We implement **4 of 6** required features (minimum is 2 for full bonus credit).

### Feature 1: Multi-Voice Support

Define distinct voice profiles for different phases of the day-to-night transition:

```python
VOICE_PROFILES = {
    "narrator_day":   {"style": "warm",    "pitch": 1.05, "rate": 1.0,  "energy": 0.7},
    "narrator_dusk":  {"style": "calm",    "pitch": 0.95, "rate": 0.95, "energy": 0.5},
    "narrator_night": {"style": "mystery", "pitch": 0.85, "rate": 0.9,  "energy": 0.3},
}

def select_voice(frame_index, num_frames):
    t = frame_index / (num_frames - 1)
    if t < 0.35:   return VOICE_PROFILES["narrator_day"]
    elif t < 0.65: return VOICE_PROFILES["narrator_dusk"]
    else:          return VOICE_PROFILES["narrator_night"]
```

### Feature 2: Emotion-Controlled Synthesis

Parse the prompt for emotional keywords and adjust TTS parameters:

```python
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

def parse_emotion(prompt_text):
    prompt_lower = prompt_text.lower()
    params = {"energy": "neutral", "pace": 1.0, "pitch_shift": 0.0}
    for keyword, emo_params in EMOTION_MAP.items():
        if keyword in prompt_lower:
            params.update(emo_params)
    return params
```

### Feature 3: Context-Aware Synthesis

Insert pause tokens based on punctuation and clause structure:

```python
import re

def insert_pause_tokens(text):
    """Insert <|pause|> tokens for natural breathing."""
    # Comma → 150ms pause
    text = text.replace(",", ", <|pause|>")
    # Period → 300ms pause  
    text = text.replace(".", ". <|pause|><|pause|>")
    # Scene transitions → 500ms pause
    scene_markers = ["as the sun sets", "meanwhile", "gradually", "then"]
    for marker in scene_markers:
        text = text.replace(marker, f"<|pause|><|pause|> {marker}")
    return text
```

### Feature 4: Synchronization

Time-stretch generated audio to exactly match video duration:

```python
import librosa
import soundfile as sf

def sync_audio_to_video(audio_path, video_duration_sec, output_path):
    y, sr = librosa.load(audio_path)
    audio_duration = len(y) / sr
    stretch_factor = audio_duration / video_duration_sec
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
    sf.write(output_path, y_stretched, sr)
```

---

## Narration Script Generation

```python
def generate_narration_script(prompt_text, num_frames):
    """Generate time-stamped narration segments aligned to frames."""
    segments = []
    t = lambda f: f / (num_frames - 1)
    
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
```

---

## Full TTS Pipeline

```python
# tts/tts_pipeline.py
class QwenTTSPipeline:
    def __init__(self, model_name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
    
    def generate_narration(self, prompt_text, video_path):
        num_frames = get_frame_count(video_path)
        video_duration = num_frames / 8.0  # 8 FPS
        
        # Generate script
        segments = generate_narration_script(prompt_text, num_frames)
        voice_profile = parse_emotion(prompt_text)
        
        # Synthesize per segment
        audio_segments = []
        for start_frame, end_frame, text in segments:
            text = insert_pause_tokens(text)
            voice = select_voice(start_frame, num_frames)
            
            audio = self.model.synthesize(
                text,
                voice_profile={**voice, **voice_profile},
            )
            audio_segments.append(((start_frame / 8.0), audio))
        
        # Merge and sync
        combined = self.merge_audio_segments(audio_segments, video_duration)
        sync_audio_to_video(combined, video_duration, "outputs/narration.wav")
        
        # Composite with video
        self.composite_audio_video(video_path, "outputs/narration.wav", "outputs/final_with_audio.mp4")
        return "outputs/final_with_audio.mp4"
```

---

## Evaluation of Audio

| Criterion | Method | Target |
|-----------|--------|--------|
| Voice distinctness | User survey (1-5) | > 4.0 for voice switching clarity |
| Emotion recognizability | User survey (1-5) | > 4.0 for matching emotion to scene |
| Sync accuracy | Frame-to-audio offset | < 50ms |
| Overall quality | Mean Opinion Score (MOS) | > 4.0 |

---

## Deliverables
- [ ] `tts/tts_pipeline.py` — complete TTS pipeline
- [ ] Multi-voice support (day/dusk/night profiles)
- [ ] Emotion keyword parser
- [ ] Context-aware pause insertion
- [ ] Audio-video synchronization
- [ ] Final narrated video (≥4s, day-to-night, narrated)
