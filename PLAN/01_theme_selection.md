# Theme Selection — Urban Metamorphosis

## Theme Concept
Generate short video clips showing smooth, cinematic transitions from daytime to nighttime in urban environments. The focus is on **city scenes** where the lighting shift creates dramatic visual changes: sky gradient, shadows, street lights, neon signs, reflections.

## Why This Theme
| Factor | Benefit |
|--------|---------|
| **Clear Visual Markers** | Sky color, shadows, street lights, window glow — easily measured |
| **Semantic Gradation** | Day → Golden Hour → Dusk → Night — natural CLIP-SIM test |
| **Flickering Visibility** | Lighting jumps are immediately noticeable — strong LPIPS signal |
| **Evaluable** | Per-frame CLIP-SIM curves clearly show alignment quality |
| **Visually Impressive** | Time-lapse city transitions are universally appealing |

## Prompt Engineering

### Base Prompt Template
```
cinematic time-lapse of {urban_scene} transitioning from day to night, 
{city_style}, {lighting_detail}, {camera_motion}, 4k, highly detailed
```

### Urban Scene Variations (50 total for eval set)
| Category | Examples | Count |
|----------|----------|-------|
| **Skyline** | city skyline, downtown skyline, waterfront skyline | 10 |
| **Street Level** | busy street, narrow alley, boulevard,霓虹 street | 10 |
| **Transit** | train station, highway, bridge, intersection | 8 |
| **Waterfront** | harbor, pier, riverwalk, canal | 6 |
| **Parks/Plazas** | city park, town square, rooftop terrace | 6 |
| **Special** | rain reflections, foggy dusk, festival lights | 10 |

### 10 "Break Temporal Consistency" Prompts
Designed to expose flickering:
1. "sunset over a city skyline, sudden darkness"
2. "street with flickering neon signs at dusk"
3. "rapid day-to-night transition in a bustling market"
4. "city bridge with passing cars, headlights turning on"
5. "alley with alternating shadow and light at sunset"
6. "city skyline, clouds moving fast, lightning transition"
7. "intersection with traffic lights changing at dusk"
8. "harbor with boats, sunset reflecting on rippling water"
9. "skyline with skyscrapers, windows lighting up one by one"
10. "city street, rain starts exactly at dusk"

### 10 "Semantic Alignment" Prompts
Designed to test per-frame semantic accuracy:
1. "neon signs flicker on as the last sunlight fades"
2. "street lamps cast warm glow on wet pavement after sunset"
3. "city park, golden hour shadows, then blue hour"
4. "rooftop view, sky transitions from orange to deep purple"
5. "busy plaza, daytime crowds thin out as evening sets in"
6. "highway overpass, car headlights stream into darkness"
7. "waterfront, sunset reflections transition to city lights"
8. "downtown street, glass buildings reflect golden to night"
9. "train station, natural light fades, artificial lights take over"
10. "alley with string lights, day to night with increasing glow"

### Negative Prompt
```
static, flickering, inconsistent lighting, sudden change, abrupt transition,
jittery, low quality, blurry, distorted, deformed, flat lighting,
oversaturated, undersaturated, posterized
```

## Day/Night Prompt Pairs (for Enhancement A)
| Scene | Day Prompt | Night Prompt |
|-------|-----------|-------------|
| Skyline | "bright sunny day in a city, clear blue sky" | "dark night city skyline, neon lights, starry sky" |
| Street | "sunlit city street, bright daylight, shadows" | "night street with street lamps, neon glow, dark" |
| Harbor | "sunny harbor, bright blue water, clear sky" | "moonlit harbor, city lights reflection, dark water" |
| Park | "sunlit city park, green trees, bright day" | "night park, warm street lamps, dark sky" |
| Transit | "bright daylight at train station, sunny" | "night train station, artificial lights, dark" |

## Evaluation Dataset
- **Source:** Custom JSON file with 50 prompts
- **File:** `data/eval_prompts.json`
- **Structure:**
```json
[
  {
    "id": 0,
    "category": "skyline",
    "prompt": "cinematic time-lapse of a city skyline transitioning from day to night",
    "prompt_day": "bright sunny day in a city, clear blue sky",
    "prompt_night": "dark night city skyline, neon lights, starry sky",
    "difficulty": "normal",
    "test_aspect": "general"
  },
  ...
]
```
