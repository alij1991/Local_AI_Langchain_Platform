# Image Editor & Enhancement Section — Implementation Prompt

## Context

You are working on **Local AI Langchain Platform** — a desktop application with a Python FastAPI backend and Flutter frontend. The app runs AI models **locally** on the user's hardware.

**User's hardware**: RTX 4060 Laptop, 8GB VRAM, 32GB RAM, Windows 11, Python 3.11

The app already has a **comprehensive image generation system** (text2img, img2img, inpaint, ControlNet, LoRA, upscaling, batch, schedulers, step previews) in `src/local_ai_platform/images/service.py` (6700+ lines). The new Image Editor section is a **separate page** that focuses on **non-generative and hybrid AI+classical editing operations** — things you do *after* generating an image or to *any* image you bring in.

---

## Goal

Build a dedicated **Image Editor & Enhancement** section that combines:
1. **AI-powered operations** (background removal, face restoration, super-resolution, style transfer, object removal, instruction-based editing)
2. **Classical image processing** (crop, resize, rotate, brightness/contrast, color correction, filters, sharpening, noise reduction)
3. **A non-destructive editing pipeline** with undo/redo, layers, and before/after comparison

The editor must work **entirely locally** — no cloud APIs. All ML models run on the user's GPU or CPU.

---

## Current State (What Exists)

### Image System Already Built
- `src/local_ai_platform/images/service.py` — 6700 lines, full diffusers pipeline
- Supports: text2img, img2img, inpainting (mask-based), ControlNet (6 types), LoRA, upscaling (RealESRGAN + LANCZOS), batch generation, 9 schedulers, step previews, prompt enhancement
- `apply_basic_edit()` — very limited: only rotate 90, flip, grayscale via keyword matching
- `upscale_image()` — RealESRGAN 4x (optional install) + LANCZOS fallback
- Quality profiles: fast/balanced/quality/low_memory
- Device auto-detection: CUDA > OpenVINO > CPU

### Database
```sql
CREATE TABLE image_sessions (id TEXT PK, title TEXT, created_at, updated_at);
CREATE TABLE images (id TEXT PK, session_id FK, parent_image_id FK, model_id, operation, prompt, negative_prompt, params_json, file_path, run_id, created_at);
```
Operations tracked: "generate", "edit", "upscale" — chained via `parent_image_id`

### API Endpoints (Image-Related)
```
POST /images/generate          — text2img/img2img/inpaint/controlnet/batch
POST /images/edit              — Routes to generate() with init_image
POST /images/upscale           — RealESRGAN or LANCZOS
POST /images/preprocess        — ControlNet preprocessing preview
GET  /images/controlnet/types  — Available ControlNet types
GET  /images/files/{sid}/{fn}  — Serve image file
GET  /images/generate/progress — Real-time progress
POST /images/generate/cancel   — Kill running job
POST /images/enhance-prompt    — LLM prompt improvement
GET  /images/models            — List image models
GET  /images/runtime           — Device/VRAM status
GET  /images/schedulers        — Available schedulers
```

### Flutter Images Page (`images_page.dart`, 2000+ lines)
- Left panel: session list
- Center: image viewer (single image, no canvas/layers)
- Right panel: generation controls (model, prompt, params)
- Inpainting: basic brush strokes → mask → send to API
- No classical editing tools (crop, brightness, filters, etc.)
- No comparison view, no undo/redo, no layers

### Dependencies Already Installed
```
diffusers>=0.30.0, torch, torchvision, Pillow>=10.0.0
opencv-python-headless, numpy, controlnet-aux>=0.0.9
```

### Dependencies NOT Installed (Needed)
```
rembg              — Background removal (u2net)
realesrgan         — Super-resolution (4x upscale)
gfpgan             — Face restoration (generative)
basicsr            — Required by realesrgan/gfpgan
onnxruntime        — Already installed (for rembg)
```

---

## What to Build

### Architecture: Image Editing Service

Create a new **ImageEditorService** class (separate from the existing ImageGenerationService) that handles all editing operations. Keep the generation service for generation, and the editor service for post-processing and editing.

```
src/local_ai_platform/images/
    __init__.py
    service.py          — Existing generation service (6700 lines, DON'T TOUCH)
    editor.py           — NEW: ImageEditorService
    processors.py       — NEW: Classical image processing functions
    ai_enhance.py       — NEW: AI enhancement models (bg removal, face fix, etc.)
```

### 1. Classical Image Processing (`processors.py`)

Pure Pillow + OpenCV operations, no ML models. All operations take a PIL Image and return a PIL Image.

**Crop & Transform:**
```python
def crop(image, x, y, width, height) -> Image
def resize(image, width, height, maintain_aspect=True) -> Image
def rotate(image, degrees, expand=True) -> Image
def flip_horizontal(image) -> Image
def flip_vertical(image) -> Image
def auto_crop(image, threshold=10) -> Image  # Trim blank borders
```

**Color & Exposure:**
```python
def adjust_brightness(image, factor: float) -> Image    # 0.0-2.0, 1.0=original
def adjust_contrast(image, factor: float) -> Image
def adjust_saturation(image, factor: float) -> Image
def adjust_sharpness(image, factor: float) -> Image
def adjust_color_temperature(image, kelvin: int) -> Image  # Warm/cool shift
def auto_levels(image) -> Image                          # Auto stretch histogram
def auto_white_balance(image) -> Image                   # Gray-world algorithm
def adjust_curves(image, points: list) -> Image          # Tone curve adjustment
def adjust_hue(image, shift: int) -> Image               # Hue rotation (-180 to 180)
```

**Filters & Effects:**
```python
def blur(image, radius: float) -> Image
def gaussian_blur(image, sigma: float) -> Image
def sharpen(image, amount: float) -> Image               # Unsharp mask
def denoise(image, strength: int) -> Image               # OpenCV fastNlMeansDenoisingColored
def edge_detect(image) -> Image                          # Canny/Sobel
def emboss(image) -> Image
def vignette(image, intensity: float) -> Image
def grain(image, amount: float) -> Image                 # Film grain noise
```

**Format & Output:**
```python
def convert_format(image, format: str) -> bytes          # PNG, JPEG, WebP
def compress_jpeg(image, quality: int) -> bytes
def strip_metadata(image) -> Image                       # Remove EXIF
def add_watermark(image, text: str, opacity: float) -> Image
```

### 2. AI Enhancement Models (`ai_enhance.py`)

Each AI operation is a self-contained function that lazily loads its model on first use and caches it. Models are downloaded on demand from HuggingFace.

**Background Removal:**
```python
def remove_background(image: Image, model: str = "u2net") -> Image:
    """Remove background using rembg. Returns RGBA image with transparent background."""
    # Uses: rembg library (u2net, u2net_human_seg, isnet-general-use)
    # Models: ~170MB, auto-downloaded on first use
    # Fallback: If rembg not installed, raise clear error

def replace_background(image: Image, background: Image | str) -> Image:
    """Remove background and composite onto new background (image or solid color)."""
```

**Face Restoration:**
```python
def restore_faces(image: Image, model: str = "gfpgan") -> Image:
    """Fix blurry/damaged faces using GFPGAN or CodeFormer."""
    # Models: GFPGAN v1.4 (~350MB), CodeFormer (~400MB)
    # Auto-detects faces, enhances each, composites back
    # Supports: gfpgan, codeformer
    # Fallback: return original if no faces detected

def detect_faces(image: Image) -> list[dict]:
    """Detect face bounding boxes and landmarks."""
    # Returns: [{"bbox": [x1,y1,x2,y2], "confidence": 0.98, "landmarks": [...]}]
```

**Super-Resolution / Upscaling:**
```python
def upscale(image: Image, scale: int = 4, model: str = "realesrgan") -> Image:
    """AI upscale using RealESRGAN. Falls back to LANCZOS."""
    # Models: RealESRGAN_x4plus (~65MB), RealESRGAN_x4plus_anime (~65MB)
    # Selectable: realesrgan, realesrgan_anime, lanczos

def upscale_face(image: Image, scale: int = 2) -> Image:
    """Face-aware upscaling using GFPGAN + RealESRGAN pipeline."""
    # First upscale with RealESRGAN, then enhance faces with GFPGAN
```

**Instruction-Based Editing (AI):**
```python
def instruct_edit(image: Image, instruction: str, strength: float = 1.0) -> Image:
    """Edit image based on text instruction using InstructPix2Pix."""
    # Model: timbrooks/instruct-pix2pix (~5GB, SD 1.5 based)
    # Input: "make it sunset", "add snow", "make the car red"
    # Uses diffusers StableDiffusionInstructPix2PixPipeline
    # Runs in subprocess like other diffusers operations (VRAM management)
```

**Style Transfer (IP-Adapter):**
```python
def apply_style(image: Image, style_image: Image, strength: float = 0.8) -> Image:
    """Transfer style from reference image using IP-Adapter."""
    # Uses IP-Adapter with SDXL or SD 1.5 pipeline
    # Style-only injection (InstantStyle approach: up_blocks + down_blocks only)
```

**Object Removal / Inpainting Enhancement:**
```python
def remove_object(image: Image, mask: Image) -> Image:
    """Remove an object from the image using AI inpainting."""
    # Uses the existing inpainting pipeline but with optimized prompt
    # Prompt: "clean background, empty space, seamless texture"
    # Better than raw inpaint for object removal specifically

def auto_segment(image: Image, point: tuple[int, int] | None = None) -> Image:
    """Auto-segment objects for mask generation."""
    # If point provided: segment object at that point
    # Uses simple color/edge-based segmentation (OpenCV GrabCut)
    # For better results: can use rembg's trimap-based approach
```

**Image Denoising & Restoration:**
```python
def denoise_ai(image: Image, strength: float = 0.5) -> Image:
    """AI-powered noise reduction (better than OpenCV for heavy noise)."""
    # Light: OpenCV fastNlMeansDenoising
    # Heavy: SD img2img at very low strength (0.1-0.3) to clean noise

def remove_artifacts(image: Image) -> Image:
    """Remove JPEG artifacts and compression noise."""
    # Uses img2img with very low denoising strength
```

### 3. Image Editor Service (`editor.py`)

Orchestrates classical + AI operations with undo/redo and edit history.

```python
class EditOperation:
    """A single edit step in the history."""
    op_type: str           # "crop", "brightness", "remove_bg", "upscale", etc.
    params: dict           # Operation parameters
    timestamp: datetime
    preview_path: str      # Path to result image

class ImageEditorService:
    """Non-destructive image editing with history tracking."""

    def __init__(self, image_service: ImageGenerationService):
        self.image_service = image_service  # For AI operations that need diffusers
        self._edit_sessions: dict[str, EditSession] = {}

    def open_image(self, session_id: str, image_path: str) -> dict:
        """Open an image for editing. Creates edit session."""

    def apply_edit(self, session_id: str, operation: str, params: dict) -> dict:
        """Apply an edit operation. Returns new image path + metadata."""
        # Dispatches to processors.py or ai_enhance.py based on operation type

    def undo(self, session_id: str) -> dict:
        """Undo last edit. Returns previous image."""

    def redo(self, session_id: str) -> dict:
        """Redo last undone edit."""

    def get_history(self, session_id: str) -> list[dict]:
        """Get edit history for session."""

    def compare(self, session_id: str, step_a: int, step_b: int) -> dict:
        """Get two images for side-by-side comparison."""

    def export(self, session_id: str, format: str, quality: int) -> bytes:
        """Export final image in requested format."""

    def get_available_operations(self) -> list[dict]:
        """List all available operations with their parameters and requirements."""
        # Returns categorized list: which operations need GPU, which models
        # are installed, estimated time, etc.
```

### 4. API Endpoints

```
# ── Editor Session Management ──
POST   /editor/open                — Open image for editing (from file, URL, or session image)
GET    /editor/{session_id}        — Get edit session state (current image, history)
DELETE /editor/{session_id}        — Close edit session

# ── Edit Operations ──
POST   /editor/{session_id}/edit   — Apply an edit operation
POST   /editor/{session_id}/undo   — Undo last edit
POST   /editor/{session_id}/redo   — Redo
GET    /editor/{session_id}/history — Get full edit history with thumbnails

# ── Comparison ──
GET    /editor/{session_id}/compare?a=0&b=current — Compare two versions

# ── Export ──
POST   /editor/{session_id}/export — Export final image (format, quality)

# ── AI Operations (may take longer, support progress) ──
POST   /editor/{session_id}/ai/remove-background
POST   /editor/{session_id}/ai/restore-faces
POST   /editor/{session_id}/ai/upscale
POST   /editor/{session_id}/ai/instruct-edit     — {instruction: "make it look like winter"}
POST   /editor/{session_id}/ai/style-transfer     — {style_image_path: "..."}
POST   /editor/{session_id}/ai/remove-object      — {mask_base64: "..."}
POST   /editor/{session_id}/ai/denoise

# ── Info ──
GET    /editor/operations          — List available operations + their status (installed/needs model)
GET    /editor/models              — List installed AI models for editing
POST   /editor/models/install      — Download a model (rembg, gfpgan, realesrgan)
```

### 5. Edit Operation Request Format

```json
{
  "operation": "brightness",
  "params": {
    "factor": 1.3
  }
}
```

For AI operations:
```json
{
  "operation": "remove_background",
  "params": {
    "model": "u2net",
    "return_mask": false
  }
}
```

### 6. Flutter Editor Page (`editor_page.dart`)

A new page in the navigation rail (between Images and Runs, or as a tab within Images).

**Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Toolbar: [Undo] [Redo] [Compare] [Export] [Reset]      │
├───────────┬─────────────────────────────┬───────────────┤
│           │                             │               │
│  Edit     │     Canvas / Image          │  Properties   │
│  Tools    │     (zoomable, pannable)    │  Panel        │
│  Panel    │                             │  (sliders,    │
│  (icons)  │     [Before] ← → [After]   │   params for  │
│           │     comparison slider       │   selected    │
│           │                             │   operation)  │
│           │                             │               │
├───────────┴─────────────────────────────┴───────────────┤
│  History Strip: [Original] → [Crop] → [Brightness] → …  │
└─────────────────────────────────────────────────────────┘
```

**Left Panel — Edit Tools (icon buttons, grouped):**
```
TRANSFORM
  [Crop] [Resize] [Rotate] [Flip H] [Flip V]

ADJUST
  [Brightness] [Contrast] [Saturation] [Sharpness]
  [Temperature] [Hue] [Auto Levels] [Auto WB]

FILTERS
  [Blur] [Sharpen] [Denoise] [Grain] [Vignette]

AI ENHANCE
  [Remove BG] [Restore Faces] [Upscale] [AI Denoise]

AI EDIT
  [Instruct Edit] [Style Transfer] [Remove Object]
```

**Center — Canvas:**
- Zoomable + pannable image viewer with `InteractiveViewer`
- Before/After comparison slider (horizontal drag divider)
- For crop: draggable crop rectangle overlay
- For remove object: brush tool to paint mask

**Right Panel — Properties:**
- Shows controls for the selected tool
- Sliders for adjustment values (brightness factor, blur radius, etc.)
- Model selector for AI operations
- "Apply" button to execute
- Preview toggle for real-time preview (classical ops only)

**Bottom — History Strip:**
- Horizontal scroll of thumbnails representing each edit step
- Click to jump to any step (like undo to specific point)
- Current step highlighted

**Comparison View:**
- Side-by-side mode: two images next to each other
- Slider mode: draggable vertical line reveals before/after
- Toggle between original vs current, or any two history steps

### 7. Database Changes

```sql
-- Editor sessions (separate from generation sessions)
CREATE TABLE IF NOT EXISTS editor_sessions (
    id TEXT PRIMARY KEY,
    source_image_path TEXT NOT NULL,    -- Original image
    current_image_path TEXT NOT NULL,   -- Current state
    source_type TEXT,                   -- "file", "generated", "url"
    source_session_id TEXT,            -- FK to image_sessions if from generation
    source_image_id TEXT,              -- FK to images if from generation
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Edit operations history
CREATE TABLE IF NOT EXISTS edit_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    operation TEXT NOT NULL,            -- "crop", "brightness", "remove_bg", etc.
    params_json TEXT,                  -- Operation parameters
    result_image_path TEXT NOT NULL,   -- Path to result
    duration_ms INTEGER,              -- How long the operation took
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES editor_sessions(id) ON DELETE CASCADE
);
```

### 8. Navigation Integration

Add "Editor" to the navigation rail in `studio_shell.dart`:
```dart
NavigationRailDestination(
  icon: Icon(Icons.photo_filter),
  selectedIcon: Icon(Icons.photo_filter),
  label: Text('Editor'),
),
```

Position: After "Images" (index 7) or as a sub-tab within the Images section.

### 9. Opening Images in Editor

Users should be able to open images in the editor from:
1. **Images page** — "Edit" button on any generated image opens it in the editor
2. **File picker** — "Open Image" button in editor loads from disk
3. **Chat page** — When agent generates an image, "Edit" button opens it in editor
4. **Drag & drop** — Drop an image file onto the editor page

---

## Implementation Priorities

### Phase 1: Foundation (Editor Service + Classical Ops)
1. Create `processors.py` with all classical operations
2. Create `editor.py` with EditSession, undo/redo, history
3. Add `editor_sessions` + `edit_history` tables to db.py
4. Add editor API endpoints (open, edit, undo, redo, history, export)
5. Build basic Flutter editor page with canvas + tool panel + history strip
6. Implement crop, resize, rotate, flip in Flutter with interactive overlays
7. Implement brightness/contrast/saturation sliders with live preview

### Phase 2: AI Enhancement
8. Implement background removal (rembg)
9. Implement face restoration (GFPGAN)
10. Implement AI upscaling (RealESRGAN — complete the existing partial code)
11. Implement AI denoising
12. Add model management endpoint (install/check models)
13. Add progress tracking for AI operations
14. Update Flutter with AI tool buttons and model status indicators

### Phase 3: Advanced AI Editing
15. Implement InstructPix2Pix instruction-based editing
16. Implement style transfer via IP-Adapter
17. Implement object removal (mask painting + AI inpainting)
18. Add mask painting tool in Flutter (brush + eraser on canvas)
19. Add comparison slider in Flutter

### Phase 4: Polish
20. Before/after comparison view
21. History strip with thumbnails
22. Export dialog (format, quality, dimensions)
23. Open from Images page / Chat page integration
24. Keyboard shortcuts (Ctrl+Z undo, Ctrl+Shift+Z redo)

---

## Key Design Principles

1. **Non-destructive editing**: Every operation creates a new image file. The original is never modified. Users can undo to any point in history.

2. **Lazy model loading**: AI models are loaded on first use and cached in memory. The first AI operation is slow (model download + load), subsequent ones are fast.

3. **VRAM-aware**: Before running an AI operation, check available VRAM. If insufficient, either use CPU fallback or warn the user. Never crash from OOM.

4. **Classical ops are instant**: Pillow/OpenCV operations should complete in <100ms. No loading indicators needed. Live preview where possible.

5. **AI ops show progress**: AI operations take 2-30 seconds. Show progress indicators. Support cancellation.

6. **Subprocess isolation for heavy AI**: InstructPix2Pix and style transfer use the existing diffusers subprocess pattern (same as image generation) to avoid VRAM conflicts with loaded LLMs.

7. **Everything works offline**: No cloud APIs. All models downloadable from HuggingFace and cached locally.

8. **Reuse existing infrastructure**: Use the existing image file serving (`/images/files/`), device detection, VRAM checking, and subprocess isolation patterns from `service.py`. Don't rebuild what exists.

---

## Technical Constraints

- **Python 3.11** on Windows 11
- **8GB VRAM** — Most AI models need 1-4GB. Can't run simultaneously with LLM + diffusion model
- **SQLite** for persistence (existing pattern)
- **FastAPI** backend + **Flutter** frontend
- **No Docker** — native execution
- **Pillow + OpenCV** for classical ops (already installed)
- **rembg, gfpgan, realesrgan** for AI ops (need to be added as optional deps)
- Image files stored in `data/images/editor/{session_id}/` directory
- All operations return the new image path and metadata (duration, dimensions, file size)

---

## Dependencies to Add (`pyproject.toml`)

```toml
# Image editor AI models (optional — graceful fallback if missing)
editor = [
  "rembg>=2.0.50",                   # Background removal (u2net, ~170MB model)
  "gfpgan>=1.3.8",                   # Face restoration (~350MB model)
  "realesrgan>=0.3.0",               # Super-resolution (~65MB model)
  "basicsr>=1.4.2",                  # Required by gfpgan/realesrgan
]
```
