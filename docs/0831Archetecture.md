Below is the **Initial Architecture Specification (v0)** for your Azur Lane bot. It is written so engineering agents can implement it end‑to‑end without needing additional product notes or future roadmap. Everything here is **in scope** for v0.

---

# Azur Lane Bot — Initial Architecture Specification (v0)

**Target OS:** Debian Linux
**Target Android environment:** **Waydroid** (single supported emulator for v0)
**Control stack:** ADB (primary) + optional minitouch (pluggable)
**Capture:** `adb exec-out screencap -p` (PNG)
**UI:** Desktop, **PySide6** (always-on window)
**Vision/Reasoning:** LLM (Gemini Flash 2.5) + OCR + classical CV
**Primary goals (v0):**

1. Collect all easy **main-menu pickups**.
2. Navigate to **Commissions** and read/record current commissions.
3. Read and record **currency balances** (Oil, Coins, Gems; Cubes optional).

> All actions must be **resolution-agnostic** and **model-agnostic** via selectors and a resolver. No raw pixel positions hard-coded.

---

## 1. Top-level Design

**Loop:** *Sense → Think → Act → Check*

1. **Sense:** Capture screen (PNG), detect letterboxing, normalize.
2. **Think (LLM):** Given the goal + last frame, propose a minimal plan in strict JSON (schema below).
3. **Resolve:** Convert LLM selectors to concrete coordinates using OCR + template + feature matching (ensemble, scale-invariant).
4. **Act:** Execute taps/swipes/back via ADB/minitouch.
5. **Check:** Recapture; verify success predicates; log; continue or recover.

**Key invariants**

* All coordinates internally are **normalized \[0..1]** with origin at top-left **after letterboxing crop**.
* The LLM never dictates raw pixels; it emits **selectors** (text/icon/region/bbox/point) that the **resolver** validates.
* Numeric truth (balances, timers) flows through **OCR** (never trust LLM text for numbers).

---

## 2. Repository Layout

```
azl_bot/
  pyproject.toml
  README.md
  azl_bot/
    core/
      device.py
      actuator.py
      capture.py
      resolver.py
      ocr.py
      llm_client.py
      planner.py
      screens.py
      datastore.py
      loggingx.py
      configs.py
    tasks/
      pickups.py
      commissions.py
      currencies.py
    ui/
      app.py
      overlays.py
      state.py
    config/
      app.yaml
      selectors/
        synonyms.yaml
      templates/
        mail_icon.png
        missions_icon.png
        commissions_icon.png
        back_arrow.png
    data/
      migrations.sql
    tests/
      unit/
      integration/
      fixtures/
        frames/  # sample screenshots for CI
    scripts/
      waydroid_setup.sh
      run_gui.sh
      run_task.sh
```

---

## 3. Dependencies (pin versions in `pyproject.toml`)

* **Python** ≥ 3.10
* **ADB** (`android-tools-adb` package on Debian)
* `pydantic>=2`
* `numpy`, `opencv-python`
* `Pillow`
* `PySide6`
* **OCR**: `paddlepaddle`, `paddleocr` (primary)

  * optional fallback: `pytesseract`, `tesseract-ocr` (Debian)
* `requests` (LLM HTTP)
* `uvloop` (optional on Linux, for snappier I/O)
* `sqlalchemy` + `sqlite` (stdlib)
* `rapidfuzz` (string similarity)
* `loguru` or `structlog` (structured logs)

---

## 4. Configuration

**File:** `config/app.yaml`

```yaml
emulator:
  kind: waydroid
  adb_serial: "127.0.0.1:5555"        # required
  package_name: "com.example.AzurLane" # user to set (EN/JP/GL differ)
display:
  target_fps: 2
  orientation: "landscape"
  force_resolution: null               # e.g., "1920x1080" for testing only
llm:
  provider: "gemini"
  model: "flash-2.5"
  endpoint: "https://..."              # user sets
  api_key_env: "GEMINI_API_KEY"
  max_tokens: 2048
  temperature: 0.1
resolver:
  ocr: "paddle"                        # "tesseract" to switch
  thresholds:
    ocr_text: 0.75
    ncc_edge: 0.60
    ncc_gray: 0.70
    orb_inliers: 12
    combo_accept: 0.65
  regions:                             # normalized [x,y,w,h]
    top_bar:     [0.00, 0.00, 1.00, 0.12]
    bottom_bar:  [0.00, 0.85, 1.00, 0.15]
    left_panel:  [0.00, 0.12, 0.20, 0.73]
    center:      [0.20, 0.12, 0.60, 0.73]
    right_panel: [0.80, 0.12, 0.20, 0.73]
data:
  base_dir: "~/.azlbot"
logging:
  level: "INFO"
  keep_frames: true
  overlay_draw: true
ui:
  show_llm_json: false
  zoom_overlay: true
```

**Selectors synonyms:** `config/selectors/synonyms.yaml`

```yaml
Commissions: ["Commissions","Commission","委托","委託"]
Mailbox: ["Mail","Mailbox","信箱","メール"]
Missions: ["Missions","Tasks","任務"]
Gems: ["Gems","钻石","ダイヤ"]
Oil: ["Oil","燃料"]
Coins: ["Coins","金币","資金"]
Back: ["Back","返回","戻る"]
```

---

## 5. Data Model (SQLite, in `~/.azlbot/azl.db`)

**Schema (`data/migrations.sql`):**

```sql
CREATE TABLE runs (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,
  task TEXT NOT NULL,
  device_serial TEXT NOT NULL
);
CREATE TABLE actions (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  ts TEXT NOT NULL,
  screen TEXT,
  action TEXT NOT NULL,
  selector_json TEXT,
  method TEXT,
  point_norm_x REAL,
  point_norm_y REAL,
  confidence REAL,
  success INTEGER,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE TABLE currencies (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  oil INTEGER,
  coins INTEGER,
  gems INTEGER,
  cubes INTEGER
);
CREATE TABLE commissions (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  slot_id INTEGER,
  name TEXT,
  rarity TEXT,
  time_remaining_s INTEGER,
  status TEXT            -- "idle" | "in_progress" | "ready"
);
```

---

## 6. Core Interfaces

### 6.1 Device & Actuation

`core/device.py`

```python
class DeviceInfo(TypedDict):
    width: int
    height: int
    density: int

class Device:
    def __init__(self, serial: str): ...
    def info(self) -> DeviceInfo: ...
    def screencap_png(self) -> bytes: ...        # raw PNG from adb exec-out
    def key_back(self) -> None: ...
    def key_home(self) -> None: ...
```

`core/actuator.py`

```python
class Actuator:
    def __init__(self, device: Device, backend: Literal["adb","minitouch"]="adb"): ...
    def tap_norm(self, x: float, y: float) -> None: ...
    def swipe_norm(self, x1: float, y1: float, x2: float, y2: float, ms: int=200) -> None: ...
```

**Requirements**

* Pixel conversion uses **active area** (after letterbox detection).
* Debounce taps: suppress duplicate taps within 500 ms within 1% of screen.

### 6.2 Capture & Preprocessing

`core/capture.py`

```python
@dataclass
class Frame:
    png_bytes: bytes
    image_bgr: np.ndarray        # cropped to active area (no letterbox)
    full_w: int
    full_h: int
    active_rect: tuple[int,int,int,int]  # (x,y,w,h) in full pixels
    ts: float

class Capture:
    def __init__(self, device: Device): ...
    def grab(self) -> Frame: ...
    def detect_letterbox(self, img_bgr: np.ndarray) -> tuple[int,int,int,int]: ...
```

**Letterbox detection**

* Scan borders for uniform bands (near-black or static color).
* Always return an **active\_rect**; if none found, it’s the full frame.

### 6.3 OCR

`core/ocr.py`

```python
class OCRClient:
    def text_in_roi(self, img_bgr: np.ndarray, roi_norm: tuple[float,float,float,float],
                    numeric_only: bool=False) -> list[dict]:
        """
        Returns list[{text: str, conf: float, box_norm: [x,y,w,h]}]
        Coordinates are normalized to the input image (0..1).
        """
```

**PaddleOCR defaults**

* EN + CN model; adaptive thresholding; small dilation for UI fonts.
* Numeric-only path for currencies/timers.

### 6.4 LLM Client & Schemas

`core/llm_client.py`

```python
from pydantic import BaseModel
from typing import Literal, Optional

class Target(BaseModel):
    kind: Literal["text","icon","bbox","point","region"]
    value: Optional[str] = None
    bbox: Optional[list[float]] = None  # [x,y,w,h] normalized
    point: Optional[list[float]] = None # [x,y]
    region_hint: Optional[str] = None   # e.g., "bottom_bar"
    confidence: float

class Step(BaseModel):
    action: Literal["tap","swipe","wait","back","assert"]
    target: Optional[Target] = None
    rationale: Optional[str] = None

class Plan(BaseModel):
    screen: str
    steps: list[Step]
    done: bool

class LLMClient:
    def __init__(self, cfg): ...
    def propose_plan(self, frame: Frame, goal: dict, context: dict) -> Plan: ...
```

**System prompt (literal string in code):**

> You are a UI navigator for the mobile game **Azur Lane** running on an Android emulator.
> Respond **only** with JSON that conforms to the provided schema.
> Prefer **text/icon selectors** with `region_hint` over coordinates.
> Use short rationales. Never invent UI; if uncertain, return a single `back` step.

**Per-call user prompt template:**

* Inputs: base64 PNG of frame, device size, known region rectangles, current goal (e.g., `"open_commissions_and_list"`), last screen label if available.
* Instruction: “Return the minimal steps to achieve the goal. If already achieved, set `done=true`.”

### 6.5 Selector Resolver

`core/resolver.py`

```python
@dataclass
class Candidate:
    point: tuple[float,float]   # normalized (active area)
    confidence: float
    method: str                 # "ocr","template","orb","llm"

class Resolver:
    def __init__(self, cfg, ocr: OCRClient, templates_dir: Path): ...
    def resolve(self, selector: Target, frame: Frame) -> Candidate: ...
    def to_pixels(self, point_norm: tuple[float,float], frame: Frame) -> tuple[int,int]: ...
```

**Matching policy (must implement):**

1. **OCR text match** in `region_hint` ROI (if provided), using synonyms and Jaro–Winkler ≥ `ocr_text` threshold.
2. **Template NCC on edges** over multi-scale pyramid (0.70–1.30 step 0.05).
3. **ORB+RANSAC** feature matching for robust cases.
4. **LLM point/bbox** fallback if above fail and within expected region.
5. Confidence fusion: if two methods agree within 2% distance, average confidence +0.1 bonus.

**Safety rules:**

* Clamp to \[0..1]; reject points near edges (<1% margin) unless target is in top/bottom bars.
* Return `Candidate` with `confidence`; planner decides acceptance.

### 6.6 Planner & Execution

`core/planner.py`

```python
class Planner:
    def __init__(self, device: Device, capture: Capture, resolver: Resolver, ocr: OCRClient, llm: LLMClient, store, logger): ...
    def run_task(self, task: "Task") -> None: ...
    def run_step(self, step: Step, frame: Frame) -> bool: ...
```

**Execution contract:**

* For each step:

  * Resolve selector → pixel point.
  * Execute via `Actuator`.
  * Sleep 0.5–1.0s (configurable) → capture → **check predicate** (task supplies).
* On failure:

  * Retry with next best candidate.
  * If still failing, send **Back**, re-center to Home (up to 3 backs).

---

## 7. Screens & Regions

`core/screens.py`

```python
class Regions:
    top_bar = (0.00, 0.00, 1.00, 0.12)
    bottom_bar = (0.00, 0.85, 1.00, 0.15)
    left_panel = (0.00, 0.12, 0.20, 0.73)
    center = (0.20, 0.12, 0.60, 0.73)
    right_panel = (0.80, 0.12, 0.20, 0.73)

def expected_home_elements() -> list[str]:
    return ["Commissions","Missions","Mailbox"]
```

**Badge detection helper (for pickups):**

* HSV red mask in small circular ROIs near known icons; return boolean + centroid.

---

## 8. Tasks (Goal logic)

All tasks implement a common interface:

```python
class Task(Protocol):
    name: str
    def goal(self) -> dict: ...
    def success(self, frame: Frame, context: dict) -> bool: ...
    def on_success(self, planner: Planner, frame: Frame) -> None: ...
```

### 8.1 `tasks/currencies.py`

**Purpose:** Read Oil/Coins/Gems (and optional Cubes) from top bar.

* **Steps:** None (no navigation if values visible). If not on a screen with top bar, LLM navigates “Home”.
* **Extraction:** `OCRClient.text_in_roi(frame.image_bgr, Regions.top_bar, numeric_only=True)` with small per-label ROIs derived by LLM selectors or fixed offsets once label anchors are found.
* **Output:** Insert into `currencies` with OCR confidence averages.
* **Success predicate:** All required numbers parsed (ints), conf ≥ 0.75.

### 8.2 `tasks/pickups.py`

**Purpose:** Clear visible main-menu pickups (mailbox, daily missions, etc.).

* **Loop:** While any badge is detected in bottom/top bars:

  * Ask LLM for next tap (e.g., `Mailbox`), resolve and tap, wait, then attempt **Claim/Collect** using selectors (`text:"Claim"`, `icon:"checkmark"`) with region hints.
  * After claim, **Back** to Home.
* **Success predicate:** No remaining red badges; Home screen confirmed by presence of expected elements.

### 8.3 `tasks/commissions.py`

**Purpose:** Open Commissions and record slots.

* **Navigation:** LLM selects `Commissions` from Home; resolve and tap.
* **Parsing:** Once inside:

  * Ask LLM to **structure** the visible list into JSON:

    ```json
    {"slots":[{"slot_id":0,"name":"", "rarity":"", "time_remaining_s": 1234, "status":"ready|in_progress|idle"}]}
    ```
  * For each time value, run OCR on the respective row region to **validate** numbers (convert `HH:MM:SS` to seconds). Prefer OCR over LLM text if mismatch.
* **Success predicate:** At least N visible rows parsed (N≥3 if available); rows persisted to `commissions`.

---

## 9. Logging & Storage

`core/loggingx.py`

* Write one **JSONL** per run at `~/.azlbot/runs/{timestamp}/actions.jsonl`.
* Each entry: `{ts, screen, action, selector_json, method, point_norm, confidence, success}`.
* Save frames to `.../frames/frame_XXXX.png` (configurable).
* Optionally save **overlay** images with drawn boxes and tap points.

`core/datastore.py` (SQLAlchemy)

* `insert_run(task, device_serial) -> run_id`
* `append_action(run_id, ...)`
* `record_currencies(...)`
* `record_commissions(...)`

---

## 10. UI Specification (PySide6)

`ui/app.py`

* **Main Window Layout**

  * **Left panel:** Live frame view (QLabel/QPixmap).
  * **Right panel:**

    * Controls: **Start/Stop**, Task dropdown (`Currencies`, `Pickups`, `Commissions`), “Run Once”.
    * Status: current screen, last action, confidence.
    * Data: latest currencies table; latest commissions snapshot.
    * (Collapsible) **LLM JSON** view (readonly).
* **Overlay rendering:** `ui/overlays.py` draws: last resolved point (circle), candidate boxes, region ROIs.

`ui/state.py`

* Thread-safe state: current run id, last frame ts, last plan JSON.

**Threading model**

* Worker thread for the planner loop (no blocking UI).
* Qt timer at \~2 Hz refreshes preview from the latest frame.

---

## 11. Resolution-Agnostic & Matching Details

**Normalization**

* Convert between pixel → normalized using `active_rect`.
* When tapping: `(x_norm*active_w + active_x, y_norm*active_h + active_y)`.

**OCR strategy**

* Restrict to ROIs; denoise: grayscale → adaptive threshold → slight dilation.
* Numeric mode uses whitelist `0-9:`.

**Template matching**

* Precompute edge maps for templates (Canny).
* Runtime: compute edges for ROI only; run multi-scale NCC; accept ≥ `ncc_edge`.

**Feature matching (ORB)**

* `nfeatures=1500`, ratio test 0.75, RANSAC `reprojThresh=3.0`.
* Accept if inliers ≥ `orb_inliers`.

**Confidence gating (final)**

* Accept candidate if:

  * (OCR ≥ 0.65 **and** NCC ≥ 0.60) **or**
  * (ORB inliers ≥ 12) **or**
  * (single method ≥ 0.85).
* After action: **two-frame confirmation** required.

**Badge detection (HSV)**

* Hue ∈ \[0–10]∨\[350–360], S≥0.6, V≥0.4; area ≥ (20 px @1080p scaled by area).
* Check ROIs near icon anchors to avoid false positives.

---

## 12. Error Handling & Recovery

* **Lost state:** Press **Back** up to 3 times; if still unknown, tap top-left 2% (heuristic back/home).
* **Selector not found:** Next best candidate; if none, re-prompt LLM with cropped tiles (center + region) and the validation error message.
* **Low confidence (<0.55):** Disallow action unless two methods agree.
* **ADB errors:** Attempt `adb reconnect` once; surface blocking error to UI with retry button.

---

## 13. Command-line Utilities

`scripts/run_gui.sh`

```bash
#!/usr/bin/env bash
export AZL_CONFIG=${AZL_CONFIG:-"./config/app.yaml"}
python -m azl_bot.ui.app
```

`scripts/run_task.sh`

```bash
#!/usr/bin/env bash
task=${1:-"currencies"}  # or "pickups", "commissions"
python - <<'PY'
from azl_bot.core.bootstrap import run_task_cli
run_task_cli("'''$task'''")
PY
```

`scripts/waydroid_setup.sh`

* Sanity checks: `adb devices`, connect to specified serial, verify `wm size`, capture test frame.

---

## 14. Testing Plan

### 14.1 Unit tests (`tests/unit`)

* **Coordinate math:** pixel↔normalized with letterbox.
* **OCR parsing:** `HH:MM:SS` → seconds, numeric cleaning.
* **String matching:** synonyms & fuzzy thresholds.
* **Resolver ranking:** deterministic with synthetic inputs.

### 14.2 Integration tests (`tests/integration`)

* **Resolver on fixtures:** run OCR+NCC+ORB on provided screenshots; assert candidate within 2% of expected.
* **LLM plan validation:** mock LLM responses; Pydantic validation errors cause auto-reprompt (simulate).
* **Planner step:** simulate tap → confirm with “screen changed” mock (compare two frames).

### 14.3 End-to-end (offline)

* Sequence of recorded frames for each task; mock actuator to no-op; verify success predicates and DB writes.

**Fixtures needed:**

* At least 3 resolutions per screen: 1280×720, 1920×1080, 2560×1440.
* Home screen with badges/no badges.
* Commissions screen with (ready/in\_progress/idle) examples.

---

## 15. Acceptance Criteria (must pass before iteration)

1. **Currencies**

   * From Home, one run writes a row to `currencies` with **Oil, Coins, Gems** present, all ints, conf≥0.75.
   * UI shows these values within 2 seconds of run completion.

2. **Pickups**

   * If a mail/mission badge is present in the fixture or live emulator, task clears it and returns to Home.
   * Logs show action sequence with `success=1` for claim taps; after run, **no badge** detected.

3. **Commissions**

   * Task opens Commissions, captures visible slots (≥3 when present), stores each with `status`, `name`, and `time_remaining_s`.
   * Numeric times are verified by OCR; mismatches resolved in favor of OCR.
   * UI presents a table of the parsed slots.

4. **Resolution-agnostic**

   * The same binaries succeed on the three test resolutions above without code changes.

5. **UI**

   * Window remains responsive during runs; preview updates \~2 FPS; overlays show last tap.
   * “Start/Stop” and “Run Once” operate without race conditions.

6. **Logging**

   * A run directory with `actions.jsonl` + saved frames exists; DB rows present for each task.

---

## 16. Implementation Details by File

### `core/configs.py`

* Load/validate `app.yaml`; expose typed accessors.
* Hot-reload not required.

### `core/loggingx.py`

* `init_logger(level)`, `open_run_dir()`, `log_action(...)`, `save_frame(frame, with_overlay=False)`.

### `core/bootstrap.py`

* Wire all components from config; dependency injection for tests.
* Provide `run_task_cli(task_name:str)` used by `run_task.sh`.

### `ui/app.py`

* Build widgets; spawn **Planner** worker thread.
* Signal/slot to push latest frame and plan JSON to UI.
* Menu: “Export currencies CSV”.

### `ui/overlays.py`

* Draw regions, candidate points, tap circles, and bounding boxes onto QPixmap.

### `core/llm_client.py`

* HTTP client; **strict JSON** extraction (strip markdown fences, retry on parse error).
* Enforce Pydantic validation; if invalid, re-prompt with error message and the previous output.

### `core/resolver.py` (algorithms)

* **OCR path:** ROI = region\_hint if given else entire image; fuzzy match using `rapidfuzz.fuzz.WRatio`.
* **Template path:** Preload `config/templates/*.png` as edge maps; for each candidate ROI, run `cv2.matchTemplate` on edges.
* **Feature path:** ORB detect+compute on template and ROI; BFMatcher KNN; homography → anchor point.
* **Red badge:** Dedicated helper for pickups task.

### `tasks/*`

* Each task provides `goal()` (a small dict sent to LLM) and `success()` predicate; minimal task-specific hints for the LLM (e.g., “Commissions is usually in the bottom navigation area”).

---

## 17. Prompts (exact strings shipped in repo)

**System Prompt** (single line string):

```
You are a UI navigator for the mobile game Azur Lane running on an Android emulator. Respond only with JSON conforming to the schema provided. Prefer text/icon selectors with region_hint over coordinates. If uncertain, return a single back step. Be concise.
```

**User Prompt Template** (Jinja-style, stored as a `.txt` file):

```
GOAL:
{{ goal_json }}

DEVICE:
width={{ device_w }}, height={{ device_h }}, regions={{ regions_json }}

LAST_SCREEN:
{{ last_screen or "unknown" }}

INSTRUCTIONS:
Return a Plan with minimal steps to achieve the goal. Use `kind:"text"` when a button has a label. Always include a region_hint if possible. If the goal is already achieved, set done=true.

CURRENT_FRAME_BASE64:
{{ frame_base64 }}
```

---

## 18. Security/Privacy

* No personal data.
* LLM receives screenshots only; redact nothing for v0.
* API keys read from env; never stored in DB or logs.

---

## 19. Build & Run

```bash
# Debian prerequisites
sudo apt-get install -y android-tools-adb tesseract-ocr
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Configure
cp config/app.yaml.example config/app.yaml
$EDITOR config/app.yaml   # set adb_serial, package_name, LLM endpoint/key

# Launch GUI
./scripts/run_gui.sh

# Run a single task from CLI
./scripts/run_task.sh currencies
```

---

## 20. Engineering Checklist (v0 sign‑off)

* [ ] `adb devices` shows configured serial; `device.info()` returns width/height/density.
* [ ] Capture works; letterbox detected; normalized coordinates round-trip.
* [ ] OCR returns expected tokens on top-bar fixtures.
* [ ] Template + ORB matching find `Commissions` icon on fixtures at all three resolutions.
* [ ] LLM returns valid `Plan` JSON; validation-rescue path works.
* [ ] Planner executes `tap → wait → check` successfully for each task.
* [ ] DB rows written for currencies and commissions; JSONL log produced.
* [ ] GUI renders frames & overlays; controls responsive.
* [ ] Acceptance Criteria (Section 15) all green.

---

This is the complete v0 architecture. If you want, I can follow up with a **starter code scaffold** (files with method stubs and TODOs) aligned 1:1 with this spec so agents can start implementing immediately.
