# Azur Lane Bot - Implementation Guide

**Version:** 1.0
**Last Updated:** 2025-08-31

## Repository Structure

```
azl_bot/
  pyproject.toml          # UV package management
  uv.lock                 # Locked dependencies
  README.md
  azl_bot/
    core/                 # Core functionality
      device.py          # ADB device interface
      actuator.py        # Input generation
      capture.py         # Screen capture & preprocessing
      resolver.py        # Selector to coordinate resolution
      ocr.py            # OCR text extraction
      llm_client.py     # LLM API interface
      planner.py        # Task orchestration
      screens.py        # Screen detection & regions
      datastore.py      # SQLAlchemy persistence
      loggingx.py       # Structured logging
      configs.py        # Configuration management
      bootstrap.py      # Dependency injection
    tasks/              # Game-specific tasks
      pickups.py        # Badge collection
      commissions.py    # Commission reading
      currencies.py     # Resource tracking
    ui/                 # GUI components
      app.py           # Main PySide6 window
      overlays.py      # Visual annotations
      state.py         # Thread-safe state
    config/            # Configuration files
      app.yaml         # Main configuration
      selectors/       # Selector definitions
        synonyms.yaml  # Text synonyms
      templates/       # Template images
        *.png         # Icon templates
    data/             # Data schemas
      migrations.sql  # Database schema
    tests/           # Test suites
    scripts/         # Utility scripts
```

## Core Implementation Details

### Device Interface (`core/device.py`)

```python
from typing import TypedDict
import subprocess

class DeviceInfo(TypedDict):
    width: int
    height: int
    density: int

class Device:
    def __init__(self, serial: str):
        self.serial = serial
        self._connect()
    
    def _connect(self):
        """Establish ADB connection"""
        subprocess.run(['adb', 'connect', self.serial], check=True)
    
    def info(self) -> DeviceInfo:
        """Get device display information"""
        # Execute: adb shell wm size
        # Execute: adb shell wm density
        # Parse and return DeviceInfo
    
    def screencap_png(self) -> bytes:
        """Capture screen as PNG bytes"""
        cmd = ['adb', '-s', self.serial, 'exec-out', 'screencap', '-p']
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout
    
    def key_back(self):
        """Send BACK key event"""
        subprocess.run(['adb', '-s', self.serial, 'shell', 'input', 'keyevent', '4'])
    
    def key_home(self):
        """Send HOME key event"""
        subprocess.run(['adb', '-s', self.serial, 'shell', 'input', 'keyevent', '3'])
```

### Actuator Implementation (`core/actuator.py`)

```python
from typing import Literal
import time

class Actuator:
    def __init__(self, device: Device, backend: Literal["adb", "minitouch"] = "adb"):
        self.device = device
        self.backend = backend
        self.last_tap = (0, 0, 0)  # (x, y, timestamp)
    
    def tap_norm(self, x: float, y: float):
        """Execute tap at normalized coordinates"""
        # Debounce check
        now = time.time()
        if self._is_duplicate_tap(x, y, now):
            return
        
        # Convert to pixels
        info = self.device.info()
        px = int(x * info['width'])
        py = int(y * info['height'])
        
        # Execute tap
        cmd = ['adb', '-s', self.device.serial, 'shell', 'input', 'tap', str(px), str(py)]
        subprocess.run(cmd, check=True)
        
        self.last_tap = (x, y, now)
    
    def _is_duplicate_tap(self, x: float, y: float, now: float) -> bool:
        """Check for duplicate tap within 500ms and 1% distance"""
        dx = abs(x - self.last_tap[0])
        dy = abs(y - self.last_tap[1])
        dt = now - self.last_tap[2]
        return dt < 0.5 and dx < 0.01 and dy < 0.01
    
    def swipe_norm(self, x1: float, y1: float, x2: float, y2: float, ms: int = 200):
        """Execute swipe between normalized coordinates"""
        info = self.device.info()
        px1, py1 = int(x1 * info['width']), int(y1 * info['height'])
        px2, py2 = int(x2 * info['width']), int(y2 * info['height'])
        
        cmd = ['adb', '-s', self.device.serial, 'shell', 'input', 'swipe',
               str(px1), str(py1), str(px2), str(py2), str(ms)]
        subprocess.run(cmd, check=True)
```

### Frame Capture (`core/capture.py`)

```python
import numpy as np
from dataclasses import dataclass
from PIL import Image
import io
import time

@dataclass
class Frame:
    png_bytes: bytes
    image_bgr: np.ndarray  # Cropped active area
    full_w: int
    full_h: int
    active_rect: tuple[int, int, int, int]  # (x, y, w, h)
    ts: float

class Capture:
    def __init__(self, device: Device):
        self.device = device
    
    def grab(self) -> Frame:
        """Capture and process frame"""
        png_bytes = self.device.screencap_png()
        
        # Load image
        img = Image.open(io.BytesIO(png_bytes))
        img_bgr = np.array(img)[:, :, ::-1]  # RGB to BGR
        
        # Detect letterbox
        active_rect = self.detect_letterbox(img_bgr)
        
        # Crop to active area
        x, y, w, h = active_rect
        active_bgr = img_bgr[y:y+h, x:x+w]
        
        return Frame(
            png_bytes=png_bytes,
            image_bgr=active_bgr,
            full_w=img.width,
            full_h=img.height,
            active_rect=active_rect,
            ts=time.time()
        )
    
    def detect_letterbox(self, img_bgr: np.ndarray) -> tuple[int, int, int, int]:
        """Detect and remove black bars"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Scan for uniform borders
        threshold = 20  # Near-black threshold
        
        # Top border
        top = 0
        for y in range(h // 4):
            if np.mean(gray[y, :]) > threshold:
                top = y
                break
        
        # Bottom border
        bottom = h
        for y in range(h - 1, 3 * h // 4, -1):
            if np.mean(gray[y, :]) > threshold:
                bottom = y + 1
                break
        
        # Left border
        left = 0
        for x in range(w // 4):
            if np.mean(gray[top:bottom, x]) > threshold:
                left = x
                break
        
        # Right border
        right = w
        for x in range(w - 1, 3 * w // 4, -1):
            if np.mean(gray[top:bottom, x]) > threshold:
                right = x + 1
                break
        
        return (left, top, right - left, bottom - top)
```

### OCR Implementation (`core/ocr.py`)

```python
from paddleocr import PaddleOCR
import numpy as np

class OCRClient:
    def __init__(self, lang='en'):
        self.reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=True)
    
    def text_in_roi(self, img_bgr: np.ndarray, 
                    roi_norm: tuple[float, float, float, float],
                    numeric_only: bool = False) -> list[dict]:
        """Extract text from region of interest"""
        h, w = img_bgr.shape[:2]
        x, y, rw, rh = roi_norm
        
        # Convert normalized to pixel coordinates
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + rw) * w), int((y + rh) * h)
        
        # Crop ROI
        roi_img = img_bgr[y1:y2, x1:x2]
        
        # Preprocess for better OCR
        if numeric_only:
            roi_img = self._preprocess_numeric(roi_img)
        
        # Run OCR
        results = self.reader.ocr(roi_img, cls=True)
        
        # Parse results
        output = []
        for line in results[0] if results[0] else []:
            bbox, (text, conf) = line
            
            # Filter numeric if requested
            if numeric_only:
                text = ''.join(c for c in text if c.isdigit() or c == ':')
            
            # Normalize bbox coordinates back to full image
            norm_bbox = self._normalize_bbox(bbox, x1, y1, w, h)
            
            output.append({
                'text': text,
                'conf': conf,
                'box_norm': norm_bbox
            })
        
        return output
    
    def _preprocess_numeric(self, img: np.ndarray) -> np.ndarray:
        """Optimize image for numeric OCR"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.dilate(binary, kernel, iterations=1)
        return processed
    
    def _normalize_bbox(self, bbox, offset_x, offset_y, img_w, img_h):
        """Convert bbox to normalized coordinates"""
        x_min = min(bbox[0][0], bbox[3][0]) + offset_x
        y_min = min(bbox[0][1], bbox[1][1]) + offset_y
        x_max = max(bbox[1][0], bbox[2][0]) + offset_x
        y_max = max(bbox[2][1], bbox[3][1]) + offset_y
        
        return [
            x_min / img_w,
            y_min / img_h,
            (x_max - x_min) / img_w,
            (y_max - y_min) / img_h
        ]
```

### LLM Client (`core/llm_client.py`)

```python
import requests
import json
import base64
from pydantic import BaseModel, ValidationError
from typing import Literal, Optional

class Target(BaseModel):
    kind: Literal["text", "icon", "bbox", "point", "region"]
    value: Optional[str] = None
    bbox: Optional[list[float]] = None
    point: Optional[list[float]] = None
    region_hint: Optional[str] = None
    confidence: float = 0.5

class Step(BaseModel):
    action: Literal["tap", "swipe", "wait", "back", "assert"]
    target: Optional[Target] = None
    rationale: Optional[str] = None

class Plan(BaseModel):
    screen: str
    steps: list[Step]
    done: bool = False

class LLMClient:
    SYSTEM_PROMPT = """You are a UI navigator for the mobile game Azur Lane running on an Android emulator. 
    Respond only with JSON conforming to the schema provided. 
    Prefer text/icon selectors with region_hint over coordinates. 
    If uncertain, return a single back step. Be concise."""
    
    def __init__(self, cfg):
        self.endpoint = cfg['llm']['endpoint']
        self.api_key = os.environ.get(cfg['llm']['api_key_env'])
        self.model = cfg['llm']['model']
        self.max_tokens = cfg['llm']['max_tokens']
        self.temperature = cfg['llm']['temperature']
    
    def propose_plan(self, frame: Frame, goal: dict, context: dict) -> Plan:
        """Request action plan from LLM"""
        # Encode frame as base64
        img_base64 = base64.b64encode(frame.png_bytes).decode('utf-8')
        
        # Build prompt
        user_prompt = self._build_prompt(img_base64, goal, context, frame)
        
        # Call LLM API
        response = self._call_api(user_prompt)
        
        # Parse and validate response
        plan = self._parse_response(response)
        
        return plan
    
    def _build_prompt(self, img_base64: str, goal: dict, context: dict, frame: Frame) -> str:
        """Construct user prompt from template"""
        return f"""
GOAL:
{json.dumps(goal)}

DEVICE:
width={frame.full_w}, height={frame.full_h}

LAST_SCREEN:
{context.get('last_screen', 'unknown')}

INSTRUCTIONS:
Return a Plan with minimal steps to achieve the goal. Use kind:"text" when a button has a label. 
Always include a region_hint if possible. If the goal is already achieved, set done=true.

CURRENT_FRAME_BASE64:
{img_base64}
"""
    
    def _call_api(self, prompt: str) -> str:
        """Make API request to LLM"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self.SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'response_format': {'type': 'json_object'}
        }
        
        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        
        return response.json()['choices'][0]['message']['content']
    
    def _parse_response(self, response_text: str) -> Plan:
        """Parse and validate LLM response"""
        # Strip markdown fences if present
        response_text = response_text.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1].rsplit('\n```', 1)[0]
        
        try:
            data = json.loads(response_text)
            return Plan(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            # Re-prompt on failure
            raise ValueError(f"Invalid LLM response: {e}")
```

### Selector Resolver (`core/resolver.py`)

```python
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from rapidfuzz import fuzz
import yaml

@dataclass
class Candidate:
    point: tuple[float, float]  # Normalized coordinates
    confidence: float
    method: str  # "ocr", "template", "orb", "llm"

class Resolver:
    def __init__(self, cfg, ocr: OCRClient, templates_dir: Path):
        self.cfg = cfg
        self.ocr = ocr
        self.templates = self._load_templates(templates_dir)
        self.synonyms = self._load_synonyms()
        self.thresholds = cfg['resolver']['thresholds']
    
    def resolve(self, selector: Target, frame: Frame) -> Candidate:
        """Convert selector to coordinate"""
        candidates = []
        
        # Try OCR matching
        if selector.kind == "text" and selector.value:
            candidate = self._resolve_by_ocr(selector, frame)
            if candidate:
                candidates.append(candidate)
        
        # Try template matching
        if selector.kind == "icon" and selector.value:
            candidate = self._resolve_by_template(selector, frame)
            if candidate:
                candidates.append(candidate)
        
        # Try feature matching
        if selector.kind in ["icon", "region"] and selector.value:
            candidate = self._resolve_by_features(selector, frame)
            if candidate:
                candidates.append(candidate)
        
        # LLM fallback
        if selector.kind in ["point", "bbox"]:
            candidate = self._resolve_by_llm(selector, frame)
            if candidate:
                candidates.append(candidate)
        
        # Select best candidate
        if not candidates:
            raise ValueError(f"Could not resolve selector: {selector}")
        
        # Apply ensemble bonus
        best = max(candidates, key=lambda c: c.confidence)
        if len(candidates) > 1:
            # Check for agreement
            for other in candidates:
                if other != best and self._points_agree(best.point, other.point):
                    best.confidence = min(1.0, best.confidence + 0.1)
        
        return best
    
    def _resolve_by_ocr(self, selector: Target, frame: Frame) -> Optional[Candidate]:
        """Resolve using OCR text matching"""
        # Determine ROI
        roi = self._get_roi(selector.region_hint)
        
        # Extract text
        texts = self.ocr.text_in_roi(frame.image_bgr, roi)
        
        # Find best match
        best_match = None
        best_score = 0
        
        for text_info in texts:
            # Check synonyms
            target_texts = self.synonyms.get(selector.value, [selector.value])
            
            for target in target_texts:
                score = fuzz.WRatio(text_info['text'], target) / 100.0
                
                if score > best_score and score >= self.thresholds['ocr_text']:
                    best_score = score
                    # Calculate center point of bbox
                    x, y, w, h = text_info['box_norm']
                    best_match = Candidate(
                        point=(x + w/2, y + h/2),
                        confidence=score * text_info['conf'],
                        method="ocr"
                    )
        
        return best_match
    
    def _resolve_by_template(self, selector: Target, frame: Frame) -> Optional[Candidate]:
        """Resolve using template matching"""
        template_name = f"{selector.value}.png"
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        
        # Multi-scale template matching
        scales = np.arange(0.7, 1.3, 0.05)
        best_match = None
        best_val = 0
        
        for scale in scales:
            # Resize template
            scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
            
            # Edge detection
            frame_edges = cv2.Canny(frame.image_bgr, 50, 150)
            template_edges = cv2.Canny(scaled_template, 50, 150)
            
            # Match
            result = cv2.matchTemplate(frame_edges, template_edges, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val and max_val >= self.thresholds['ncc_edge']:
                best_val = max_val
                h, w = scaled_template.shape[:2]
                # Convert to normalized coordinates
                center_x = (max_loc[0] + w/2) / frame.image_bgr.shape[1]
                center_y = (max_loc[1] + h/2) / frame.image_bgr.shape[0]
                best_match = Candidate(
                    point=(center_x, center_y),
                    confidence=max_val,
                    method="template"
                )
        
        return best_match
    
    def _resolve_by_features(self, selector: Target, frame: Frame) -> Optional[Candidate]:
        """Resolve using ORB feature matching"""
        template_name = f"{selector.value}.png"
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        
        # Initialize ORB
        orb = cv2.ORB_create(nfeatures=1500)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(frame.image_bgr, None)
        
        if des1 is None or des2 is None:
            return None
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.thresholds['orb_inliers']:
            return None
        
        # Find homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if M is None:
            return None
        
        # Transform template center
        h, w = template.shape[:2]
        center = np.float32([[w/2, h/2]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(center, M)
        
        # Convert to normalized
        point_x = transformed[0][0][0] / frame.image_bgr.shape[1]
        point_y = transformed[0][0][1] / frame.image_bgr.shape[0]
        
        # Count inliers
        inliers = np.sum(mask)
        
        return Candidate(
            point=(point_x, point_y),
            confidence=min(1.0, inliers / 50.0),
            method="orb"
        )
    
    def _resolve_by_llm(self, selector: Target, frame: Frame) -> Optional[Candidate]:
        """Use LLM-provided coordinates as fallback"""
        if selector.point:
            return Candidate(
                point=tuple(selector.point),
                confidence=selector.confidence,
                method="llm"
            )
        elif selector.bbox:
            x, y, w, h = selector.bbox
            return Candidate(
                point=(x + w/2, y + h/2),
                confidence=selector.confidence,
                method="llm"
            )
        return None
    
    def _get_roi(self, region_hint: str) -> tuple[float, float, float, float]:
        """Get region of interest from hint"""
        regions = self.cfg['resolver']['regions']
        return regions.get(region_hint, (0, 0, 1, 1))
    
    def _points_agree(self, p1: tuple, p2: tuple, threshold: float = 0.02) -> bool:
        """Check if two points are within threshold distance"""
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        return dx < threshold and dy < threshold
    
    def _load_templates(self, templates_dir: Path) -> dict:
        """Load template images"""
        templates = {}
        for template_path in templates_dir.glob("*.png"):
            img = cv2.imread(str(template_path))
            templates[template_path.name] = img
        return templates
    
    def _load_synonyms(self) -> dict:
        """Load text synonyms"""
        synonyms_path = Path(self.cfg['config_dir']) / 'selectors' / 'synonyms.yaml'
        with open(synonyms_path) as f:
            return yaml.safe_load(f)
    
    def to_pixels(self, point_norm: tuple[float, float], frame: Frame) -> tuple[int, int]:
        """Convert normalized to pixel coordinates"""
        x, y, w, h = frame.active_rect
        px = int(point_norm[0] * w + x)
        py = int(point_norm[1] * h + y)
        return (px, py)
```

### Planner Implementation (`core/planner.py`)

```python
import time
from typing import Protocol

class Task(Protocol):
    name: str
    def goal(self) -> dict: ...
    def success(self, frame: Frame, context: dict) -> bool: ...
    def on_success(self, planner: 'Planner', frame: Frame) -> None: ...

class Planner:
    def __init__(self, device: Device, capture: Capture, resolver: Resolver, 
                 ocr: OCRClient, llm: LLMClient, store, logger):
        self.device = device
        self.capture = capture
        self.resolver = resolver
        self.ocr = ocr
        self.llm = llm
        self.store = store
        self.logger = logger
        self.actuator = Actuator(device)
        self.context = {'last_screen': 'unknown'}
    
    def run_task(self, task: Task):
        """Execute a complete task"""
        self.logger.info(f"Starting task: {task.name}")
        run_id = self.store.insert_run(task.name, self.device.serial)
        
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            # Capture current state
            frame = self.capture.grab()
            
            # Check if task is complete
            if task.success(frame, self.context):
                self.logger.info(f"Task {task.name} completed successfully")
                task.on_success(self, frame)
                break
            
            # Get plan from LLM
            goal = task.goal()
            plan = self.llm.propose_plan(frame, goal, self.context)
            
            # Update context
            self.context['last_screen'] = plan.screen
            
            # Check if LLM thinks we're done
            if plan.done:
                self.logger.info("LLM reports task complete")
                if not task.success(frame, self.context):
                    self.logger.warning("LLM done but task success check failed")
                break
            
            # Execute steps
            for step in plan.steps:
                success = self.run_step(step, frame)
                self.store.append_action(
                    run_id=run_id,
                    screen=plan.screen,
                    action=step.action,
                    selector_json=step.target.dict() if step.target else None,
                    success=success
                )
                
                if not success:
                    self.logger.warning(f"Step failed: {step}")
                    self.recover()
                    break
                
                # Wait between actions
                time.sleep(1.0)
                
                # Capture new frame for next step
                frame = self.capture.grab()
            
            iteration += 1
        
        if iteration >= max_iterations:
            self.logger.error(f"Task {task.name} exceeded max iterations")
    
    def run_step(self, step: Step, frame: Frame) -> bool:
        """Execute a single step"""
        try:
            if step.action == "tap":
                if not step.target:
                    return False
                
                # Resolve to coordinates
                candidate = self.resolver.resolve(step.target, frame)
                
                # Check confidence
                if candidate.confidence < 0.55:
                    self.logger.warning(f"Low confidence: {candidate.confidence}")
                    return False
                
                # Execute tap
                self.actuator.tap_norm(*candidate.point)
                
                # Verify with two-frame confirmation
                time.sleep(0.5)
                frame2 = self.capture.grab()
                return self._frames_different(frame, frame2)
            
            elif step.action == "swipe":
                if not step.target or not step.target.bbox:
                    return False
                
                x1, y1, x2, y2 = step.target.bbox
                self.actuator.swipe_norm(x1, y1, x2, y2)
                return True
            
            elif step.action == "wait":
                time.sleep(2.0)
                return True
            
            elif step.action == "back":
                self.device.key_back()
                return True
            
            elif step.action == "assert":
                # Assertion step - check condition
                return True
            
            else:
                self.logger.warning(f"Unknown action: {step.action}")
                return False
                
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            return False
    
    def recover(self):
        """Attempt to recover from error state"""
        self.logger.info("Attempting recovery")
        
        # Try back button up to 3 times
        for _ in range(3):
            self.device.key_back()
            time.sleep(1.0)
            
            frame = self.capture.grab()
            if self._is_home_screen(frame):
                self.logger.info("Recovered to home screen")
                self.context['last_screen'] = 'home'
                return
        
        # Last resort - try home key
        self.device.key_home()
        time.sleep(2.0)
    
    def _frames_different(self, frame1: Frame, frame2: Frame) -> bool:
        """Check if two frames are significantly different"""
        diff = cv2.absdiff(frame1.image_bgr, frame2.image_bgr)
        mean_diff = np.mean(diff)
        return mean_diff > 5.0  # Threshold for change detection
    
    def _is_home_screen(self, frame: Frame) -> bool:
        """Check if we're on the home screen"""
        # Look for expected home elements
        expected = ["Commissions", "Missions", "Mailbox"]
        found = 0
        
        for text in expected:
            try:
                selector = Target(kind="text", value=text, confidence=0.5)
                candidate = self.resolver.resolve(selector, frame)
                if candidate.confidence > 0.6:
                    found += 1
            except:
                pass
        
        return found >= 2
```

### Task Implementations

#### Currencies Task (`tasks/currencies.py`)

```python
class CurrenciesTask:
    name = "currencies"
    
    def goal(self) -> dict:
        return {"action": "read_currencies", "required": ["oil", "coins", "gems"]}
    
    def success(self, frame: Frame, context: dict) -> bool:
        """Check if currencies have been successfully read"""
        return context.get('currencies_read', False)
    
    def on_success(self, planner: Planner, frame: Frame):
        """Extract and store currency values"""
        # Define ROIs for each currency
        regions = {
            'oil': (0.1, 0.02, 0.15, 0.08),
            'coins': (0.3, 0.02, 0.15, 0.08),
            'gems': (0.5, 0.02, 0.15, 0.08),
        }
        
        values = {}
        for currency, roi in regions.items():
            texts = planner.ocr.text_in_roi(frame.image_bgr, roi, numeric_only=True)
            
            if texts and texts[0]['conf'] >= 0.75:
                try:
                    value = int(texts[0]['text'].replace(',', ''))
                    values[currency] = value
                except ValueError:
                    planner.logger.warning(f"Failed to parse {currency}: {texts[0]['text']}")
        
        # Store in database
        if len(values) >= 3:
            planner.store.record_currencies(**values)
            planner.context['currencies_read'] = True
            planner.logger.info(f"Currencies recorded: {values}")
```

#### Pickups Task (`tasks/pickups.py`)

```python
class PickupsTask:
    name = "pickups"
    
    def goal(self) -> dict:
        return {"action": "clear_pickups", "targets": ["mail", "missions", "rewards"]}
    
    def success(self, frame: Frame, context: dict) -> bool:
        """Check if all badges have been cleared"""
        # Detect red badges
        badges = self._detect_badges(frame)
        return len(badges) == 0
    
    def on_success(self, planner: Planner, frame: Frame):
        """Log successful pickup collection"""
        planner.logger.info("All pickups collected")
    
    def _detect_badges(self, frame: Frame) -> list[tuple[float, float]]:
        """Detect red notification badges"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2HSV)
        
        # Red color range in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        badges = []
        min_area = 20  # Minimum badge area in pixels
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get center
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"] / frame.image_bgr.shape[1]
                    cy = M["m01"] / M["m00"] / frame.image_bgr.shape[0]
                    badges.append((cx, cy))
        
        return badges
```

#### Commissions Task (`tasks/commissions.py`)

```python
class CommissionsTask:
    name = "commissions"
    
    def goal(self) -> dict:
        return {"action": "read_commissions", "navigate_to": "commissions_screen"}
    
    def success(self, frame: Frame, context: dict) -> bool:
        """Check if commissions have been read"""
        return context.get('commissions_read', False)
    
    def on_success(self, planner: Planner, frame: Frame):
        """Parse and store commission data"""
        # Ask LLM to structure visible commissions
        # This would normally be done through the LLM with a specific prompt
        
        # For each commission row, extract with OCR
        commission_rows = self._find_commission_rows(frame)
        
        commissions = []
        for i, row_roi in enumerate(commission_rows):
            # Extract text from row
            texts = planner.ocr.text_in_roi(frame.image_bgr, row_roi)
            
            # Parse time remaining
            time_text = self._extract_time(texts)
            time_seconds = self._parse_time(time_text)
            
            # Determine status
            status = self._determine_status(texts, time_seconds)
            
            commission = {
                'slot_id': i,
                'name': self._extract_name(texts),
                'time_remaining_s': time_seconds,
                'status': status
            }
            
            commissions.append(commission)
            planner.store.record_commission(**commission)
        
        if len(commissions) >= 3:
            planner.context['commissions_read'] = True
            planner.logger.info(f"Recorded {len(commissions)} commissions")
    
    def _find_commission_rows(self, frame: Frame) -> list[tuple[float, float, float, float]]:
        """Identify commission row regions"""
        # Fixed regions for commission slots
        rows = []
        y_start = 0.25
        row_height = 0.12
        
        for i in range(4):  # Up to 4 commission slots
            roi = (0.1, y_start + i * row_height, 0.8, row_height)
            rows.append(roi)
        
        return rows
    
    def _extract_time(self, texts: list[dict]) -> str:
        """Extract time string from OCR results"""
        for text_info in texts:
            if ':' in text_info['text']:
                return text_info['text']
        return "00:00:00"
    
    def _parse_time(self, time_str: str) -> int:
        """Convert HH:MM:SS to seconds"""
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(int, parts)
                return m * 60 + s
        except:
            pass
        return 0
    
    def _determine_status(self, texts: list[dict], time_seconds: int) -> str:
        """Determine commission status"""
        text_combined = ' '.join(t['text'] for t in texts).lower()
        
        if 'complete' in text_combined or 'ready' in text_combined:
            return 'ready'
        elif time_seconds > 0:
            return 'in_progress'
        else:
            return 'idle'
    
    def _extract_name(self, texts: list[dict]) -> str:
        """Extract commission name"""
        # Take the longest text that isn't a time
        non_time_texts = [t['text'] for t in texts if ':' not in t['text']]
        if non_time_texts:
            return max(non_time_texts, key=len)
        return "Unknown"
```

### Database Schema (`data/migrations.sql`)

```sql
-- Run tracking
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task TEXT NOT NULL,
    device_serial TEXT NOT NULL
);

-- Action logging
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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

-- Currency tracking
CREATE TABLE IF NOT EXISTS currencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    oil INTEGER,
    coins INTEGER,
    gems INTEGER,
    cubes INTEGER
);

-- Commission tracking
CREATE TABLE IF NOT EXISTS commissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    slot_id INTEGER,
    name TEXT,
    rarity TEXT,
    time_remaining_s INTEGER,
    status TEXT CHECK(status IN ('idle', 'in_progress', 'ready'))
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task);
CREATE INDEX IF NOT EXISTS idx_actions_run ON actions(run_id);
CREATE INDEX IF NOT EXISTS idx_currencies_ts ON currencies(ts);
CREATE INDEX IF NOT EXISTS idx_commissions_ts ON commissions(ts);
```

### Configuration File (`config/app.yaml`)

```yaml
emulator:
  kind: waydroid  # or "genymotion"
  adb_serial: "127.0.0.1:5555"
  package_name: "com.YoStarEN.AzurLane"

display:
  target_fps: 2
  orientation: "landscape"
  force_resolution: null  # e.g., "1920x1080"

llm:
  provider: "gemini"
  model: "gemini-1.5-flash"
  endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
  api_key_env: "GEMINI_API_KEY"
  max_tokens: 2048
  temperature: 0.1

resolver:
  ocr: "paddle"  # or "tesseract"
  thresholds:
    ocr_text: 0.75
    ncc_edge: 0.60
    ncc_gray: 0.70
    orb_inliers: 12
    combo_accept: 0.65
  regions:
    top_bar: [0.00, 0.00, 1.00, 0.12]
    bottom_bar: [0.00, 0.85, 1.00, 0.15]
    left_panel: [0.00, 0.12, 0.20, 0.73]
    center: [0.20, 0.12, 0.60, 0.73]
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

### Synonyms Configuration (`config/selectors/synonyms.yaml`)

```yaml
# English/Chinese/Japanese text variants
Commissions: ["Commissions", "Commission", "委托", "委託"]
Mailbox: ["Mail", "Mailbox", "信箱", "メール"]
Missions: ["Missions", "Tasks", "任務", "任务"]
Gems: ["Gems", "钻石", "ダイヤ"]
Oil: ["Oil", "燃料"]
Coins: ["Coins", "金币", "資金"]
Back: ["Back", "返回", "戻る"]
Claim: ["Claim", "Collect", "领取", "受け取る"]
Confirm: ["Confirm", "OK", "确认", "確認"]
Close: ["Close", "X", "关闭", "閉じる"]
```

## Testing Implementation

### Basic Component Test (`tests/basic_test.py`)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all core modules can be imported"""
    modules = [
        'azl_bot.core.device',
        'azl_bot.core.actuator',
        'azl_bot.core.capture',
        'azl_bot.core.resolver',
        'azl_bot.core.ocr',
        'azl_bot.core.llm_client',
        'azl_bot.core.planner',
        'azl_bot.tasks.currencies',
        'azl_bot.tasks.pickups',
        'azl_bot.tasks.commissions',
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            return False
    
    return True

def test_config():
    """Test configuration loading"""
    from azl_bot.core.configs import load_config
    
    try:
        cfg = load_config()
        assert 'emulator' in cfg
        assert 'llm' in cfg
        assert 'resolver' in cfg
        print("✓ Configuration loaded")
        return True
    except Exception as e:
        print(f"✗ Configuration: {e}")
        return False

if __name__ == "__main__":
    success = test_imports() and test_config()
    sys.exit(0 if success else 1)
```

## Deployment Scripts

### GUI Launch Script (`scripts/run_gui.sh`)

```bash
#!/usr/bin/env bash
set -e

# Set configuration path
export AZL_CONFIG=${AZL_CONFIG:-"./config/app.yaml"}

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run: uv venv && uv pip install -e ."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Launch GUI
echo "Starting Azur Lane Bot GUI..."
python -m azl_bot.ui.app
```

### Task Runner Script (`scripts/run_task.sh`)

```bash
#!/usr/bin/env bash
set -e

# Get task name
TASK=${1:-"currencies"}

# Set configuration
export AZL_CONFIG=${AZL_CONFIG:-"./config/app.yaml"}

# Activate virtual environment
source .venv/bin/activate

# Run task
echo "Running task: $TASK"
python -c "
from azl_bot.core.bootstrap import run_task_cli
run_task_cli('$TASK')
"
```

### Genymotion Launch Script (`scripts/launch_genymotion.sh`)

```bash
#!/usr/bin/env bash
set -e

echo "=== Genymotion Device Launcher ==="
echo "NOTE: This script uses GUI-based Genymotion (Free Edition)"
echo "Premium gmtool commands are NOT available"

# Check if Genymotion is installed
if ! command -v genymotion &> /dev/null; then
    echo "Error: Genymotion not found. Please install Genymotion first."
    exit 1
fi

# Launch Genymotion GUI
echo "Launching Genymotion GUI..."
genymotion &

echo ""
echo "Please manually:"
echo "1. Create or select an Android device in the GUI"
echo "2. Start the device"
echo "3. Note the device IP address shown in the title bar"
echo ""
echo "Then connect with ADB:"
echo "  adb connect <device_ip>:5555"
echo ""
echo "Update config/app.yaml with the device IP:"
echo "  adb_serial: '<device_ip>:5555'"
```

## Development Workflow

### Initial Setup

```bash
# Clone repository
git clone <repository_url>
cd azl_bot

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Copy configuration
cp config/app.yaml.example config/app.yaml

# Edit configuration
nano config/app.yaml

# Set API key
export GEMINI_API_KEY="your-api-key"

# Run tests
python tests/basic_test.py

# Launch GUI
./scripts/run_gui.sh
```

### Adding New Dependencies

```bash
# Always use UV for dependency management
uv add <package-name>

# For development dependencies
uv add --dev pytest black mypy

# Update lock file
uv lock --upgrade
```

### Code Quality Checks

```bash
# Format code
black azl_bot/ --line-length 88

# Sort imports
isort azl_bot/ --profile black

# Type checking
mypy azl_bot/

# Linting
flake8 azl_bot/
```

## Troubleshooting Guide

### Common Issues

1. **ADB Connection Failed**
   - Verify device is running: `adb devices`
   - Reconnect: `adb connect <ip>:5555`
   - Check firewall settings

2. **OCR Not Working**
   - Install system dependencies: `sudo apt install tesseract-ocr`
   - Verify PaddleOCR GPU support: Check CUDA installation
   - Try fallback: Set `ocr: "tesseract"` in config

3. **LLM Timeout**
   - Check API key is set: `echo $GEMINI_API_KEY`
   - Verify network connectivity
   - Increase timeout in config

4. **Low Frame Rate**
   - Check CPU usage
   - Reduce `target_fps` in config
   - Disable `overlay_draw` for performance

5. **Template Matching Fails**
   - Ensure templates match game version
   - Check resolution settings
   - Update templates from current screenshots