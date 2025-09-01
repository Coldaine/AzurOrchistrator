# Azur Lane Bot - Architecture & Design Decisions

**Version:** 1.0
**Last Updated:** 2025-08-31

## Executive Summary

The Azur Lane Bot is an automated assistant for the mobile game Azur Lane, designed to run on Linux with Android emulation. It employs a multi-modal vision approach combining LLM reasoning, OCR, and computer vision to achieve resolution-agnostic, model-agnostic automation.

## System Requirements

### Environment Requirements
- **Target OS:** Debian/Ubuntu Linux (Fedora-based distros supported)
- **Python:** ≥ 3.10
- **Android Environment:** Waydroid or Genymotion (Free Edition)
- **Control Stack:** ADB (Android Debug Bridge)
- **Memory:** Minimum 8GB RAM recommended
- **Storage:** ~2GB for dependencies and data

### External Dependencies
- Android emulator (Waydroid or Genymotion)
- ADB tools (`android-tools-adb` package)
- Gemini API access (for LLM functionality)
- Internet connection for LLM API calls

## Primary Goals (v1.0)

1. **Automated Pickups:** Collect all easy main-menu pickups (mail, missions, etc.)
2. **Commission Management:** Navigate to Commissions and read/record current commissions
3. **Resource Tracking:** Read and record currency balances (Oil, Coins, Gems; Cubes optional)
4. **Resolution Agnostic:** Support any screen resolution without code changes
5. **User Interface:** Provide real-time monitoring and control via desktop GUI

## Core Architecture

### Design Philosophy

**Loop Pattern:** Sense → Think → Act → Check

This cyclical pattern ensures robust operation with built-in error recovery and verification at each step.

### Key Design Decisions

#### 1. Resolution-Agnostic Design
- **Decision:** Use normalized coordinates [0.0-1.0] internally
- **Rationale:** Enables the bot to work across different device resolutions without modification
- **Trade-off:** Requires coordinate transformation overhead but ensures portability

##### Implementation Details

**Coordinate Space Definition:**
- Game viewport is normalized to [0.0-1.0] after letterbox removal
- Origin (0.0, 0.0) = top-left corner of game viewport
- (1.0, 1.0) = bottom-right corner of game viewport
- (0.5, 0.5) = center of game viewport
- All internal processing uses this normalized space

**Transformation Pipeline:**
1. **Capture Phase:** Raw screenshot at device resolution (e.g., 1920x1080)
2. **Letterbox Detection:** Identify black bars/padding, extract actual game viewport
3. **Vision Processing:** OCR/template matching outputs in viewport pixel coordinates
4. **Normalization:** Resolver converts pixel coords to [0.0-1.0] space
   ```
   normalized_x = (pixel_x - viewport_x) / viewport_width
   normalized_y = (pixel_y - viewport_y) / viewport_height
   ```
5. **Planning:** LLM works exclusively in normalized space
6. **Denormalization:** Actuator converts back to device pixels for ADB
   ```
   device_x = viewport_x + (normalized_x * viewport_width)
   device_y = viewport_y + (normalized_y * viewport_height)
   ```

**LLM Prompt Requirements:**
- System prompt must explicitly define coordinate space
- Include examples: "center button at (0.5, 0.8)", "top-right corner at (0.95, 0.05)"
- Enforce JSON schema with coordinate validation (0.0 ≤ x,y ≤ 1.0)
- Provide viewport aspect ratio to LLM for better spatial reasoning

**Debugging & Validation:**
- Log both coordinate systems at each transformation point
- Visual overlays in UI show normalized coords with pixel equivalents
- Validation assertions ensure coordinates stay within bounds
- Test suite includes multiple resolution fixtures (16:9, 18:9, 4:3)

#### 2. Multi-Modal Vision Approach
- **Decision:** Combine LLM reasoning with OCR and classical CV
- **Rationale:** 
  - LLM provides high-level understanding and planning
  - OCR ensures accurate text/number extraction
  - Classical CV provides reliable template/feature matching
- **Trade-off:** Higher complexity but significantly improved reliability

#### 3. Selector-Based Element Location
- **Decision:** LLM outputs abstract selectors, not pixel coordinates
- **Rationale:** Maintains separation between planning and execution, enables resolution independence
- **Trade-off:** Requires sophisticated resolver component

#### 4. Structured JSON Communication
- **Decision:** All LLM responses use strict JSON schemas (Pydantic)
- **Rationale:** Ensures predictable, parseable responses and enables validation
- **Trade-off:** Requires prompt engineering and validation logic

#### 5. Task-Based Automation
- **Decision:** Organize automation as discrete, composable tasks
- **Rationale:** Promotes modularity, testability, and incremental development
- **Trade-off:** Some code duplication across tasks

## System Components

### Core Layer
Fundamental services that all other components depend on:
- **Device Management:** Interface with Android emulator via ADB
- **Screen Capture:** Efficient frame grabbing with letterbox detection
- **Action Execution:** Tap/swipe/key event generation
- **Configuration:** Centralized settings management

### Vision Layer
Multi-modal visual processing pipeline:
- **OCR Engine:** Text extraction (PaddleOCR primary, Tesseract fallback)
- **Template Matching:** Scale-invariant template detection
- **Feature Matching:** ORB-based robust matching
- **Image Hashing:** Perceptual hash-based change detection
- **Selector Resolution:** Convert abstract selectors to coordinates
- **LLM Vision:** Semantic understanding and final arbitration

### Intelligence Layer
Decision-making and planning:
- **LLM Client:** Interface with Gemini for reasoning
- **Planner:** Orchestrate task execution with retry logic
- **Screen Detection:** Identify current game state

### Task Layer
Game-specific automation logic:
- **Pickups:** Clear notifications and collect rewards
- **Commissions:** Read and record commission status
- **Currencies:** Extract resource values from UI

### UI Layer
User interaction and monitoring:
- **Desktop GUI:** PySide6-based interface
- **Live Preview:** Real-time frame display with overlays
- **State Management:** Thread-safe state synchronization

### Data Layer
Persistence and logging:
- **SQLite Database:** Structured data storage
- **Action Logging:** JSONL format for debugging
- **Frame Storage:** Optional screenshot retention

## Data Flow Architecture

```
User Input → UI → Planner → LLM
                     ↓
                  Resolver ← Frame Capture
                     ↓
                  Actuator → Device
                     ↓
                  Verification → Data Store
```

## Confidence & Verification Strategy

### Multi-Method Validation
Elements are located using multiple methods in parallel:
1. OCR text matching with synonym support
2. Template matching with edge detection
3. ORB feature matching for complex patterns
4. **LLM Vision as Ultimate Arbiter** (not fallback)

### Confidence Scoring & Arbitration
- Individual method thresholds (OCR: 0.75, NCC: 0.60-0.70, ORB: 12 inliers)
- Ensemble bonus when methods agree (+0.1 confidence)
- **Disagreement Resolution:** When methods conflict, LLM with vision capabilities makes final decision
- LLM analyzes the actual screenshot and resolves ambiguity with semantic understanding
- Minimum acceptance threshold: 0.55 (or LLM confirmation)

### Verification Protocol
- Two-frame confirmation after actions
- Success predicates for each task
- Automatic retry with next-best candidates
- Recovery procedures for lost state

## Screen Transition & Wait Strategy

### Intelligent Waiting
- **Hash-Based Stability Detection:** Wait for hash to stabilize (no change for 2 samples)
- **Loading Screen Recognition:** Identify common loading patterns, reduce FPS to 0.2
- **Animation Filtering:** Ignore small hash differences from particle effects/animations
- **Timeout Escalation:** Gradually increase wait time if no changes detected

### Transition Detection Patterns
1. **Quick Check:** Fast hash at 2 FPS initially
2. **Stability Wait:** Drop to 0.5 FPS once movement detected
3. **Completion Verification:** Two identical hashes = transition complete
4. **Action Ready:** Resume normal processing once stable

### Computational Savings
- **Skip Redundant Processing:** ~80% of frames during transitions are identical
- **Early Exit:** Hash comparison takes <50ms vs 500ms+ for full vision pipeline
- **Smart Caching:** Reuse vision results for unchanged screen regions
- **Resource Focus:** Allocate compute only to meaningful frame changes

## Error Handling Philosophy

### Graceful Degradation
1. Try primary method (highest confidence)
2. Fall back to alternative candidates
3. **LLM Vision Arbitration:** Send screenshot to LLM for authoritative decision
4. Request LLM re-planning with error context if needed
5. Navigate to known state (Home screen)
6. Surface blocking errors to user

### Recovery Mechanisms
- **Lost State:** Back button up to 3 times, then home navigation
- **Low Confidence:** Require multi-method agreement
- **ADB Errors:** Automatic reconnection attempt
- **LLM Failures:** Validation with re-prompt on schema violations

## Security & Privacy Considerations

### Data Protection
- No personal data collection
- API keys stored in environment variables only
- Local-only operation (except LLM API calls)
- Screenshots remain on local filesystem

### Operational Security
- Read-only game interaction (no account modifications)
- No network traffic interception
- Transparent operation with full logging

## Performance & Efficiency Strategy

### Dynamic Frame Rate Management
- **Maximum:** 2 FPS (never exceeded)
- **Waiting/Idle:** 0.2-0.5 FPS during loading screens or animations
- **Navigation:** 1-2 FPS during active interaction
- **Verification:** Single frame capture after actions

### Change Detection via Image Hashing

**Hash Sensitivity Challenge:**
- Perceptual hashes (pHash/dHash) are designed to match similar images
- Game screens need opposite: detect small but meaningful changes
- A dialog popup or button state change must produce different hash

**Hashing Strategy:**
- **Average Hash (aHash):** More sensitive to small changes than pHash
- **Difference Hash (dHash):** Good for detecting UI element changes
- **Block-Based Hashing:** Divide screen into regions, hash each separately
- **Critical Region Focus:** Higher sensitivity for UI areas (buttons, dialogs)
- **Hamming Distance Tuning:** 
  - Threshold of 0-2 for "same screen" (ignore particles/animations)
  - Threshold of 3+ indicates meaningful change requiring processing

**Implementation Approach:**
```
1. Full screen dHash for general change detection
2. Regional hashes for UI zones (bottom navigation, popups)
3. Combined hash comparison with weighted regions
4. Lower threshold for interactive elements
```

**Benefits:**
  - Skip processing only truly identical frames
  - Detect subtle UI state changes (button enabled/disabled)
  - Identify loading completion and popup appearances
  - Balance between efficiency and sensitivity

### Processing Efficiency
- **Frame Deduplication:** Hash comparison before vision pipeline
- **Selective Processing:** Only run expensive operations on changed frames
- **Cached Results:** Reuse OCR/template results for unchanged regions
- **Adaptive Waiting:** Exponential backoff when no changes detected

### Performance Targets
- **Action Latency:** <2 seconds per tap/swipe
- **LLM Response:** <5 seconds per planning cycle
- **Change Detection:** <50ms via hashing
- **Memory Usage:** <500MB baseline, <1GB peak

## Scalability Considerations

### Extensibility Points
- Plugin architecture for new emulators
- Modular task system for new automations
- Configurable vision methods
- Alternative LLM providers

### Future Enhancements (Post-v1.0)
- Battle automation framework
- Event-specific task modules
- Multi-account management
- Cloud-based configuration sync
- Advanced fleet management

## Testing Strategy

### Test Levels
1. **Unit Tests:** Component isolation testing
2. **Integration Tests:** Multi-component interaction
3. **End-to-End Tests:** Full task execution with fixtures
4. **Acceptance Tests:** User-facing functionality

### Test Coverage Goals
- Core components: >90% coverage
- Vision pipeline: Fixture-based validation
- Task success: All primary paths tested
- Error recovery: Failure scenario coverage

## Deployment Architecture

### Development Environment
- UV for dependency management
- Git for version control
- Structured logging for debugging
- Hot-reload configuration (where applicable)

### Production Deployment
- Single-user desktop application
- Self-contained Python environment
- User-configurable via YAML
- Automated setup scripts

## Monitoring & Observability

### Logging Architecture
- Structured JSON logging (loguru)
- Action replay via JSONL
- Frame capture for debugging
- Performance metrics tracking

### User Feedback
- Real-time status in GUI
- Visual overlays for actions
- Confidence indicators
- Error notifications

## Acceptance Criteria

### Functional Requirements
1. Successfully read and store currency values with >75% OCR confidence
2. Clear all detected pickup badges and return to home
3. Parse and store commission data with verified timestamps
4. Operate on 3+ different resolutions without code changes
5. Maintain responsive GUI during automation

### Non-Functional Requirements
1. <2 second action completion time
2. <500MB memory usage at idle
3. Graceful handling of all error scenarios
4. Complete action audit trail
5. Zero manual pixel coordinate configuration