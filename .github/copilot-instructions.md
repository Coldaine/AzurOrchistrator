# Instructions for AI coding agents

Python 3.10+ Azur Lane automation bot targeting Android emulators. Core loop: Sense → Think → Resolve → Act → Check. Everything uses normalized active-area coordinates (after letterbox crop).

## Big picture (files to know)
- Sense: `core.device.Device` (ADB) → `core.capture.Capture.grab()`; frame has `active_rect` and cropped `image_bgr`.
- Think: `core.planner.Planner` → `core.llm_client.LLMClient.propose_plan(frame, goal, context)`.
- Resolve: `core.resolver.Resolver.resolve(Target, frame)` fuses OCR (Paddle/Tesseract), edge NCC templates, ORB → one point+confidence.
- Act: `core.actuator.Actuator` taps/swipes via ADB (debounces duplicate taps).
- UI/Persistence: `ui/app.py` shows live feed/overlays; actions/results recorded via `core.datastore.DataStore`.

Key paths: `azl_bot/core/*.py`, tasks `azl_bot/tasks/*.py`, UI `azl_bot/ui/*.py`, templates `config/templates/*.png`, synonyms `config/selectors/synonyms.yaml`.

## Conventions (follow these)
- Coordinates are normalized in the active area; convert with `Capture.norm_to_pixels`/`Resolver.to_pixels` only when needed.
- Regions: one of `top_bar`, `bottom_bar`, `left_panel`, `center`, `right_panel` (see `core/screens.py` and `ResolverRegions`).
- Plan/Step JSON schema:
  - Step.action ∈ {tap, swipe, wait, back, assert}
  - Prefer Target kind {text|icon|bbox|point|region} with `region_hint`; avoid raw coords unless unavoidable.
- Resolver thresholds in `configs.ResolverConfig.thresholds` (e.g., `combo_accept` gates candidates).
- Templates: small PNG crops in `config/templates/` (e.g., `commissions_icon.png`); Target.value is the file stem.

## How things run
- Bootstrap: `core.bootstrap.bootstrap_from_config()` wires Device→Capture→OCR→Resolver→LLM→Actuator→Planner + tasks.
- Tasks implement: `goal() -> dict`, `success(frame, ctx) -> bool`, `on_success(planner, frame)` (see currencies/pickups/commissions).
- Planner context: device_w/h, `screens.get_region_info()`, `last_screen` (from `identify_screen`), full-frame OCR.

## Dev workflow
- GUI: `./scripts/run_gui.sh` or `python -m azl_bot.ui.app`.
- CLI tasks: `./scripts/run_task.sh {currencies|pickups|commissions}` or `python -m azl_bot.core.bootstrap <task>`.
- Config/Secrets: copy `config/app.yaml.example` → `config/app.yaml` (or set `AZL_CONFIG`); set `GEMINI_API_KEY` for LLM.
- Quick tests: `python tests/basic_test.py` or `python -c "from azl_bot.core.bootstrap import test_components; test_components()"`.

## Emulator/IO constraints
- ADB-only: use standard `adb`; do NOT use Genymotion gmtool/admin APIs. Waydroid default serial: `127.0.0.1:5555`.
- Verify connection/capture/input: `scripts/waydroid_setup.sh`.

## Example plan expected by Planner
```json
{"screen":"home","steps":[{"action":"tap","target":{"kind":"text","value":"Commissions","region_hint":"bottom_bar"}},{"action":"assert","target":{"kind":"text","value":"Commission","region_hint":"top_bar"}}],"done":false}
```

## Extending
- New task: add `azl_bot/tasks/<name>.py` (Task protocol) and register in `core/bootstrap.py`.
- New selector/icon: add PNG to `config/templates/`; add text synonyms in `config/selectors/synonyms.yaml`.
- Tune resolver: edit `AppConfig.resolver` in `config/app.yaml` or defaults in `core/configs.py`.

Unclear or missing? Tell us (e.g., DB schema details, extra screens), and we’ll refine these rules.
