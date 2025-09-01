# Tasks playbook — first end-to-end run

This page outlines a concise, practical sequence to get the bot capturing, resolving, and tapping reliably.

## Overview

- Create a real config and wire ADB + LLM keys
- Verify emulator connectivity and screencap
- Seed 2–3 templates and text synonyms
- Launch the GUI and confirm OCR/overlays
- Run one CLI task and tune thresholds if needed

## 1) Create `config/app.yaml`

- Copy `config/app.yaml.example` to `config/app.yaml` and edit:
  - `emulator.adb_serial`: Waydroid default is `127.0.0.1:5555`
  - `llm.api_key_env`: ensure you export `GEMINI_API_KEY`
  - Optionally set `resolver.ocr: tesseract` if PaddleOCR isn’t installed yet

Example environment setup (put in your shell profile):

```bash
export GEMINI_API_KEY="<your_key>"
```

## 2) Verify ADB connection and screencap

- Ensure the emulator is running and reachable at the serial you set.
- Validate that `adb -s <serial> exec-out screencap -p` returns image bytes.
- If not connected, try `adb disconnect <serial>` then `adb connect <serial>`.

## 3) Seed selectors: templates + synonyms

- Add small cropped PNGs in `config/templates/` for commonly tapped items:
  - `commissions_icon.png`, `mailbox_icon.png`, `missions_icon.png` (examples)
- Add text variants in `config/selectors/synonyms.yaml`:
  - Include forms like `Commission`, `Commissions`, `Tasks`, etc.

Why: The Resolver combines OCR, template (edge NCC), and ORB features; seeding a few high-signal selectors increases early success.

## 4) Launch the GUI and validate the pipeline

- Run the GUI via `./scripts/run_gui.sh` or `python -m azl_bot.ui.app`.
- Check you see:
  - Live capture and correct active-area crop (no letterbox in `image_bgr`)
  - Overlays drawn (if `logging.overlay_draw` is enabled)
  - OCR returning some text in logs

## 5) Run one task

- Try one of:
  - `./scripts/run_task.sh pickups`
  - `./scripts/run_task.sh commissions`
  - `./scripts/run_task.sh currencies`
- Watch logs for Resolver confidence and chosen method (`ocr`, `template`, `orb`).

If resolution is flaky:

- Slightly relax thresholds in `config/app.yaml` (under `resolver.thresholds`):
  - `ncc_edge` (e.g., 0.55–0.60), `combo_accept` (e.g., 0.60–0.65)
- Add another, tighter template crop or a synonym variant.

## Success criteria

- GUI shows live capture with correct active area
- Resolver locates at least one seeded icon/text on the home screen with acceptable confidence
- A task performs a tap and logs a non-zero confidence

## Troubleshooting quickies

- OCR
  - Tesseract: install system binary (e.g., `tesseract-ocr` on Debian/Ubuntu) and `pytesseract`
  - PaddleOCR: install `paddlepaddle` + `paddleocr`; set `resolver.ocr: paddle`
- ADB
  - Ensure `adb devices` lists your serial
  - On errors, try reconnect (`adb disconnect`/`adb connect`)
- LLM
  - Ensure `GEMINI_API_KEY` is exported and reachable
  - Planner will be disabled if LLM init fails; logs will note this

## Next small improvement

- Add a tiny static test image and a unit test around `Resolver` template matching to tune thresholds without the emulator.
