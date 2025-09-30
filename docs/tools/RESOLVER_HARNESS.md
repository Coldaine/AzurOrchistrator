# Resolver Harness

The **Resolver Harness** is an offline testing tool that allows you to calibrate, debug, and regression-test OCR, template matching, and ORB feature detection without requiring a running emulator.

## Purpose

- **Accelerate iteration**: Test resolver changes on a directory of screenshots without waiting for the emulator
- **Regression testing**: Ensure resolver updates don't break existing detections
- **Calibration**: Tune thresholds (NCC, ORB inliers, confidence weights) with immediate feedback
- **Debugging**: Visualize candidates with overlays showing detection methods, confidence scores, and coordinates

## Quick Start

### Basic Usage

Run the harness on a single image:

```bash
python -m azl_bot.tools.resolver_harness \
  --input screenshots/home_screen.png \
  --target "text:Commissions" \
  --save-overlays \
  --out ./results
```

Run on a directory of images:

```bash
python -m azl_bot.tools.resolver_harness \
  --input screenshots/ \
  --target "icon:battle_icon" \
  --save-overlays \
  --out ./batch_results
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Path to image file or directory (required) |
| `--target`, `-t` | Target to resolve: `kind:value` (e.g., `text:Battle`, `icon:commissions_icon`) |
| `--save-overlays` | Save visualization images with detection boxes and labels |
| `--out`, `-o` | Output directory (default: `./harness_output`) |
| `--config`, `-c` | Path to config file (default: uses `AZL_CONFIG` env or defaults) |
| `--format` | Result format: `csv`, `json`, or `both` (default: `both`) |

## Understanding Results

### CSV Output

The harness generates `results.csv` with columns:

- `image_path`: Path to source image
- `target_kind`: Target type (text, icon, region, etc.)
- `target_value`: Target identifier
- `top_method`: Detection method that won (ocr, template, orb, region_hint)
- `top_x_norm`, `top_y_norm`: Normalized coordinates of detected point (0.0-1.0)
- `top_confidence`: Confidence score (0.0-1.0)
- `resolve_time_ms`: Total resolution time in milliseconds
- `num_candidates`: Number of candidates found

### JSON Output

The `results.json` file contains detailed information:

```json
{
  "image_path": "screenshots/home_screen.png",
  "target_kind": "text",
  "target_value": "Commissions",
  "candidates": [
    {
      "method": "ocr",
      "x_norm": 0.512,
      "y_norm": 0.891,
      "confidence": 0.85
    }
  ],
  "top_candidate": {
    "method": "ocr",
    "x_norm": 0.512,
    "y_norm": 0.891,
    "confidence": 0.85
  },
  "resolve_time_ms": 142.3,
  "method_times": {
    "text": 138.2,
    "template": 0.0,
    "region": 4.1
  }
}
```

### Overlay Visualization

When `--save-overlays` is used, the harness generates annotated images:

- **Circles** mark candidate locations
- **Color** indicates detection method:
  - Green: OCR
  - Blue: Template (NCC)
  - Orange: ORB features
  - Gray: Region hint
  - Magenta: LLM arbitration
- **Labels** show method abbreviation and confidence (e.g., `ocr 0.85`)
- **Crosshairs** highlight the top selected candidate

## Tuning Thresholds

Edit `config/app.yaml` (or pass `--config`) to adjust resolver thresholds:

```yaml
resolver:
  thresholds:
    ocr_text: 0.75       # Minimum OCR text match confidence
    ncc_edge: 0.60       # Edge-based NCC threshold
    ncc_gray: 0.70       # Grayscale NCC threshold
    orb_inliers: 12      # Minimum ORB inliers for match
    combo_accept: 0.65   # Minimum weighted confidence to accept
    weights:
      ocr: 1.0           # Weight for OCR method
      template: 1.0      # Weight for template matching
      orb: 0.9           # Weight for ORB features
      region_hint: 0.5   # Weight for region hints
      llm_arbitration: 1.2  # Weight for LLM arbitration
```

Workflow for tuning:

1. Capture a dataset of screenshots (see [Dataset Capture](../data/DATASET_CAPTURE.md))
2. Run harness with current thresholds: `python -m azl_bot.tools.resolver_harness --input dataset/20240930/ --target "text:Battle"`
3. Check success rate and inspect overlays
4. Adjust thresholds in config
5. Re-run harness and compare

## Comparing Runs

To compare threshold changes:

```bash
# Baseline run
python -m azl_bot.tools.resolver_harness --input dataset/ --target "icon:sortie" --out results_baseline

# After threshold adjustment
python -m azl_bot.tools.resolver_harness --input dataset/ --target "icon:sortie" --out results_tuned

# Compare CSV outputs
diff results_baseline/results.csv results_tuned/results.csv
```

Use a spreadsheet or data analysis tool to compare success rates, average confidence, and timing.

## Examples

### Example 1: Test Text Detection

```bash
python -m azl_bot.tools.resolver_harness \
  --input screenshots/commissions_screen.png \
  --target "text:Commission" \
  --save-overlays \
  --out test_commission
```

### Example 2: Batch Template Matching

```bash
python -m azl_bot.tools.resolver_harness \
  --input dataset/20240930/ \
  --target "icon:battle_icon" \
  --format json \
  --out batch_battle
```

### Example 3: Custom Config

```bash
python -m azl_bot.tools.resolver_harness \
  --input screenshots/ \
  --target "text:Sortie" \
  --config config/test_resolver.yaml \
  --save-overlays \
  --out custom_config_test
```

## Interpreting Confidence Scores

- **0.9-1.0**: Excellent match, very reliable
- **0.75-0.89**: Good match, generally reliable
- **0.60-0.74**: Acceptable match, may need verification
- **Below 0.60**: Weak match, likely incorrect

The combined confidence is computed as:

```
combined_confidence = method_confidence * method_weight
```

Where `method_weight` comes from `resolver.thresholds.weights` in the config.

## Normalization and Coordinates

- All coordinates are **normalized** (0.0 to 1.0) relative to the **active area** of the frame
- The active area excludes letterbox borders
- Top-left is `(0.0, 0.0)`, bottom-right is `(1.0, 1.0)`
- To convert to pixel coordinates for a specific frame:
  ```python
  x_px = x_norm * active_width
  y_px = y_norm * active_height
  ```

## Performance Expectations

Typical resolve times on a modern CPU:

- **Text (OCR)**: 50-200ms per image (depends on OCR backend)
- **Template (NCC)**: 5-50ms per scale
- **ORB fallback**: 50-150ms

The harness reports total time and per-method breakdown to help identify bottlenecks.

## Troubleshooting

### No candidates found

- **Cause**: Target not present in image, or thresholds too strict
- **Fix**: Lower thresholds, verify target name matches template filename or OCR text

### Low confidence scores

- **Cause**: Template quality, lighting differences, or OCR errors
- **Fix**: Recapture templates from current game version, adjust NCC/OCR thresholds

### High false positive rate

- **Cause**: Template too generic, thresholds too permissive
- **Fix**: Increase thresholds, use more specific templates

### Slow performance

- **Cause**: Large images, many scales, OCR backend
- **Fix**: Reduce `max_dim` in dataset capture, limit template scales, switch OCR backend

## Advanced Usage

### Custom target types

The harness supports all target kinds:

- `text:SomeText` - OCR-based text matching
- `icon:template_name` - Template matching (requires `config/templates/template_name.png`)
- `region:region_name` - Region hint (top_bar, bottom_bar, center, etc.)
- `point:0.5,0.5` - Direct normalized coordinates

### Batch processing with scripts

```bash
#!/bin/bash
for target in "text:Battle" "text:Commission" "icon:sortie"; do
  python -m azl_bot.tools.resolver_harness \
    --input dataset/latest/ \
    --target "$target" \
    --out "results_${target//[:\/]/_}"
done
```

## See Also

- [Dataset Capture Guide](../data/DATASET_CAPTURE.md) - How to collect test images
- [Resolver Implementation](../IMPLEMENTATION.md#selector-resolver) - Technical details
- [Configuration Reference](../../config/app.yaml.example) - Full config options
