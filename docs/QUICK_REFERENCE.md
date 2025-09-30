# Offline Resolver Harness & Dataset Capture - Quick Reference

## Resolver Harness

Test resolver without emulator:

```bash
# Single image
python -m azl_bot.tools.resolver_harness \
  --input screenshot.png \
  --target "text:Battle" \
  --save-overlays

# Directory of images
python -m azl_bot.tools.resolver_harness \
  --input ./screenshots/ \
  --target "icon:battle_icon" \
  --save-overlays \
  --out ./results

# Custom config
python -m azl_bot.tools.resolver_harness \
  --input ./dataset/20240930/ \
  --target "text:Commission" \
  --config ./config/test.yaml \
  --format both
```

**Outputs**: `results.csv`, `results.json`, overlay images (if `--save-overlays`)

## Dataset Capture

Enable in `config/app.yaml`:

```yaml
data:
  base_dir: "~/.azlbot"
  capture_dataset:
    enabled: true
    sample_rate_hz: 0.5
    max_dim: 1280
    format: "jpg"
    jpeg_quality: 85
    dedupe:
      method: "dhash"
      hamming_threshold: 3
    retention:
      max_files: 2000
      max_days: 60
    metadata: true
```

**Storage**: `~/.azlbot/dataset/YYYYMMDD/`

## Workflow

1. **Enable dataset capture** in config
2. **Run bot normally** - images captured passively
3. **Run harness** on captured dataset to test resolver
4. **Tune thresholds** in config
5. **Re-test** with harness
6. **Deploy** changes

## Key Files

- `azl_bot/tools/resolver_harness.py` - CLI tool
- `azl_bot/core/dataset_capture.py` - Capture module
- `azl_bot/core/resolver.py` - Template/ORB matching
- `docs/tools/RESOLVER_HARNESS.md` - Full guide
- `docs/data/DATASET_CAPTURE.md` - Capture guide
- `examples/resolver_harness_demo.py` - Demo script

## Resolver Thresholds

Tune in `config/app.yaml`:

```yaml
resolver:
  thresholds:
    ocr_text: 0.75
    ncc_edge: 0.60
    ncc_gray: 0.70
    orb_inliers: 12
    combo_accept: 0.65
    weights:
      ocr: 1.0
      template: 1.0
      orb: 0.9
      region_hint: 0.5
      llm_arbitration: 1.2
```

## Template Matching

- Multi-scale edge-NCC (5 scales: 0.8, 0.9, 1.0, 1.1, 1.2)
- ORB fallback for low NCC scores
- Pyramid caching for performance
- Typical resolve time: 5-50ms per scale

## Performance

- **Capture overhead**: <1% CPU, minimal I/O
- **Resolve time**: 50-200ms (OCR), 10-100ms (template+ORB)
- **Storage**: ~100KB per image at default settings
- **Dedup rate**: ~70-90% (depends on content)

## Testing

```bash
# Run unit tests
python tests/test_dataset_capture.py
python tests/test_template_cache.py
python tests/test_resolver_harness.py

# Create demo data and test
python examples/resolver_harness_demo.py
```

## See Also

- Full documentation in `docs/tools/` and `docs/data/`
- Config example: `config/app.yaml.example`
- Test fixtures: `tests/fixtures/README.md`
