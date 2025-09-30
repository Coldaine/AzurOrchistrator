# Dataset Capture

The **Dataset Capture** feature passively collects screenshots during normal bot operation to build a reproducible test dataset. This accelerates offline development, regression testing, and resolver calibration.

## Overview

Dataset capture runs in the background while you use the bot, saving deduplicated, resized images with metadata. The feature is:

- **Passive**: No impact on bot behavior or timing
- **Storage-aware**: Automatic deduplication and retention limits
- **Reproducible**: Metadata includes timestamps, hashes, frame dimensions, and context

## Enabling Dataset Capture

### Via Configuration

Edit `config/app.yaml`:

```yaml
data:
  base_dir: "~/.azlbot"
  capture_dataset:
    enabled: true  # Enable capture
    sample_rate_hz: 0.5  # Capture up to 0.5 images per second
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

Then restart the bot to apply changes.

### Via GUI Toggle

(To be implemented) The GUI will have a toggle button to enable/disable capture on the fly, plus:

- Counter showing images captured this session
- Button to open the current day's capture directory

## Configuration Options

### `enabled`

- **Type**: Boolean
- **Default**: `false`
- **Description**: Master switch for dataset capture

### `sample_rate_hz`

- **Type**: Float
- **Default**: `0.5`
- **Description**: Maximum capture rate in images per second. At 0.5 Hz, the bot captures at most one image every 2 seconds.

### `max_dim`

- **Type**: Integer
- **Default**: `1280`
- **Description**: Maximum dimension (width or height) for captured images. Larger images are resized while preserving aspect ratio.

### `format`

- **Type**: String (`jpg` or `png`)
- **Default**: `jpg`
- **Description**: Image format for captured files. JPEG is recommended for storage efficiency.

### `jpeg_quality`

- **Type**: Integer (1-100)
- **Default**: `85`
- **Description**: JPEG compression quality. Higher values mean better quality but larger files.

### `dedupe.method`

- **Type**: String
- **Default**: `dhash`
- **Description**: Deduplication algorithm. Currently only `dhash` (difference hash) is supported.

### `dedupe.hamming_threshold`

- **Type**: Integer
- **Default**: `3`
- **Description**: Maximum Hamming distance between hashes to consider images duplicates. Lower values mean stricter deduplication.

### `retention.max_files`

- **Type**: Integer
- **Default**: `2000`
- **Description**: Maximum number of captured images to keep. Oldest files are deleted first.

### `retention.max_days`

- **Type**: Integer
- **Default**: `60`
- **Description**: Maximum age of captured images in days. Older files are automatically deleted.

### `metadata`

- **Type**: Boolean
- **Default**: `true`
- **Description**: Save JSON metadata files alongside images.

## Directory Structure

Captured images are organized by date:

```
~/.azlbot/dataset/
├── 20240930/
│   ├── 20240930_143022_456_a1b2c3d4.jpg
│   ├── 20240930_143022_456.json
│   ├── 20240930_143025_789_e5f6g7h8.jpg
│   ├── 20240930_143025_789.json
│   └── ...
├── 20241001/
│   └── ...
└── ...
```

### Filenames

Images: `<timestamp>_<short_hash>.<ext>`

- `timestamp`: `YYYYMMDD_HHMMSS_mmm` (millisecond precision)
- `short_hash`: First 8 characters of dhash
- `ext`: `jpg` or `png`

Metadata: `<timestamp>.json`

## Metadata Format

Each captured image has a JSON sidecar with:

```json
{
  "timestamp": 1696086622.456,
  "timestamp_str": "2024-09-30T14:30:22.456000",
  "hash": "a1b2c3d4e5f6g7h8",
  "frame_size": {
    "full_width": 1920,
    "full_height": 1080
  },
  "active_rect": {
    "x": 0,
    "y": 60,
    "width": 1920,
    "height": 960
  },
  "captured_size": {
    "width": 1280,
    "height": 720
  },
  "context": {
    "screen": "home",
    "task": "currencies",
    "action": "tap"
  }
}
```

- `timestamp`: Unix timestamp (seconds since epoch)
- `timestamp_str`: ISO 8601 formatted timestamp
- `hash`: Full dhash for deduplication
- `frame_size`: Original emulator resolution
- `active_rect`: Active area after letterbox detection
- `captured_size`: Final image size after resizing
- `context`: Optional context (screen name, task, action, etc.)

## Deduplication

Dataset capture uses **difference hashing (dhash)** to avoid storing near-identical frames:

1. Each image is converted to grayscale and resized to 8x8
2. Horizontal gradients are computed
3. The resulting 64-bit hash is compared to recent hashes
4. If Hamming distance ≤ `hamming_threshold`, the image is skipped

This removes redundant captures (e.g., static screens, animation frames) while keeping meaningful changes.

## Retention Policy

To prevent unbounded disk usage, old files are automatically deleted based on:

1. **Age**: Files older than `max_days` are removed
2. **Count**: If total files exceed `max_files`, oldest files are removed first

Cleanup runs every 10th capture to minimize overhead. Empty date directories are also removed.

## Disk Usage Estimates

Approximate storage per image at default settings (1280px max dimension, JPEG quality 85):

- **Simple UI screens**: 50-150 KB
- **Complex scenes**: 150-300 KB

With `max_files: 2000` and average 100 KB per image:

- **Total storage**: ~200 MB

Actual usage depends on game content complexity and capture rate.

## Using Captured Dataset

### With Resolver Harness

Test resolver on captured images:

```bash
python -m azl_bot.tools.resolver_harness \
  --input ~/.azlbot/dataset/20240930/ \
  --target "text:Commissions" \
  --save-overlays \
  --out harness_results
```

See [Resolver Harness Guide](../tools/RESOLVER_HARNESS.md) for details.

### Regression Testing

Before deploying resolver changes:

```bash
# Capture baseline results
python -m azl_bot.tools.resolver_harness \
  --input ~/.azlbot/dataset/ \
  --target "icon:battle" \
  --out baseline

# Make code changes...

# Capture new results
python -m azl_bot.tools.resolver_harness \
  --input ~/.azlbot/dataset/ \
  --target "icon:battle" \
  --out updated

# Compare
diff baseline/results.csv updated/results.csv
```

### Manual Inspection

Browse captured images to:

- Verify letterbox detection
- Check for capture quality
- Identify interesting frames for template creation

## Best Practices

### Recommended Settings

For **development and testing**:

```yaml
capture_dataset:
  enabled: true
  sample_rate_hz: 1.0  # More frequent captures
  max_dim: 1280
  format: "jpg"
  jpeg_quality: 90  # Higher quality for detailed inspection
```

For **long-term operation**:

```yaml
capture_dataset:
  enabled: true
  sample_rate_hz: 0.2  # Less frequent captures
  max_dim: 1024
  format: "jpg"
  jpeg_quality: 80  # Lower quality for storage efficiency
```

### Capture Strategy

- **Enable during varied gameplay**: Capture while navigating different screens to build diverse dataset
- **Disable for repetitive tasks**: Turn off capture for farming/grinding to avoid storage waste
- **Periodic review**: Check `~/.azlbot/dataset/` weekly to ensure quality

### Storage Management

- Monitor disk usage: `du -sh ~/.azlbot/dataset/`
- Adjust `max_files` and `max_days` based on available space
- Archive old datasets before major bot updates

## Troubleshooting

### Capture not working

1. Check `enabled: true` in config
2. Verify `base_dir` path is writable
3. Check logs for errors (search for "dataset capture")

### Too many duplicates captured

- Increase `dedupe.hamming_threshold` (e.g., from 3 to 5)
- This makes deduplication stricter

### Too few captures

- Increase `sample_rate_hz`
- Decrease `dedupe.hamming_threshold` (but may capture more duplicates)
- Check bot activity - no captures if bot is idle

### High disk usage

- Decrease `max_files`
- Decrease `max_days`
- Lower `jpeg_quality` (e.g., from 85 to 70)
- Reduce `max_dim` (e.g., from 1280 to 960)

### Metadata missing

- Ensure `metadata: true` in config
- Check write permissions for dataset directory

## GUI Integration (Planned)

Future UI features:

- **Toggle button**: Enable/disable capture without editing config
- **Session counter**: Show images captured this session
- **Folder button**: Open current day's directory in file manager
- **Storage indicator**: Display disk usage and estimate time until retention limit

## Performance Impact

Dataset capture is designed to have minimal impact:

- **CPU**: Negligible (dhash computation is fast)
- **I/O**: Asynchronous writes (no blocking)
- **Memory**: Small buffer for recent hashes (~1 KB)

At default settings (0.5 Hz, 1280px), capture uses <1% CPU on modern hardware.

## Privacy and Security

- **Local storage only**: Captured images never leave your machine
- **No network activity**: Dataset capture does not send data anywhere
- **Sensitive data**: Be mindful of captured images containing personal info (usernames, etc.)

## See Also

- [Resolver Harness Guide](../tools/RESOLVER_HARNESS.md) - Using captured images for testing
- [Configuration Reference](../../config/app.yaml.example) - Full config options
- [Hashing Implementation](../../azl_bot/core/hashing.py) - Frame change detection details
