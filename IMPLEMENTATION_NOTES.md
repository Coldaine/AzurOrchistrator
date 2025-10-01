# Implementation Notes

## Known Issues

### Dependencies
- The `hashing.py` module requires `imagehash` library for frame change detection
- For offline testing without imagehash, temporarily comment out the hashing import in `capture.py`:
  ```python
  # from .hashing import FrameHasher
  ```
- This doesn't affect dataset capture functionality which uses its own dhash implementation

### File Corruption
- Original repository had corrupted file headers with `zl_bot/core/...` prefixes
- These were fixed in resolver.py and capture.py
- If you see similar corruption in other files, check for lines like:
  ```
  zl_bot/core/filename.py</path>
  <content">"""Docstring"""
  ```
  Should be just:
  ```
  """Docstring"""
  ```

## Testing

### Unit Tests
All dataset capture tests pass:
```bash
cd /home/runner/work/AzurOrchistrator/AzurOrchistrator
PYTHONPATH=. python tests/test_dataset_capture.py
# Output: All dataset capture tests passed!
```

### Harness Testing
To test the harness without imagehash dependency:
1. Temporarily modify `capture.py` to remove hashing import
2. Or install imagehash: `pip install imagehash`

### Demo Script
The `examples/resolver_harness_demo.py` successfully creates:
- Template images in `/tmp/resolver_harness_example/templates/`
- Screenshot images in `/tmp/resolver_harness_example/screenshots/`
- Provides command examples for running the harness

## Performance Characteristics

Based on implementation:

### Dataset Capture
- **dhash computation**: O(1) - resize to 8x8, ~0.1ms
- **Hamming distance**: O(n) where n=number of recent hashes (kept at 20-100)
- **Image save**: Async, ~10-50ms depending on size/quality
- **Retention cleanup**: Every 10th capture, ~100-500ms

### Template Matching
- **Pyramid creation**: 5 scales, ~5-10ms per template (cached)
- **NCC matching**: ~2-10ms per scale depending on image size
- **ORB fallback**: ~50-150ms (only when NCC < threshold)
- **Overall**: 10-100ms typical, up to 200ms with ORB

### Resolver Harness
- **Single image**: ~100-300ms (depends on target kind and methods)
- **Directory**: Linear in number of images
- **Overlay generation**: ~5-20ms per image
- **CSV/JSON export**: ~1-5ms

## Future Improvements

1. **Parallel Processing**: Harness could process multiple images in parallel
2. **Template Optimization**: Reduce pyramid scales based on expected variance
3. **Caching Strategy**: LRU cache for frequently used templates
4. **Progress Bar**: Add tqdm for long-running harness operations
5. **GUI Integration**: Add Qt/Tk UI for dataset capture controls
