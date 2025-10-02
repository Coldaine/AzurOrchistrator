# Test Refactoring Proposal

## Introduction

This proposal outlines targeted refactoring of the Azur Lane bot test suite to consolidate redundant and low-value tests into fewer, higher-quality integration tests. Based on analysis of the 127 existing tests (detailed in docs/dev/testingInitialReview.md) and recent changes from the PR merge report (docs/reports/PR_MERGE_REPORT_2025-10-01.md), the current suite has significant redundancy, particularly in configuration validation, coordinate transformations, and vision pipeline testing.

The refactoring focuses on three key areas where consolidation can reduce test count by approximately 40 tests (from 127 to ~87) while improving end-to-end coverage and maintainability. This aligns with the merge report's recommendations for test consolidation, particularly addressing duplicate hash testing, non-standard test runners, and scattered screen detection tests.

## Area 1: Configuration and Bootstrap

### Current State
Configuration and bootstrap testing is fragmented across 8 tests with significant redundancy:
- `basic_test.py`: 3 tests (test_config_loading, test_components, test_basic_initialization) - integration-level testing with no mocks
- `test_config_validation.py`: 3 tests (test_valid_config, test_invalid_resolution, test_threshold_validation) - superficial validation with broken tests
- `test_registry.py`: 4 tests - basic registry functionality

Issues include duplicate config loading tests, broken validation tests (F grade), and integration tests that don't isolate failures properly.

### Proposed Changes
Create a comprehensive integration test `test_config_bootstrap_pipeline` that validates the complete configuration-to-runtime pipeline:

```python
def test_config_bootstrap_pipeline(tmp_path):
    """Test complete config loading → validation → bootstrap → component wiring."""
    # Load and validate config
    config = load_config_from_file("config/app.yaml")
    assert config.emulator.adb_serial is not None
    assert config.resolver.thresholds.ocr_text > 0

    # Bootstrap all components
    components = bootstrap_from_config_object(config)

    # Verify component wiring and dependencies
    assert components["device"].serial == config.emulator.adb_serial
    assert components["resolver"].ocr is components["ocr"]
    assert components["actuator"].capture is components["capture"]

    # Test task registry integration
    tasks = get_all_tasks()
    assert "currencies" in tasks
    assert "commissions" in tasks

    # Verify configuration propagation
    assert components["llm"].config.api_key == config.llm.api_key
```

### Eliminated Tests
- `basic_test.py::test_config_loading` (redundant with new integration test)
- `basic_test.py::test_components` (covered by bootstrap verification)
- `basic_test.py::test_basic_initialization` (duplicate of test_components)
- `test_config_validation.py::test_valid_config` (integrated into pipeline)
- `test_registry.py::test_registry_import` (D grade, trivial)
- `test_registry.py::test_global_registry` (covered by task verification)

### Benefits
Eliminates 6 redundant/superficial tests while providing comprehensive end-to-end validation of the critical config-to-runtime pipeline. Improves failure isolation by testing components in proper dependency order rather than isolated units.

## Area 2: Coordinate System Pipeline

### Current State
The coordinate system has 20+ granular unit tests in `test_coordinates.py`, covering individual transform functions with high coverage but excessive granularity. Tests include basic transforms (A grade), edge cases (A grade), but also trivial validations (B/C grade). The merge report notes this area has good testing but could be more efficient.

### Proposed Changes
Consolidate into two focused integration tests that validate the complete coordinate pipeline:

```python
def test_coordinate_pipeline_end_to_end():
    """Test complete norm ↔ pixel ↔ device coordinate transformations."""
    # Test multiple resolutions and viewports
    resolutions = [(1920, 1080), (1440, 900), (2560, 1440)]

    for width, height in resolutions:
        viewport = (0, 40, width, height - 40)  # Letterboxed viewport

        # Test round-trip accuracy
        test_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.25, 0.75)]
        for norm_x, norm_y in test_points:
            # norm → pixel → norm (should be identical)
            pixel_x, pixel_y = denormalize_point(norm_x, norm_y, viewport)
            back_norm_x, back_norm_y = normalize_point(pixel_x, pixel_y, viewport)
            assert abs(back_norm_x - norm_x) < 1e-6
            assert abs(back_norm_y - norm_y) < 1e-6

def test_coordinate_bounds_and_safety():
    """Test coordinate validation, bounds checking, and actuator integration."""
    # Test bounds clamping
    assert clamp_to_bounds(-0.1, 1.5) == (0.0, 1.0)

    # Test actuator integration with bounds
    mock_device = Mock()
    actuator = Actuator(mock_device)

    # Should clamp out-of-bounds coordinates
    actuator.tap_norm(1.5, -0.2, active_rect=(0, 0, 1920, 1080))
    mock_device.input_tap.assert_called_with(1919, 0)  # Clamped to bounds
```

### Eliminated Tests
- `test_coordinates.py::TestCoordinateTransforms::test_norm_to_pixels_basic` (covered by pipeline test)
- `test_coordinates.py::TestCoordinateTransforms::test_norm_to_pixels_corners` (integrated)
- `test_coordinates.py::TestCoordinateTransforms::test_pixels_to_norm_basic` (covered)
- `test_coordinates.py::TestCoordinateTransforms::test_pixels_to_norm_corners` (integrated)
- `test_coordinates.py::TestCoordinateTransforms::test_round_trip_conversion` (explicitly tested)
- `test_coordinates.py::TestCoordinateTransforms::test_transform_pipeline_integration` (replaced)
- `test_coordinates.py::TestCoordinateTransforms::test_coordinate_precision` (covered by accuracy checks)
- `test_coordinates.py::TestCoordinateTransforms::test_actuator_coordinate_validation` (integrated into safety test)

### Benefits
Reduces 8+ granular tests to 2 comprehensive integration tests while maintaining full pipeline coverage. Focuses on end-to-end correctness and multi-resolution compatibility rather than individual function testing. Time tradeoff: ~10s for complete pipeline validation vs. ~2s for individual unit tests.

## Area 3: Resolver Detection Pipeline

### Current State
Vision and resolver testing is scattered across 15+ tests with mixed quality:
- `test_resolver_harness.py`: 6 tests (B grade) - tool testing rather than core logic
- `test_template_cache.py`: 3 tests (A grade) - good caching tests
- `test_dataset_capture.py`: Hash-related tests (duplicate with test_hashing.py)
- Screen detection split across `test_state_machine.py` and `test_ui_enhancements.py`

The merge report specifically calls out duplicate hash testing and recommends consolidating resolver-related tests.

### Proposed Changes
Create integrated vision pipeline tests that validate multi-method resolution:

```python
def test_resolver_multi_method_integration():
    """Test complete resolver pipeline: OCR + template + ORB + LLM arbitration."""
    # Mock frame with known elements
    mock_frame = create_mock_frame_with_elements()

    # Test text resolution
    text_target = Target(kind="text", value="Commissions")
    candidate = resolver.resolve(text_target, mock_frame)
    assert candidate is not None
    assert candidate.confidence > 0.7
    assert candidate.method == "ocr"

    # Test icon resolution with fallback
    icon_target = Target(kind="icon", value="commission_button")
    candidate = resolver.resolve(icon_target, mock_frame)
    assert candidate is not None
    # Should try template first, fall back to ORB if needed

    # Test arbitration scenario (force disagreement)
    with patch.object(resolver, '_detect_by_ocr', return_value=[]):
        with patch.object(resolver, '_detect_by_template', return_value=[]):
            # Force LLM arbitration
            candidate = resolver.resolve(text_target, mock_frame)
            assert candidate.method == "llm"

def test_vision_pipeline_performance():
    """Test vision pipeline performance and caching."""
    # Test template caching
    start_time = time.time()
    for _ in range(10):
        template = resolver._get_template("test_icon")
    cache_time = time.time() - start_time

    # Test uncached performance
    resolver._template_cache.clear()
    start_time = time.time()
    for _ in range(10):
        template = resolver._get_template("test_icon")
    uncached_time = time.time() - start_time

    assert cache_time < uncached_time * 0.5  # At least 2x speedup
```

### Eliminated Tests
- `test_resolver_harness.py::test_load_image_as_frame` (tool-specific, not core logic)
- `test_resolver_harness.py::test_draw_candidates_overlay` (UI tool testing)
- `test_resolver_harness.py::test_save_results_csv` (export testing)
- `test_resolver_harness.py::test_save_results_json` (export testing)
- `test_dataset_capture.py::test_compute_dhash` (duplicate with test_hashing.py)
- `test_dataset_capture.py::test_hamming_distance` (duplicate)
- `test_dataset_capture.py::test_dhash_similarity` (duplicate)
- `test_template_cache.py::test_template_cache_warm_cold` (integrated into performance test)

### Benefits
Consolidates 8+ scattered tests into 2 focused integration tests covering the complete multi-modal vision pipeline. Eliminates duplicate hash testing as recommended in the merge report. Provides better validation of the resolver's arbitration logic and performance characteristics.

## Implementation Plan

1. **Create new test files** in `tests/core/` for consolidated tests
2. **Move existing tests** to appropriate locations (e.g., hash tests to `tests/core/test_hashing.py`)
3. **Update imports** and remove eliminated test functions
4. **Add shared fixtures** in `tests/conftest.py` for common test data
5. **Run test suite** to verify no regressions
6. **Update CI configuration** if needed for new test structure

## Conclusion

This refactoring reduces the test suite from 127 to approximately 87 tests (-32%) by eliminating redundancy while improving integration coverage. The three targeted areas address the merge report's key concerns: duplicate functionality, non-standard test patterns, and scattered component testing. The consolidated tests provide better end-to-end validation of critical pipelines (config→bootstrap, coordinate transforms, vision resolution) with improved maintainability and faster execution through reduced setup overhead.

Expected outcomes include fewer false positives from broken tests, easier debugging through integrated assertions, and better alignment with the project's focus on robust automation pipelines rather than granular unit isolation.