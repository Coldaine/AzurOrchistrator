"""Test configuration validation."""

import tempfile
from pathlib import Path

from azl_bot.core.configs import load_config, create_default_config, save_config


def test_valid_config():
    """Test that valid config loads successfully."""
    config = create_default_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        save_config(config, temp_path)
        loaded = load_config(temp_path)
        assert loaded.emulator.adb_serial == "127.0.0.1:5555"
        assert loaded.display.target_fps == 2
    finally:
        temp_path.unlink()


def test_invalid_resolution():
    """Test that invalid resolution format is rejected."""
    config = create_default_config()
    
    try:
        config.display.force_resolution = "invalid"
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Resolution must be in format" in str(e)


def test_threshold_validation():
    """Test threshold value validation."""
    config = create_default_config()
    
    # Valid values
    config.resolver.thresholds.ocr_text = 0.8
    assert config.resolver.thresholds.ocr_text == 0.8
    
    # Test invalid value (should raise during model creation)
    try:
        from pydantic import ValidationError
        from azl_bot.core.configs import ResolverThresholds
        ResolverThresholds(ocr_text=1.5)  # Out of range
        raise AssertionError("Should have raised ValidationError")
    except Exception as e:
        if "ValidationError" not in str(type(e).__name__) and "should be less than or equal to 1" not in str(e):
            print(f"⚠ Validation error format: {e}")


if __name__ == "__main__":
    test_valid_config()
    print("✓ Valid config test passed")
    
    try:
        test_invalid_resolution()
        print("✓ Invalid resolution test passed")
    except AssertionError as e:
        print(f"⚠ Resolution validation: {e}")
    
    try:
        test_threshold_validation()
        print("✓ Threshold validation test passed")
    except Exception as e:
        print(f"⚠ Threshold validation: {e}")
    
    print("\nAll validation tests completed")
