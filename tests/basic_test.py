"""Basic test for component initialization."""

import sys
import tempfile
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from azl_bot.core.bootstrap import test_components
from azl_bot.core.configs import create_default_config


def test_basic_initialization():
    """Test that components can be initialized without errors."""
    print("Testing component initialization...")
    
    # Create a temporary config
    config = create_default_config()
    config.emulator.adb_serial = "test_device"
    config.data.base_dir = str(tempfile.mkdtemp())
    
    try:
        success = test_components()
        if success:
            print("✓ Component initialization test passed")
            return True
        else:
            print("✗ Component initialization test failed")
            return False
    except Exception as e:
        print(f"✗ Component initialization test failed with exception: {e}")
        return False


def test_config_loading():
    """Test configuration loading and validation."""
    print("Testing configuration system...")
    
    try:
        from azl_bot.core.configs import create_default_config, save_config, load_config
        
        # Test default config creation
        config = create_default_config()
        assert config.emulator.adb_serial == "127.0.0.1:5555"
        assert config.llm.provider == "gemini"
        print("✓ Default config creation passed")
        
        # Test config serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        save_config(config, temp_path)
        assert temp_path.exists()
        print("✓ Config save passed")
        
        # Test config loading
        loaded_config = load_config(temp_path)
        assert loaded_config.emulator.adb_serial == config.emulator.adb_serial
        assert loaded_config.llm.provider == config.llm.provider
        print("✓ Config load passed")
        
        # Cleanup
        temp_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Azur Lane Bot - Basic Tests")
    print("===========================")
    
    tests = [
        test_config_loading,
        test_basic_initialization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()