"""Bootstrap and component initialization."""

def test_components() -> bool:
    """Test that basic components can be initialized.
    
    Returns:
        True if initialization succeeds
    """
    try:
        from .configs import create_default_config
        
        # Just test config creation for now
        config = create_default_config()
        assert config is not None
        
        return True
    except Exception as e:
        print(f"Component initialization failed: {e}")
        return False
