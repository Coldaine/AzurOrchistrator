# Test Fixtures

This directory contains test fixtures for the resolver harness and other testing tools.

## Directory Structure

```
tests/fixtures/
├── templates/          # Small template images for template matching tests
├── screenshots/        # Sample screenshots for resolver testing
└── README.md          # This file
```

## Templates

Template files should be small PNG images (30-100px) showing UI elements:

- `battle_icon.png` - Battle button icon
- `commissions_icon.png` - Commissions button icon
- etc.

These are used in unit tests to verify template matching and caching behavior.

## Screenshots

Sample screenshots for testing the resolver harness:

- Capture from emulator at various resolutions (720p, 1080p)
- Different game screens (home, battle, commissions, etc.)
- Include both success and edge cases

**Note**: Do not commit actual game art due to copyright. Use synthetic/placeholder images for testing.

## Creating Test Images

Use the following script to create simple synthetic test images:

```python
import cv2
import numpy as np

# Create a simple template
template = np.zeros((64, 64, 3), dtype=np.uint8)
cv2.circle(template, (32, 32), 20, (255, 255, 255), -1)
cv2.imwrite('templates/test_icon.png', template)

# Create a simple screenshot
screenshot = np.ones((720, 1280, 3), dtype=np.uint8) * 200
# Add some features
cv2.rectangle(screenshot, (100, 100), (300, 200), (255, 0, 0), -1)
cv2.putText(screenshot, "TEST", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
cv2.imwrite('screenshots/test_screen.png', screenshot)
```

## Usage in Tests

Tests can load fixtures like:

```python
from pathlib import Path

fixtures_dir = Path(__file__).parent / "fixtures"
template_path = fixtures_dir / "templates" / "test_icon.png"
screenshot_path = fixtures_dir / "screenshots" / "test_screen.png"
```
