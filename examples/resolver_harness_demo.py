#!/usr/bin/env python3
"""Example script demonstrating the resolver harness."""

import cv2
import numpy as np
from pathlib import Path

# Create example directory
example_dir = Path("/tmp/resolver_harness_example")
example_dir.mkdir(exist_ok=True)

screenshots_dir = example_dir / "screenshots"
templates_dir = example_dir / "templates"
screenshots_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

print(f"Creating example data in {example_dir}")

# Create a simple template (button icon)
template = np.zeros((64, 64, 3), dtype=np.uint8)
cv2.circle(template, (32, 32), 24, (255, 255, 255), -1)
cv2.circle(template, (32, 32), 20, (0, 128, 255), -1)
template_path = templates_dir / "battle_icon.png"
cv2.imwrite(str(template_path), template)
print(f"Created template: {template_path}")

# Create a screenshot with the template embedded
screenshot = np.ones((720, 1280, 3), dtype=np.uint8) * 220

# Add some UI elements
cv2.rectangle(screenshot, (0, 0), (1280, 80), (50, 50, 50), -1)  # Top bar
cv2.rectangle(screenshot, (0, 640), (1280, 720), (50, 50, 50), -1)  # Bottom bar

# Add text
cv2.putText(screenshot, "Azur Lane", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
cv2.putText(screenshot, "Battle", (920, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(screenshot, "Commission", (520, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Embed the template at a known location
template_h, template_w = template.shape[:2]
x, y = 900, 660
screenshot[y:y+template_h, x:x+template_w] = template

screenshot_path = screenshots_dir / "home_screen.png"
cv2.imwrite(str(screenshot_path), screenshot)
print(f"Created screenshot: {screenshot_path}")

# Create a second screenshot (different scene)
screenshot2 = np.ones((720, 1280, 3), dtype=np.uint8) * 200
cv2.rectangle(screenshot2, (0, 0), (1280, 80), (40, 40, 40), -1)
cv2.putText(screenshot2, "Commissions", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
cv2.putText(screenshot2, "Available: 4", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

screenshot2_path = screenshots_dir / "commission_screen.png"
cv2.imwrite(str(screenshot2_path), screenshot2)
print(f"Created screenshot: {screenshot2_path}")

print("\n" + "="*60)
print("Example data created! Now run the harness:")
print("="*60)
print()
print("# Test text detection:")
print(f"python -m azl_bot.tools.resolver_harness \\")
print(f"  --input {screenshots_dir} \\")
print(f"  --target 'text:Battle' \\")
print(f"  --save-overlays \\")
print(f"  --out {example_dir}/results_text")
print()
print("# Test template matching:")
print(f"python -m azl_bot.tools.resolver_harness \\")
print(f"  --input {screenshot_path} \\")
print(f"  --target 'icon:battle_icon' \\")
print(f"  --save-overlays \\")
print(f"  --out {example_dir}/results_template")
print()
print("# Batch processing:")
print(f"python -m azl_bot.tools.resolver_harness \\")
print(f"  --input {screenshots_dir} \\")
print(f"  --target 'text:Commission' \\")
print(f"  --format json \\")
print(f"  --out {example_dir}/results_batch")
print()
print("="*60)
print(f"\nResults will be saved to: {example_dir}/results_*")
print("Check CSV/JSON files for detailed results and PNG overlays for visualizations.")
