# Azur Lane Bot Template Images

This directory contains template images used for computer vision-based element detection in the Azur Lane game interface.

## Required Template Images

The following template images need to be created for the bot to function properly:

### 1. home_button.png
- **Purpose**: Detect the home button in the bottom navigation
- **Source**: Screenshot of the home button from Azur Lane's bottom navigation bar
- **Cropping**: Crop tightly around the button icon, include minimal background
- **Contrast**: Ensure high contrast between button and background
- **Size**: Approximately 50-100px square (will be scaled during matching)

### 2. back_button.png
- **Purpose**: Detect the back button in dialogs and menus
- **Source**: Screenshot of the back arrow button (usually top-left)
- **Cropping**: Crop tightly around the arrow icon
- **Contrast**: High contrast preferred
- **Size**: Approximately 40-80px square

### 3. commission_icon.png
- **Purpose**: Detect commission-related UI elements
- **Source**: Screenshot of the commission icon from the home screen
- **Cropping**: Crop tightly around the commission icon
- **Contrast**: Ensure the icon stands out clearly
- **Size**: Approximately 60-120px square

### 4. mail_icon.png
- **Purpose**: Detect mailbox/mail icon for pickup notifications
- **Source**: Screenshot of the mail icon (usually shows notification badges)
- **Cropping**: Crop tightly around the envelope or mail icon
- **Contrast**: High contrast to distinguish from background
- **Size**: Approximately 50-100px square

### 5. close_x.png
- **Purpose**: Detect close/X buttons in dialogs and popups
- **Source**: Screenshot of the close button (usually red X in corner)
- **Cropping**: Crop tightly around the X symbol
- **Contrast**: Very high contrast (red X on white/light background)
- **Size**: Approximately 30-60px square

## How to Create Template Images

### Step 1: Capture Screenshots
1. Run Azur Lane on your emulator
2. Navigate to screens containing the target UI elements
3. Take high-resolution screenshots (preferably 1080p or higher)

### Step 2: Crop Images
1. Use image editing software (GIMP, Photoshop, or similar)
2. Crop tightly around each UI element
3. Remove as much background as possible while keeping the element intact
4. Maintain aspect ratio of the original element

### Step 3: Optimize Contrast
1. Adjust brightness/contrast to maximize difference between element and background
2. Convert to grayscale if it improves clarity
3. Ensure the element is clearly distinguishable from any background noise

### Step 4: Save Images
1. Save as PNG format (preserves transparency if needed)
2. Use descriptive filenames matching the names above
3. Place files in this `config/templates/` directory

## Template Matching Guidelines

### Image Quality Requirements
- **Resolution**: High resolution (at least 100x100px for important elements)
- **Format**: PNG with transparency support
- **Color**: Maintain original colors unless grayscale improves matching
- **Noise**: Minimal background noise or artifacts

### Best Practices
1. **Consistency**: Use screenshots from the same device/resolution as your target
2. **Clean Background**: Avoid including dynamic elements like animations or particles
3. **Multiple Variants**: Consider creating variants for different states (pressed/unpressed)
4. **Regular Updates**: Update templates when game UI changes

### Testing Templates
After creating templates, test them by:
1. Running the bot with debug logging enabled
2. Checking template matching confidence scores
3. Adjusting cropping/contrast if confidence is low (<0.6)

## Troubleshooting

### Low Confidence Scores
- **Cause**: Poor cropping, low contrast, or background interference
- **Solution**: Recrop more tightly, increase contrast, remove background elements

### False Positives
- **Cause**: Template too generic or similar elements in UI
- **Solution**: Make template more specific, use smaller ROI in code

### No Matches Found
- **Cause**: Template doesn't match current game version
- **Solution**: Recapture template from current game version

## Advanced Template Features

### Multi-Scale Matching
The bot automatically handles different scales, but optimal template size is:
- Small elements (buttons): 30-60px
- Medium elements (icons): 50-100px
- Large elements (panels): 100-200px

### Edge Detection
Templates are processed with Canny edge detection for robust matching across different lighting conditions.

### ORB Feature Matching
For complex templates, the bot falls back to ORB feature matching when template matching fails.