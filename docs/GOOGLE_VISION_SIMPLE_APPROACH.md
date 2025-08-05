# Simple Google Vision Orientation Detection

## Overview

Simple Google Vision API-based orientation detection that just asks one question: **"Does this image need to be rotated?"**

## How It Works

### Simple Approach
```python
def ask_google_vision_if_needs_rotation(image_path: str) -> dict:
    """
    Ask Google Vision: "Does this image need to be rotated?"
    """
    # 1. Send image to Google Vision API
    # 2. Check if text is detected
    # 3. If text detected → No rotation needed
    # 4. If no text detected → Rotate 90°
```

### Simple Logic
- **Text detected** = Image is readable, no rotation needed
- **No text detected** = Image might be sideways, rotate 90°

## Implementation

### Files Updated:
1. `src/google_vision_orientation_detector.py` - Simple Google Vision implementation
2. `src/enhanced_document_processor.py` - Updated to use Google Vision
3. `tests/test_enhanced_processor_with_simple_orientation.py` - Updated tests
4. `tests/test_google_vision_simple.py` - Simple Google Vision test

### Key Functions:
- `ask_google_vision_if_needs_rotation()` - Ask Google Vision one question
- `rotate_if_needed()` - Rotate if Google Vision says yes
- `detect_and_correct_orientation()` - Integrated into main pipeline

### Usage:
```python
from src.google_vision_orientation_detector import rotate_if_needed

# Ask Google Vision if image needs rotation
corrected_image_path = rotate_if_needed("path/to/image.jpg")
```

## Processing Pipeline

```
Step 1: Ask Google Vision "Does this image need rotation?"
Step 2: If yes → Rotate 90°
Step 3: If no → Keep original
Step 4: Continue with YOLO8 + ResNet + Document AI
```

## Benefits

### ✅ **Simple**
- Just one question to Google Vision
- No complex calculations
- Easy to understand

### ✅ **Fast**
- Single API call
- Quick response
- Minimal processing

### ✅ **Reliable**
- Based on Google Vision's text detection
- Handles edge cases automatically
- Consistent results

### ✅ **Integrated**
- Seamless integration with existing pipeline
- Clear metadata tracking
- Easy to debug

## Setup Required

To use this approach, you need:
1. Google Cloud project
2. Google Vision API enabled
3. Service account credentials
4. `GOOGLE_APPLICATION_CREDENTIALS` environment variable set

## Example Output

```
🔍 Asking Google Vision: Does this image need rotation?
   Image: passport_page_1.jpg
📊 Google Vision says:
   Rotation needed: False
   Reason: Text detected and readable
✅ No rotation needed
```

This approach is much simpler than complex AI vision models and just asks Google Vision the essential question: **"Does this image need to be rotated?"** 