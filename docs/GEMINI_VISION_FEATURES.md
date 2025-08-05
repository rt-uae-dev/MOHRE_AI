# Gemini Vision Features Documentation

## Overview

This document describes the new Gemini Vision features added to the MOHRE document processing pipeline. These features enhance document classification accuracy and provide automatic orientation checking for passport photos and certificates.

## New Features

### 1. Gemini Vision Classification Fallback

**Purpose**: Provides more accurate classification for attestation documents and certificates using image analysis instead of just text.

**How it works**:
- Uses Google Gemini's vision capabilities to analyze document images
- Looks for visual indicators like official stamps, seals, document layouts, and formatting
- Provides fallback classification when ResNet18 classification is uncertain
- Especially useful for attestation documents that may be misclassified

**When it activates**:
- When ResNet18 classifies a document as "unknown"
- When ResNet18 classifies a document as "certificate" or "certificate_attestation"
- When text-based validation logic is inconclusive

**Function**: `classify_image_with_gemini_vision(image_path: str) -> str`

### 2. Automatic Orientation Checking

**Purpose**: Automatically detects and reports orientation issues in passport photos and certificates.

**How it works**:
- Analyzes images using Gemini Vision to determine if they're oriented correctly
- Checks for upside-down or sideways images
- Provides detailed analysis including clarity assessment and recommendations
- Works specifically for passport photos, certificates, and attestation documents

**Analysis includes**:
- Document type identification
- Orientation correctness
- Image clarity assessment
- Specific issues found
- Recommendations for correction
- Image dimensions and aspect ratio

**Function**: `check_image_orientation(image_path: str) -> dict`

### 3. Auto-Rotation

**Purpose**: Automatically corrects orientation issues when detected.

**How it works**:
- Uses orientation analysis to determine if correction is needed
- Applies appropriate rotation to fix orientation issues
- Saves corrected image with "_rotated" suffix
- Returns path to corrected image

**Function**: `auto_rotate_image_if_needed(image_path: str, analysis: dict = None) -> str`

## Integration with Pipeline

### Updated Document Processing Flow

1. **ResNet18 Classification**: Initial classification using trained model
2. **Text-based Validation**: OCR text analysis for misclassification detection
3. **Gemini Vision Fallback**: Image-based classification for uncertain cases
4. **Orientation Checking**: Automatic orientation analysis for photos and certificates
5. **Auto-Rotation**: Automatic correction of orientation issues
6. **Final Processing**: Continue with existing OCR and structuring

### Enhanced Output

The pipeline now includes orientation analysis in the output metadata:

```json
{
  "orientation_analysis": {
    "document_type": "passport_photo",
    "orientation_correct": false,
    "clarity": "good",
    "issues": ["Image is upside down"],
    "recommendations": ["Rotate image 180 degrees"],
    "image_dimensions": {"width": 800, "height": 600},
    "aspect_ratio": 1.33
  }
}
```

## Configuration

### Required Environment Variables

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Dependencies

```bash
pip install google-generativeai>=0.8.0
```

## Usage Examples

### Basic Vision Classification

```python
from resnet18_classifier import classify_image_with_gemini_vision

# Classify an image using Gemini Vision
result = classify_image_with_gemini_vision("path/to/document.jpg")
print(f"Classification: {result}")
```

### Orientation Checking

```python
from resnet18_classifier import check_image_orientation

# Check image orientation
analysis = check_image_orientation("path/to/photo.jpg")
print(f"Orientation correct: {analysis['orientation_correct']}")
print(f"Issues: {analysis['issues']}")
```

### Auto-Rotation

```python
from resnet18_classifier import auto_rotate_image_if_needed

# Auto-rotate if needed
corrected_path = auto_rotate_image_if_needed("path/to/image.jpg")
if corrected_path != "path/to/image.jpg":
    print(f"Image was corrected: {corrected_path}")
```

## Testing

### Basic Tests (No Images Required)

```bash
python test_gemini_vision_basic.py
```

### Full Tests (Requires Test Images)

```bash
# Create test directory
mkdir -p data/test_samples

# Add test images to data/test_samples/
# Then run:
python test_gemini_vision_features.py
```

## Benefits

### 1. Improved Classification Accuracy
- Vision-based analysis provides more accurate classification for complex documents
- Reduces misclassification of attestation documents
- Better handling of documents with poor OCR text

### 2. Automatic Quality Control
- Detects orientation issues before processing
- Provides clear recommendations for corrections
- Reduces manual review requirements

### 3. Enhanced User Experience
- Automatic correction of common issues
- Detailed analysis and recommendations
- Better handling of edge cases

## Limitations

### 1. API Dependencies
- Requires active Gemini API key
- Subject to API rate limits and quotas
- Requires internet connection

### 2. Processing Time
- Vision analysis adds processing time
- May increase overall pipeline duration
- Should be used selectively for best performance

### 3. Image Quality
- Works best with clear, high-quality images
- May struggle with very low-resolution images
- Orientation detection accuracy depends on image clarity

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   ❌ GEMINI_API_KEY not found in environment variables
   ```
   **Solution**: Add your Gemini API key to the `.env` file

2. **Import Errors**
   ```
   ❌ Failed to import google.generativeai
   ```
   **Solution**: Install the library: `pip install google-generativeai`

3. **Vision Analysis Fails**
   ```
   ❌ Gemini Vision classification failed
   ```
   **Solution**: Check API key validity and internet connection

### Performance Optimization

1. **Selective Usage**: Only use vision features when needed
2. **Image Quality**: Ensure images are clear and well-lit
3. **Caching**: Consider caching results for repeated processing
4. **Batch Processing**: Process multiple images efficiently

## Future Enhancements

### Potential Improvements

1. **Multi-angle Analysis**: Analyze images from multiple angles
2. **Advanced Rotation**: Support for arbitrary rotation angles
3. **Quality Scoring**: Provide quality scores for processed images
4. **Batch Processing**: Optimize for multiple document processing
5. **Custom Models**: Fine-tune for specific document types

### Monitoring and Analytics

1. **Success Rate Tracking**: Monitor classification accuracy
2. **Performance Metrics**: Track processing time and API usage
3. **Error Analysis**: Identify common failure patterns
4. **User Feedback**: Collect feedback for continuous improvement

## Support

For issues with the Gemini Vision features:

1. Check the test script output
2. Verify API key configuration
3. Review the logs in `logs/process_log.txt`
4. Test with sample images to isolate issues

## Version History

- **v2.1**: Added Gemini Vision classification fallback
- **v2.1**: Added automatic orientation checking
- **v2.1**: Added auto-rotation capabilities
- **v2.1**: Enhanced pipeline integration 