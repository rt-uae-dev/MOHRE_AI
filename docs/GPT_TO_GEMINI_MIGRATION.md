# GPT to Gemini Migration Guide

## Overview

This document outlines the migration from OpenAI GPT to Google Gemini API for document processing and structuring in the MOHRE project.

## Changes Made

### 1. New Files Created
- `src/structure_with_gemini.py` - Replaces `structure_with_gpt.py`
- `test_gemini_integration.py` - Test script for Gemini integration
- `docs/GPT_TO_GEMINI_MIGRATION.md` - This migration guide

### 2. Files Modified

#### Dependencies
- `requirements.txt` - Added `google-generativeai>=0.3.0`

#### Configuration
- `env_template.txt` - Added `GOOGLE_GEMINI_API_KEY` configuration
- `README.md` - Updated documentation to reflect Gemini usage

#### Source Code
- `src/main_pipeline.py` - Updated imports and function calls
- `src/resnet18_classifier.py` - Replaced OpenAI with Gemini
- `src/document_processing_pipeline.py` - Updated fallback logic
- `src/output_saving_utils.py` - Updated parameter names

### 3. Function Mapping

| GPT Function | Gemini Function | Status |
|--------------|----------------|---------|
| `structure_with_gpt()` | `structure_with_gemini()` | ✅ Migrated |
| `structure_document_with_gpt()` | `structure_document_with_gemini()` | ✅ Migrated |
| `classify_image_from_text()` | `classify_image_from_text()` | ✅ Updated (same name, different backend) |

### 4. API Changes

#### OpenAI GPT (Old)
```python
import openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)
content = response.choices[0].message.content
```

#### Google Gemini (New)
```python
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(prompt)
content = response.text
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install google-generativeai>=0.8.0
```

### 2. Get Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 3. Update Environment Variables
Add to your `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Test Integration
```bash
python test_gemini_integration.py
```

## Benefits of Migration

### 1. Cost Efficiency
- Gemini API is generally more cost-effective than GPT-4
- Better pricing for high-volume document processing

### 2. Performance
- Gemini 1.5 Flash offers fast response times
- Optimized for text processing tasks

### 3. Integration
- Better integration with existing Google Cloud services
- Unified Google ecosystem (Vision, Document AI, Gemini)

### 4. Reliability
- Google's infrastructure provides high availability
- Consistent performance across regions

## Backward Compatibility

### What Still Works
- All existing document processing workflows
- Same function signatures and return types
- Same configuration structure
- Same output formats

### What Changed
- API calls now use Gemini instead of GPT
- Environment variable name changed
- Internal prompt handling (but same prompts work)

## Troubleshooting

### Common Issues

#### 1. API Key Not Found
```
❌ GEMINI_API_KEY not found in environment variables
```
**Solution**: Add the API key to your `.env` file

#### 2. Import Error
```
❌ Failed to import google.generativeai
```
**Solution**: Install the library: `pip install google-generativeai`

#### 3. API Connection Failed
```
❌ Gemini API connection failed
```
**Solution**: Check your API key and internet connection

### Testing
Run the test script to verify everything works:
```bash
python test_gemini_integration.py
```

## Rollback Plan

If you need to revert to GPT:

1. **Restore old files**:
   - Copy `src/structure_with_gpt.py` back
   - Update imports in `src/main_pipeline.py`

2. **Update environment**:
   - Set `OPENAI_API_KEY` instead of `GEMINI_API_KEY`

3. **Revert dependencies**:
   - Remove `google-generativeai` from requirements.txt

## Support

For issues with the migration:
1. Check the test script output
2. Verify API key configuration
3. Review the logs in `logs/process_log.txt`

## Future Enhancements

### Potential Improvements
1. **Multi-modal support**: Gemini can handle images directly
2. **Batch processing**: Optimize for multiple documents
3. **Custom models**: Fine-tune Gemini for specific document types
4. **Caching**: Implement response caching for cost optimization

### Monitoring
- Track API usage and costs
- Monitor response quality
- Compare performance with previous GPT implementation 