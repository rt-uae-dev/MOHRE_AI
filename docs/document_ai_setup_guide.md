# Google Document AI Setup Guide

## 🎯 Why Document AI is Better for Passports

Google Document AI is specifically designed for structured document processing and offers several advantages over Google Vision for passport processing:

### **Key Advantages:**
1. **Pre-trained Models**: Document AI has specialized processors for identity documents
2. **Structured Data Extraction**: Automatically extracts fields like "Place of Issue", "Passport Number", "Date of Birth"
3. **Higher Accuracy**: Better at understanding document layouts and field relationships
4. **Entity Recognition**: Identifies specific document fields with confidence scores
5. **Multi-language Support**: Handles Arabic, Hindi, and other languages common in passports

### **Available Processors for Passports:**
- **Identity Document Processor**: Extracts data from passports, IDs, driver's licenses
- **Form Parser**: For structured forms
- **Document OCR**: General document text extraction

## 🔧 Setup Steps

### 1. Enable Document AI API
```bash
# In Google Cloud Console:
# 1. Go to APIs & Services > Library
# 2. Search for "Document AI API"
# 3. Click "Enable"
```

### 2. Create a Document AI Processor
```bash
# In Google Cloud Console:
# 1. Go to Document AI > Processors
# 2. Click "Create Processor"
# 3. Choose "Identity Document Processor"
# 4. Note the Processor ID (you'll need this)
```

### 3. Set Environment Variables
Add to your `.env` file:
```env
GOOGLE_CLOUD_PROJECT_ID=your-project-id
DOCUMENT_AI_PROCESSOR_ID=your-processor-id
```

### 4. Install Dependencies
```bash
pip install google-cloud-documentai
```

## 📊 Expected Results

Document AI should extract structured data like:
- **Place of Issue**: "DUBAI" (with high confidence)
- **Passport Number**: "25547821"
- **Date of Birth**: "27/11/1979"
- **Expiry Date**: "31/07/2051"
- **Full Name**: "YOGESHKUMAR ASHOKBHAI SANT"

## 🔄 Integration with Current Pipeline

The Document AI results can be integrated into your existing pipeline by:

1. **Replacing Google Vision OCR** with Document AI for passport processing
2. **Using Document AI entities** instead of GPT parsing for structured fields
3. **Falling back to Google Vision** if Document AI fails
4. **Combining both approaches** for maximum accuracy

## 💰 Cost Considerations

- **Document AI**: ~$1.50 per 1,000 pages
- **Google Vision**: ~$1.50 per 1,000 images
- **Both are similar in cost** but Document AI provides better structured data

## 🚀 Next Steps

1. Set up Document AI processor in Google Cloud Console
2. Test with passport images
3. Compare results with current Google Vision approach
4. Integrate into main pipeline if results are better 