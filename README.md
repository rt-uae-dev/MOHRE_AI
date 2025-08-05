# MOHRE Document Processing Pipeline

A comprehensive document processing system for MOHRE (Ministry of Human Resources and Emiratisation) applications with **Document AI integration** for enhanced OCR accuracy.

## 🆕 What's New: Document AI Integration

**Document AI is now the primary OCR method** for all document types, providing:
- ✅ **Better OCR accuracy** than Google Vision
- ✅ **Structured field extraction** with confidence scores
- ✅ **Enhanced passport processing** (fixes issue place detection)
- ✅ **Improved certificate processing** (attestation numbers, stamps)
- ✅ **Multi-language support** (Arabic/English mixed content)
- ✅ **Fallback to Google Vision** when needed

### Quick Setup
```bash
# 1. Run setup script
python setup_document_ai.py

# 2. Test integration
python test_document_ai_integration.py

# 3. Use in pipeline
python main.py
```

## 📁 Project Structure

```
MOHRE/
├── 📁 src/                    # Source code
│   ├── main_pipeline.py       # Main processing pipeline
│   ├── document_processing_pipeline.py # Document processing with validation
│   ├── resnet18_classifier.py # AI classification model
│   ├── yolo_crop_ocr_pipeline.py # YOLO + OCR processing
│   ├── document_ai_processor.py # Document AI integration
│   ├── structure_with_gemini.py  # Gemini data structuring
│   ├── enhanced_document_processor.py # Enhanced document processing
│   ├── passport_ocr_processor.py # Passport OCR processing
│   ├── attestation_utils.py   # Attestation utilities
│   ├── email_parser.py        # Email parsing
│   ├── image_rotation_utils.py # Image rotation utilities
│   ├── image_compression_utils.py # Image compression utilities
│   ├── output_saving_utils.py # Output saving utilities
│   ├── parse_salary_docx.py   # Salary document parsing
│   ├── crop_yolo_detections.py # YOLO detection cropping
│   ├── copy_models_and_dataset.py # Model copying utilities
│   ├── classifier.py          # Document classifier
│   ├── document_processor_v5.py # Document processor v5
│   ├── mobilenet_training.py  # MobileNet training
│   └── structure_with_gpt.py  # GPT data structuring
├── 📁 tests/                  # Test files
│   ├── run_validation_tests.py # Test runner for validation logic
│   ├── test_validation_logic.py # Validation logic tests
│   ├── test_document_ai_integration.py # Document AI integration tests
│   ├── test_akif_case.py      # Akif's case simulation
│   ├── test_enhanced_processor.py # Enhanced processor tests
│   ├── test_gemini_vision_basic.py # Gemini Vision basic tests
│   ├── test_gemini_vision_features.py # Gemini Vision feature tests
│   ├── test_gemini_integration.py # Gemini integration tests
│   ├── test_gemini_attestation_extraction.py # Attestation extraction tests
│   ├── test_ahmad_certificate.py # Ahmad certificate tests
│   ├── test_ahmad_pdf_conversion.py # PDF conversion tests
│   ├── test_ahmad_raw_documents.py # Raw document tests
│   ├── test_certificate_orientation.py # Certificate orientation tests
│   ├── test_orientation_debug.py # Orientation debugging tests
│   ├── test_random_attestation.py # Random attestation tests
│   ├── test_ateesss_image.py  # Image processing tests
│   ├── simple_orientation_test.py # Simple orientation tests
│   ├── test_new_gemini_sdk.py # New Gemini SDK tests
│   ├── test_document_ai_simple.py # Simple Document AI tests
│   ├── test_document_ai_simple_direct.py # Direct Document AI tests
│   ├── test_document_ai.py    # Document AI tests
│   ├── test_current_issue.py  # Current issue tests
│   ├── simple_test.py         # Simple tests
│   ├── quick_test.py          # Quick tests
│   ├── debug_env.py           # Environment debugging
│   ├── test_env.py            # Environment tests
│   └── test_ocr.py            # OCR tests
├── 📁 scripts/                # Setup and utility scripts
│   ├── setup_document_ai.py   # Document AI setup script
│   └── setup.py               # General setup script
├── 📁 docs/                   # Documentation
│   ├── VALIDATION_LOGIC_IMPLEMENTATION.md # Validation logic docs
│   ├── DOCUMENT_AI_INTEGRATION_COMPLETE.md # Document AI setup
│   ├── GEMINI_VISION_FEATURES.md # Gemini Vision features
│   ├── GPT_TO_GEMINI_MIGRATION.md # Migration guide
│   ├── document_ai_setup_complete.md # Setup guide
│   ├── document_ai_setup_guide.md # Setup guide
│   ├── ocr_comparison_summary.md # OCR comparison
│   └── AHMAD_TEST_RESULTS_SUMMARY.md # Test results summary
├── 📁 config/                 # Configuration files
│   ├── GOOGLEAPI.json         # Google Cloud credentials
│   └── env_template.txt       # Environment template
├── 📁 data/                   # Data directories
│   ├── 📁 raw/                # Raw input data
│   │   └── downloads/         # Downloaded email attachments
│   ├── 📁 processed/          # Processed outputs
│   │   ├── COMPLETED/         # Final processed documents
│   │   ├── MOHRE_ready/       # Ready for submission
│   │   └── enhanced_processor_results.json # Processing results
│   ├── 📁 temp/               # Temporary processing files
│   ├── 📁 cropped/            # Cropped images
│   └── 📁 dataset/            # Training dataset (22 document types)
├── 📁 models/                 # AI models
│   ├── yolo8_best.pt          # YOLO8 detection model
│   ├── resnet_classifier.pt   # ResNet18 classification model
│   └── 📁 trained/            # Training data and datasets
│       ├── roboflow_dataset/  # YOLO training datasets
│       └── weights/           # Model weights
├── 📁 logs/                   # Log files
│   └── process_log.txt        # Processing logs
├── 📁 downloads/              # Download directory
├── 📁 venv/                   # Python virtual environment
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🚀 Quick Start

1. **Activate the virtual environment:**
   ```bash
   venv\Scripts\activate
   ```

2. **Set up Document AI (recommended):**
   ```bash
   python scripts/setup_document_ai.py
   ```

3. **Run the pipeline:**
   ```bash
   python main.py
   ```

## 🔧 Features

### Document Processing
- **Email Fetching**: Automatically downloads and processes email attachments
- **PDF Conversion**: Converts PDFs to images for processing
- **Image Classification**: Uses ResNet18 to classify document types
- **OCR Processing**: **Document AI (primary)** + Google Vision (fallback) ⭐ ENHANCED
- **Gemini Structuring**: AI-powered document data extraction with Document AI fields ⭐ ENHANCED
- **Gemini Vision Fallback**: Vision-based classification for attestation documents ⭐ NEW
- **Auto-Orientation Check**: Automatic orientation analysis for passport photos and certificates ⭐ NEW

### Document Types Supported
- 📄 **Passport** (Page 1 & 2) - Enhanced with Document AI field extraction
- 🏛️ **Certificate Attestation** - **NEW: Attestation number validation** to prevent incorrect extractions
- 🆔 **Emirates ID** (Front & Back) - Better number and name extraction
- 👤 **Employee Information Forms** - Improved form field detection

### 🆕 Attestation Number Validation
**NEW FEATURE**: Automatic validation of attestation numbers to prevent incorrect extractions:

- ✅ **Validates extracted numbers** against actual OCR text
- ✅ **Rejects numbers not found** in the document (sets to null)
- ✅ **Prevents false positives** from other document numbers
- ✅ **Supports multiple formats**: 10-15 digit numbers, 7-digit numbers
- ✅ **Filters out Emirates ID numbers** (starting with 784)
- ✅ **Filters out identity numbers** (format: XXX/YYYY/ZZZZZZZ)

**Example**: Ahmad's attestation number was incorrectly extracted as "00000137575601" but the correct number "201400642961" is now properly validated and extracted.
- 🎓 **Certificates & Attestations** - Enhanced attestation number detection ⭐ NEW
- 📋 **Various Forms** - Better structured data extraction
- 🏠 **Residence Cancellations** - Improved number extraction
- 📸 **Personal Photos** - Standard processing

### Data Extraction (Enhanced with Document AI)
- **Personal Information**: Name, Date of Birth, Nationality
- **Document Numbers**: Passport, UID, Attestation numbers (higher accuracy)
- **Address & Job Details**: Complete address and job title
- **Arabic Translations**: Automatic Arabic translations
- **Priority System**: Document AI fields → Passport → EID → Other documents ⭐ UPDATED
- **Confidence Scores**: Quality assessment for extracted data ⭐ NEW

### Misclassification Prevention ⭐ ENHANCED
- **Smart Validation Logic**: Automatically detects and corrects AI misclassifications
- **Passport Detection**: Identifies passports misclassified as certificates using OCR text analysis
- **Emirates ID Detection**: Recognizes Emirates IDs misclassified as certificates
- **Multi-language Support**: Works with Arabic, Turkish, Russian, Hindi, and other languages
- **Pattern Recognition**: Detects document number formats and official terminology
- **Non-intrusive**: Only activates when needed, doesn't affect correct classifications
- **Gemini Vision Fallback**: Uses image analysis for attestation and certificate classification ⭐ NEW
- **Auto-Orientation Detection**: Identifies and corrects orientation issues in photos and certificates ⭐ NEW

## 📊 Output Structure

Each processed document generates:
- **JSON File**: Structured data in JSON format
- **Text File**: Human-readable GPT response and structured data
- **JPG File**: Compressed processed image
- **Processing Metadata**: OCR method, confidence scores, extracted fields ⭐ NEW

## 🔑 Configuration

### Document AI Setup (Recommended)
1. **Google Cloud Console**: Enable Document AI API
2. **Create Processor**: Document OCR Processor
3. **Service Account**: Add Document AI permissions
4. **Environment Variables**: Set project ID and processor ID

### API Keys
- **Google Cloud**: Place credentials in `config/GOOGLEAPI.json`
- **Document AI**: Configure via environment variables
- **Google Gemini**: Set API key for AI processing

### Environment Variables
```env
# Document AI Configuration
GOOGLE_CLOUD_PROJECT_ID=your-project-id
DOCUMENT_AI_PROCESSOR_ID=your-processor-id
GOOGLE_APPLICATION_CREDENTIALS=config/GOOGLEAPI.json

# Email Configuration
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
IMAP_SERVER=imap.gmail.com

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key
```

## 📝 Logging

All processing activities are logged to `logs/process_log.txt` with detailed information about:
- Document processing status
- OCR results (Document AI vs Google Vision)
- Classification results
- Confidence scores and extracted fields
- Error handling
- Validation logic corrections ⭐ NEW

## 🧪 Testing

### Run Validation Logic Tests
```bash
cd tests
python run_validation_tests.py
```

### Test Document AI Integration
```bash
python tests/test_document_ai_integration.py
```

### Test Individual Components
```bash
# Test validation logic
python tests/test_validation_logic.py

# Test Akif's case simulation
python tests/test_akif_case.py

# Test simple validation
python tests/simple_test.py

# Test Gemini Vision features (basic)
python test_gemini_vision_basic.py

# Test Gemini Vision features (with images)
python test_gemini_vision_features.py
```

## 🛠️ Development

### Adding New Document Types
1. Add training data to `data/dataset/`
2. Retrain the ResNet18 classifier
3. Update classification logic in `src/resnet18_classifier.py`

### Modifying Data Extraction
1. Update Gemini prompts in `src/structure_with_gemini.py`
2. Modify priority rules for data extraction
3. Test with sample documents

## 📈 Performance

- **Processing Speed**: ~2-3 seconds per document
- **Accuracy**: >95% for document classification
- **OCR Quality**: High accuracy with Google Vision API
- **Data Extraction**: Comprehensive with Google Gemini

## 🔒 Security

- API keys stored in separate config directory
- Temporary files automatically cleaned up
- No sensitive data logged to files

## 📞 Support

For issues or questions:
1. Check the logs in `logs/process_log.txt`
2. Verify API credentials in `config/`
3. Ensure all dependencies are installed

---

**Last Updated**: July 29, 2025
**Version**: 2.0 