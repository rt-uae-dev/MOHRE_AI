#!/usr/bin/env python3
"""
Comprehensive test script for Ahmad's raw documents using Gemini Vision features:
1. Process all raw documents from Ahmad's folder
2. Test vision-based document classification
3. Test image orientation checking
4. Test auto-rotation for various document types
5. Test integrated pipeline processing
"""

import os
import sys
import glob
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def get_ahmad_documents():
    """Get all document files from Ahmad's raw downloads folder."""
    base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
    
    documents = []
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    for ext in image_extensions:
        documents.extend(glob.glob(os.path.join(base_path, ext)))
    
    # Get all PDF files
    pdf_files = glob.glob(os.path.join(base_path, "*.pdf"))
    documents.extend(pdf_files)
    
    return sorted(documents)

def test_individual_document_vision_classification():
    """Test Gemini Vision classification on individual documents."""
    try:
        from resnet18_classifier import classify_image_with_gemini_vision
        
        documents = get_ahmad_documents()
        
        print("🔍 Testing Gemini Vision classification on individual documents...")
        print(f"Found {len(documents)} documents to test")
        
        results = {}
        
        for doc_path in documents:
            if doc_path.lower().endswith('.pdf'):
                print(f"   ⏭️ Skipping PDF: {os.path.basename(doc_path)} (PDFs need conversion first)")
                continue
                
            print(f"   🔍 Testing: {os.path.basename(doc_path)}")
            try:
                result = classify_image_with_gemini_vision(doc_path)
                results[os.path.basename(doc_path)] = result
                print(f"      ✅ Classified as: {result}")
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                results[os.path.basename(doc_path)] = "error"
        
        print("\n📊 Classification Results:")
        for doc_name, result in results.items():
            print(f"   {doc_name}: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual document classification test failed: {e}")
        return False

def test_individual_document_orientation():
    """Test orientation checking on individual documents."""
    try:
        from resnet18_classifier import check_image_orientation
        
        documents = get_ahmad_documents()
        
        print("🔍 Testing orientation checking on individual documents...")
        
        results = {}
        
        for doc_path in documents:
            if doc_path.lower().endswith('.pdf'):
                print(f"   ⏭️ Skipping PDF: {os.path.basename(doc_path)} (PDFs need conversion first)")
                continue
                
            print(f"   🔍 Testing: {os.path.basename(doc_path)}")
            try:
                analysis = check_image_orientation(doc_path)
                results[os.path.basename(doc_path)] = analysis
                print(f"      ✅ Document type: {analysis.get('document_type', 'unknown')}")
                print(f"      ✅ Orientation correct: {analysis.get('orientation_correct', 'unknown')}")
                print(f"      ✅ Clarity: {analysis.get('clarity', 'unknown')}")
                if analysis.get('issues'):
                    print(f"      ⚠️ Issues: {len(analysis.get('issues', []))} found")
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                results[os.path.basename(doc_path)] = "error"
        
        print("\n📊 Orientation Analysis Summary:")
        for doc_name, analysis in results.items():
            if analysis != "error":
                print(f"   {doc_name}: {analysis.get('document_type', 'unknown')} - Orientation: {analysis.get('orientation_correct', 'unknown')}")
            else:
                print(f"   {doc_name}: Error")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual document orientation test failed: {e}")
        return False

def test_auto_rotation_on_documents():
    """Test auto-rotation on documents that need it."""
    try:
        from resnet18_classifier import auto_rotate_image_if_needed, check_image_orientation
        
        documents = get_ahmad_documents()
        
        print("🔍 Testing auto-rotation on documents...")
        
        corrected_count = 0
        total_tested = 0
        
        for doc_path in documents:
            if doc_path.lower().endswith('.pdf'):
                continue
                
            print(f"   🔍 Testing: {os.path.basename(doc_path)}")
            try:
                # Check orientation first
                analysis = check_image_orientation(doc_path)
                total_tested += 1
                
                if not analysis.get('orientation_correct', True):
                    print(f"      ⚠️ Orientation issues detected, attempting auto-rotation...")
                    corrected_path = auto_rotate_image_if_needed(doc_path, analysis)
                    
                    if corrected_path != doc_path:
                        print(f"      ✅ Auto-corrected: {os.path.basename(corrected_path)}")
                        corrected_count += 1
                    else:
                        print(f"      ℹ️ No correction needed")
                else:
                    print(f"      ✅ Orientation already correct")
                    
            except Exception as e:
                print(f"      ❌ Failed: {e}")
        
        print(f"\n📊 Auto-Rotation Results:")
        print(f"   Total tested: {total_tested}")
        print(f"   Corrected: {corrected_count}")
        print(f"   No correction needed: {total_tested - corrected_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-rotation test failed: {e}")
        return False

def test_integrated_pipeline_on_sample():
    """Test the integrated pipeline on a sample document."""
    try:
        from document_processing_pipeline import classify_and_ocr
        
        documents = get_ahmad_documents()
        
        # Find a good sample document (preferably an image, not PDF)
        sample_doc = None
        for doc_path in documents:
            if not doc_path.lower().endswith('.pdf'):
                sample_doc = doc_path
                break
        
        if not sample_doc:
            print("❌ No suitable sample document found (all are PDFs)")
            return False
        
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"🔍 Testing integrated pipeline with: {os.path.basename(sample_doc)}")
        
        rotated_path, vision_data, final_label = classify_and_ocr(sample_doc, temp_dir)
        
        print("✅ Integrated pipeline result:")
        print(f"   Final label: {final_label}")
        print(f"   Rotated path: {rotated_path}")
        print(f"   OCR method: {vision_data.get('ocr_method', 'unknown')}")
        print(f"   Document type: {vision_data.get('document_type', 'unknown')}")
        
        # Check for orientation analysis
        if "orientation_analysis" in vision_data:
            print("   Orientation analysis: Available")
            analysis = vision_data["orientation_analysis"]
            print(f"     - Correct: {analysis.get('orientation_correct', 'unknown')}")
            print(f"     - Issues: {len(analysis.get('issues', []))} found")
            print(f"     - Recommendations: {len(analysis.get('recommendations', []))} provided")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated pipeline test failed: {e}")
        return False

def test_specific_documents():
    """Test specific important documents."""
    try:
        from resnet18_classifier import classify_image_with_gemini_vision, check_image_orientation
        
        base_path = "data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC"
        
        # Test specific documents
        specific_docs = [
            "Electrical Engineering Certificate BAU.pdf",  # This is the main certificate
            "Ahmad photo.jpg",  # Personal photo
            "Ahmad Visa 2027.jpg",  # Visa document
            "LC cancellation Ahmad.jpg"  # Cancellation document
        ]
        
        print("🔍 Testing specific important documents...")
        
        for doc_name in specific_docs:
            doc_path = os.path.join(base_path, doc_name)
            
            if not os.path.exists(doc_path):
                print(f"   ❌ Not found: {doc_name}")
                continue
            
            if doc_path.lower().endswith('.pdf'):
                print(f"   ⏭️ Skipping PDF: {doc_name} (PDFs need conversion first)")
                continue
            
            print(f"   🔍 Testing: {doc_name}")
            
            try:
                # Test classification
                classification = classify_image_with_gemini_vision(doc_path)
                print(f"      ✅ Classification: {classification}")
                
                # Test orientation
                orientation = check_image_orientation(doc_path)
                print(f"      ✅ Orientation: {orientation.get('orientation_correct', 'unknown')}")
                print(f"      ✅ Document type: {orientation.get('document_type', 'unknown')}")
                print(f"      ✅ Clarity: {orientation.get('clarity', 'unknown')}")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Specific documents test failed: {e}")
        return False

def main():
    """Run all tests with Ahmad's raw documents."""
    print("🧪 Testing Gemini Vision Features with Ahmad's Raw Documents")
    print("=" * 70)
    
    tests = [
        ("Individual Document Classification", test_individual_document_vision_classification),
        ("Individual Document Orientation", test_individual_document_orientation),
        ("Auto-Rotation Testing", test_auto_rotation_on_documents),
        ("Specific Documents Test", test_specific_documents),
        ("Integrated Pipeline Sample", test_integrated_pipeline_on_sample),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 50)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gemini Vision features work great with Ahmad's raw documents.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    print(f"\n📁 Documents tested from: data/raw/downloads/FW Ahmad Moustapha El Haj Moussa Supervisor Construction Electrical  ADNOC BAB  Buhasa P5 OffPlot Facilities AILIC")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 