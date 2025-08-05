#!/usr/bin/env python3
"""
Script to copy trained models and dataset to MOHRE folder
"""

import os
import shutil
import sys

def copy_models_and_dataset():
    """Copy trained models and dataset to MOHRE folder."""
    
    print("📁 Copying Models and Dataset to MOHRE Folder")
    print("=" * 60)
    
    # Source paths
    source_yolo_model = "C:/Users/Husai/Desktop/mohre-email-parser/runs/detect/train3/weights/best.pt"
    source_classifier_model = "C:/Users/Husai/Desktop/mohre-email-parser/classifier.pt"
    source_dataset = "C:/Users/Husai/Desktop/mohre-email-parser/roboflow_dataset"
    
    # Destination paths in MOHRE folder
    dest_models_dir = "models"
    dest_dataset_dir = "data/dataset"
    
    # Create destination directories
    os.makedirs(dest_models_dir, exist_ok=True)
    os.makedirs(dest_dataset_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # Copy YOLO model
    total_count += 1
    dest_yolo_model = os.path.join(dest_models_dir, "yolo8_best.pt")
    try:
        if os.path.exists(source_yolo_model):
            shutil.copy2(source_yolo_model, dest_yolo_model)
            print(f"✅ YOLO8 model copied to: {dest_yolo_model}")
            success_count += 1
        else:
            print(f"❌ YOLO8 model not found at: {source_yolo_model}")
    except Exception as e:
        print(f"❌ Failed to copy YOLO8 model: {e}")
    
    # Copy classifier model
    total_count += 1
    dest_classifier_model = os.path.join(dest_models_dir, "resnet_classifier.pt")
    try:
        if os.path.exists(source_classifier_model):
            shutil.copy2(source_classifier_model, dest_classifier_model)
            print(f"✅ ResNet classifier copied to: {dest_classifier_model}")
            success_count += 1
        else:
            print(f"❌ ResNet classifier not found at: {source_classifier_model}")
    except Exception as e:
        print(f"❌ Failed to copy ResNet classifier: {e}")
    
    # Copy dataset
    total_count += 1
    dest_dataset = os.path.join(dest_dataset_dir, "certificate_attestation")
    try:
        if os.path.exists(source_dataset):
            if os.path.exists(dest_dataset):
                shutil.rmtree(dest_dataset)  # Remove existing directory
            shutil.copytree(source_dataset, dest_dataset)
            print(f"✅ Dataset copied to: {dest_dataset}")
            success_count += 1
        else:
            print(f"❌ Dataset not found at: {source_dataset}")
    except Exception as e:
        print(f"❌ Failed to copy dataset: {e}")
    
    # Update the enhanced document processor to use local paths
    update_model_paths()
    
    print(f"\n📊 Copy Results: {success_count}/{total_count} items copied successfully")
    
    if success_count == total_count:
        print("🎉 All files copied successfully!")
        print("\n📁 New file structure:")
        print(f"   models/yolo8_best.pt")
        print(f"   models/resnet_classifier.pt")
        print(f"   data/dataset/certificate_attestation/")
    else:
        print("⚠️ Some files failed to copy. Please check the errors above.")
    
    return success_count == total_count

def update_model_paths():
    """Update the enhanced document processor to use local model paths."""
    try:
        processor_file = "src/enhanced_document_processor.py"
        
        if os.path.exists(processor_file):
            # Read the file
            with open(processor_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update YOLO model path
            content = content.replace(
                'model_path = "C:/Users/Husai/Desktop/mohre-email-parser/runs/detect/train3/weights/best.pt"',
                'model_path = "models/yolo8_best.pt"'
            )
            
            # Update classifier model path
            content = content.replace(
                'model_path = "C:/Users/Husai/Desktop/mohre-email-parser/classifier.pt"',
                'model_path = "models/resnet_classifier.pt"'
            )
            
            # Write back to file
            with open(processor_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Updated model paths in enhanced_document_processor.py")
            
    except Exception as e:
        print(f"❌ Failed to update model paths: {e}")

def main():
    """Main function to copy models and dataset."""
    success = copy_models_and_dataset()
    
    if success:
        print("\n🚀 Ready to use enhanced document processor with local models!")
        print("   Run: python test_enhanced_processor.py")
    else:
        print("\n❌ Some files failed to copy. Please check the paths and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 