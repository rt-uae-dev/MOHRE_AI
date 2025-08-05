#!/usr/bin/env python3
"""
Setup script for MOHRE Document Processing Pipeline
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw/downloads",
        "data/processed/COMPLETED", 
        "data/processed/MOHRE_ready",
        "data/temp",
        "logs",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_env_file():
    """Set up environment file if it doesn't exist"""
    if not os.path.exists(".env"):
        if os.path.exists("env_template.txt"):
            shutil.copy("env_template.txt", ".env")
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your actual credentials")
        else:
            print("⚠️  No env_template.txt found. Please create .env file manually")
    else:
        print("✅ .env file already exists")

def check_google_credentials():
    """Check if Google API credentials exist"""
    google_creds_path = os.path.join("config", "GOOGLEAPI.json")
    if os.path.exists(google_creds_path):
        print("✅ Google API credentials found")
    else:
        print("⚠️  Google API credentials not found at config/GOOGLEAPI.json")
        print("   This is required for OCR functionality")

def main():
    print("🚀 Setting up MOHRE Document Processing Pipeline...")
    print("=" * 50)
    
    check_python_version()
    create_directories()
    setup_env_file()
    check_google_credentials()
    
    print("\n📋 Next steps:")
    print("1. Edit .env file with your email credentials")
    print("2. Add Google API credentials to config/GOOGLEAPI.json")
    print("3. Run: python main.py")
    
    print("\n✅ Setup complete!")

if __name__ == "__main__":
    main() 