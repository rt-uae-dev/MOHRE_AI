#!/usr/bin/env python3
"""
MOHRE Document Processing Pipeline
Main Entry Point
"""

import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import get_config
from gui_app import run_gui

config = get_config()

# Set Google API credentials automatically
google_creds_path = config.google_application_credentials
if google_creds_path and google_creds_path.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(google_creds_path)
    print(f"✅ Google API credentials set: {google_creds_path}")
else:
    print(f"⚠️ Warning: Google API credentials file not found at {google_creds_path}")

from mohre_ai.gui_app import run_gui
if __name__ == "__main__":
    run_gui()
