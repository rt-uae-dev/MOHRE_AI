#!/usr/bin/env python3
"""
MOHRE Document Processing Pipeline
Main Entry Point
"""

import sys
import os

# Set Google API credentials automatically
google_creds_path = os.path.join(os.path.dirname(__file__), 'config', 'GOOGLEAPI.json')
if os.path.exists(google_creds_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path
    print(f"✅ Google API credentials set: {google_creds_path}")
else:
    print(f"⚠️ Warning: Google API credentials file not found at {google_creds_path}")

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main_pipeline import main

if __name__ == "__main__":
    from datetime import datetime
    import time

    while True:
        print(f"\n🔄 Starting processing cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main()
        print("⏳ Sleeping for 5 minutes before next cycle...\n")
        time.sleep(5 * 60)