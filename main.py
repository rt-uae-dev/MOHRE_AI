#!/usr/bin/env python3
"""
MOHRE Document Processing Pipeline
Main Entry Point - Continuous Monitoring Mode
"""

import os
import sys
import time
import signal
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import get_config
from main_pipeline import main as run_full_pipeline

config = get_config()

# Set Google API credentials automatically
google_creds_path = config.google_application_credentials
if google_creds_path and google_creds_path.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(google_creds_path)
    print(f"✅ Google API credentials set: {google_creds_path}")
else:
    print(f"⚠️ Warning: Google API credentials file not found at {google_creds_path}")

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    print(f"\n🛑 Received shutdown signal. Stopping pipeline...")
    running = False

def continuous_pipeline():
    """Run the pipeline continuously, checking for new emails every interval."""
    global running
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configuration
    check_interval = 300  # Check every 5 minutes (300 seconds)
    print(f"🔄 Pipeline will check for new emails every {check_interval} seconds")
    print("🛑 Press Ctrl+C to stop the pipeline")
    print("-" * 60)
    
    cycle_count = 0
    
    while running:
        cycle_count += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🔄 Cycle {cycle_count} - {current_time}")
        print("🚀 Starting MOHRE Document Processing Pipeline...")
        print("📋 Running full pipeline in continuous monitoring mode")
        print("📁 Check the output directory for results when complete.")
        
        try:
            run_full_pipeline()
            print("✅ Pipeline cycle completed successfully!")
        except Exception as e:
            print(f"❌ Pipeline error in cycle {cycle_count}: {e}")
            import traceback
            traceback.print_exc()
        
        if running:
            print(f"⏰ Waiting {check_interval} seconds before next check...")
            print("🛑 Press Ctrl+C to stop the pipeline")
            print("-" * 60)
            
            # Sleep in smaller intervals to allow for graceful shutdown
            for _ in range(check_interval):
                if not running:
                    break
                time.sleep(1)
    
    print("✅ Pipeline stopped gracefully.")

if __name__ == "__main__":
    continuous_pipeline()

