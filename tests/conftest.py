import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from logger import configure_logging

configure_logging()
