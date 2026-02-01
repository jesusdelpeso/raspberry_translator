#!/usr/bin/env python3
"""
Convenience script to run the translator from the scripts folder
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
