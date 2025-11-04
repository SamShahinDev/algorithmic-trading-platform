#!/usr/bin/env python3
"""
Quick test of pattern discovery on a smaller dataset
"""

import sys
sys.path.append('/Users/royaltyvixion/Documents/XTRADING/trading_bot')

from discover_nq_patterns import NQPatternDiscovery
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test with just 1 month of data first
discovery = NQPatternDiscovery()

# Override dates for testing (just August 2025)
discovery.train_start = datetime(2025, 7, 1)
discovery.train_end = datetime(2025, 8, 15)
discovery.test_start = datetime(2025, 8, 16)
discovery.test_end = datetime(2025, 8, 25)

logger.info("Running test discovery on 1.5 months of data...")
discovery.run_discovery()