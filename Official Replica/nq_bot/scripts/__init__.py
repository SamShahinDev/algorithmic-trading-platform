"""
NQ Bot Scripts Package
Common path setup for all scripts in this directory
"""

import sys
import os

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
nq_bot_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(nq_bot_dir)

# Insert paths to allow imports from various locations
sys.path.insert(0, nq_bot_dir)  # For accessing nq_bot modules
sys.path.insert(0, project_root)  # For accessing root-level modules

# Common imports that scripts might need
__all__ = [
    'script_dir',
    'nq_bot_dir', 
    'project_root'
]