# -*- coding: utf-8 -*-
"""
Application entry point for SciSynth AI Research Assistant.
"""

"""Wrapper script to run the Streamlit app with correct Python path."""

import os
import sys
import subprocess

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Run the Streamlit app
subprocess.run(["streamlit", "run", "ui/streamlit_app.py"])
