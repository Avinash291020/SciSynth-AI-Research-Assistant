"""Test configuration for SciSynth AI Research Assistant."""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add the app directory to Python path
app_dir = project_root / "app"
sys.path.insert(0, str(app_dir))

# Add other module directories
agents_dir = project_root / "agents"
if agents_dir.exists():
    sys.path.insert(0, str(agents_dir))

evolutionary_dir = project_root / "evolutionary"
if evolutionary_dir.exists():
    sys.path.insert(0, str(evolutionary_dir))

logic_dir = project_root / "logic"
if logic_dir.exists():
    sys.path.insert(0, str(logic_dir))

utils_dir = project_root / "utils"
if utils_dir.exists():
    sys.path.insert(0, str(utils_dir))

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_text():
    return """
    Recent studies have shown that blue light exposure affects plant growth rates
    significantly. The mechanism appears to be related to photosynthetic efficiency
    and circadian rhythm regulation. Temperature and humidity also play key roles
    in this process.
    """

@pytest.fixture
def sample_insights():
    return """
    1. Blue light increases plant growth rate
    2. Photosynthetic efficiency is affected by light wavelength
    3. Circadian rhythms are regulated by light exposure
    4. Temperature and humidity influence plant responses
    """

@pytest.fixture
def sample_dataset():
    data = {
        'light_intensity': np.random.uniform(0, 100, 100),
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.uniform(40, 80, 100),
        'growth_rate': np.random.uniform(0, 1, 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    # Create a minimal PDF file for testing
    with open(pdf_path, 'wb') as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\n0000000000 65535 f\ntrailer\n<< >>\nstartxref\n0\n%%EOF")
    return pdf_path 