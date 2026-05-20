"""
Pytest configuration and fixtures for mokutools tests.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import scipy.io
from pathlib import Path

DELIMITER = ', '


def _build_moku_mat_from_csv_text(csv_text: str, max_data_rows: int = 20) -> np.ndarray:
    """Build a Moku phasemeter .mat 'moku' array from CSV text."""
    lines = csv_text.splitlines(keepends=True)
    header_lines = [line for line in lines if line.startswith('%') or line.startswith('#')]
    data_lines = [
        line for line in lines
        if line.strip() and not (line.startswith('%') or line.startswith('#'))
    ]
    header_str = ''.join(header_lines)
    df = pd.read_csv(
        io.StringIO(''.join(data_lines[:max_data_rows])),
        header=None,
        sep=DELIMITER,
        engine='python',
    )
    moku = np.empty((1, 1), dtype=object)
    inner = np.empty((1, 2), dtype=object)
    inner[0, 0] = np.array(header_str)
    inner[0, 1] = df.values
    moku[0, 0] = inner
    return moku


@pytest.fixture
def moku_mat_zip(tmp_path):
    """Zip archive containing a single Moku-style .mat file built from the CSV fixture."""
    csv_fixture = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
    if not csv_fixture.exists():
        pytest.skip(f"Test file not found: {csv_fixture}")

    with zipfile.ZipFile(csv_fixture) as zf:
        csv_text = zf.read(zf.namelist()[0]).decode('utf-8')

    moku = _build_moku_mat_from_csv_text(csv_text)
    mat_path = tmp_path / "MokuPhasemeterData.mat"
    scipy.io.savemat(str(mat_path), {'moku': moku}, format='5')

    zip_path = tmp_path / "MokuPhasemeterData.mat.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(mat_path, arcname=mat_path.name)
    return zip_path


@pytest.fixture(scope="session", autouse=True)
def cleanup_csv_files():
    """
    Automatically clean up CSV files created during tests.
    This fixture runs once at the end of the test session.
    """
    yield
    
    # Get the project root directory (parent of tests directory)
    project_root = Path(__file__).parent.parent
    
    # Clean up specific test CSV files
    test_csv_files = [
        "test_file.csv",
    ]
    
    # Clean up CSV files in the project root
    for csv_file in test_csv_files:
        csv_path = project_root / csv_file
        if csv_path.exists():
            try:
                csv_path.unlink()
            except Exception:
                pass  # Ignore errors during cleanup
    
    # Also clean up any CSV files matching test patterns in the project root
    # This catches files like test_*.csv that might be created during tests
    for csv_path in project_root.glob("test_*.csv"):
        try:
            csv_path.unlink()
        except Exception:
            pass  # Ignore errors during cleanup

