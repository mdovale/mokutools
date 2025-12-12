"""
Pytest configuration and fixtures for mokutools tests.
"""
import pytest
from pathlib import Path


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

