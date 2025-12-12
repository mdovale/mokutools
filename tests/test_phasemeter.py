"""
Unit tests for mokutools.phasemeter module.
"""
import pytest
import numpy as np
import pandas as pd
import zipfile
import os
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mokutools.phasemeter import (
    MokuPhasemeterObject,
    parse_header,
    NCOLS_PER_CHANNEL,
    DELIMITER,
)


class TestMokuPhasemeterObject:
    """Tests for MokuPhasemeterObject class."""

    def test_init_with_zip_file(self):
        """Test initialization with a zip file containing CSV data."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Basic attributes should be set
        assert obj.filename == str(test_file)
        assert obj.fs is not None
        assert obj.fs > 0
        assert obj.date is not None
        assert obj.nchan > 0
        assert len(obj.labels) > 0
        assert obj.df is not None
        assert len(obj.df) > 0

    def test_init_with_time_slicing(self):
        """Test initialization with time slicing parameters."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        # Load full file first to get duration
        obj_full = MokuPhasemeterObject(filename=str(test_file))
        full_duration = obj_full.duration
        
        # Load a subset
        start_time = 0.1
        duration = 0.5
        obj = MokuPhasemeterObject(
            filename=str(test_file),
            start_time=start_time,
            duration=duration
        )
        
        assert obj.start_time == start_time
        assert obj.duration <= duration
        assert obj.duration > 0
        assert len(obj.df) <= int(duration * obj.fs)
        assert len(obj.df) > 0

    def test_init_with_prefix(self):
        """Test initialization with column prefix."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        prefix = "test_"
        obj = MokuPhasemeterObject(filename=str(test_file), prefix=prefix)
        
        # Check that columns have the prefix
        for col in obj.df.columns:
            assert col.startswith(prefix)

    def test_init_with_spectrums(self):
        """Test initialization with precomputed spectrums."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(
            filename=str(test_file),
            spectrums=['phase']
        )
        
        assert len(obj.ps) > 0
        # Check that phase spectrum exists for at least one channel
        phase_keys = [k for k in obj.ps.keys() if 'phase' in k]
        assert len(phase_keys) > 0

    def test_data_labels(self):
        """Test data_labels method."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        labels = obj.data_labels()
        
        # Should start with 'time'
        assert labels[0] == 'time'
        
        # Should have NCOLS_PER_CHANNEL columns per channel
        expected_cols = 1 + obj.nchan * NCOLS_PER_CHANNEL
        assert len(labels) == expected_cols
        
        # Check channel labels format
        for i in range(1, obj.nchan + 1):
            assert f'{i}_set_freq' in labels
            assert f'{i}_freq' in labels
            assert f'{i}_cycles' in labels
            assert f'{i}_i' in labels
            assert f'{i}_q' in labels

    def test_derived_quantities(self):
        """Test that derived quantities (phase, freq2phase) are computed."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Check derived quantities exist
        for i in range(1, obj.nchan + 1):
            phase_col = f'{i}_phase'
            freq2phase_col = f'{i}_freq2phase'
            
            assert phase_col in obj.df.columns
            assert freq2phase_col in obj.df.columns
            
            # Phase should be cycles * 2 * pi
            if f'{i}_cycles' in obj.df.columns:
                expected_phase = obj.df[f'{i}_cycles'] * 2 * np.pi
                np.testing.assert_array_almost_equal(
                    obj.df[phase_col],
                    expected_phase,
                    decimal=10
                )

    def test_spectrum_method_phase(self):
        """Test spectrum method for phase data."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Compute phase spectrum
        obj.spectrum('phase')
        
        # Check that phase spectra exist
        phase_keys = [k for k in obj.ps.keys() if 'phase' in k and not 'freq2phase' in k]
        assert len(phase_keys) > 0
        
        # Check spectrum structure (should be a valid spectrum object)
        for key in phase_keys:
            spec = obj.ps[key]
            # Verify spectrum object exists and is not None
            assert spec is not None
            # Check that it's a valid object (has attributes)
            assert hasattr(spec, '__class__')

    def test_spectrum_method_frequency(self):
        """Test spectrum method for frequency data."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Compute frequency spectrum
        obj.spectrum('frequency')
        
        # Check that frequency spectra exist
        freq_keys = [k for k in obj.ps.keys() if 'freq' in k and 'freq2phase' not in k]
        assert len(freq_keys) > 0

    def test_spectrum_method_freq2phase(self):
        """Test spectrum method for freq2phase data."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Compute freq2phase spectrum
        obj.spectrum('freq2phase')
        
        # Check that freq2phase spectra exist
        f2p_keys = [k for k in obj.ps.keys() if 'freq2phase' in k]
        assert len(f2p_keys) > 0

    def test_spectrum_method_multiple_channels(self):
        """Test spectrum method with specific channels."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Compute spectrum for first channel only
        if obj.nchan >= 1:
            obj.spectrum('phase', channels=[1])
            
            # Should have spectrum for channel 1
            assert '1_phase' in obj.ps
            
            # Should not have spectrum for other channels if nchan > 1
            if obj.nchan > 1:
                assert '2_phase' not in obj.ps

    def test_spectrum_method_single_column(self):
        """Test spectrum method with exact column name."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Use exact column name
        if '1_freq' in obj.df.columns:
            obj.spectrum('1_freq')
            assert '1_freq' in obj.ps

    def test_spectrum_method_invalid_channel(self):
        """Test spectrum method with invalid channel number."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Try with invalid channel
        invalid_channel = obj.nchan + 10
        with pytest.raises(ValueError, match="not present"):
            obj.spectrum('phase', channels=[invalid_channel])

    def test_init_missing_filename(self):
        """Test that initialization raises error when filename is missing."""
        with pytest.raises(ValueError, match="must be specified"):
            MokuPhasemeterObject()

    def test_init_with_logger(self):
        """Test initialization with custom logger."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        logger = logging.getLogger("test_logger")
        obj = MokuPhasemeterObject(filename=str(test_file), logger=logger)
        
        assert obj is not None
        assert obj.df is not None

    def test_dataframe_structure(self):
        """Test that the loaded dataframe has correct structure."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Check dataframe is not empty
        assert len(obj.df) > 0
        
        # Check that time column exists
        assert 'time' in obj.df.columns
        
        # Check that time is monotonically increasing
        time_diff = np.diff(obj.df['time'].values)
        assert np.all(time_diff >= 0) or np.allclose(time_diff, 1/obj.fs, rtol=1e-3)

    def test_nchan_calculation(self):
        """Test that number of channels is calculated correctly."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # nchan should be calculated from ncols
        expected_nchan = (obj.ncols - 1) // NCOLS_PER_CHANNEL
        assert obj.nchan == expected_nchan

    def test_spectrum_multiple_types(self):
        """Test spectrum method with multiple spectrum types."""
        test_file = Path(__file__).parent / "MokuPhasemeterData_MokuProTest.csv.zip"
        
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        obj = MokuPhasemeterObject(filename=str(test_file))
        
        # Compute multiple spectrum types
        obj.spectrum(['phase', 'frequency'])
        
        # Check both types exist
        phase_keys = [k for k in obj.ps.keys() if 'phase' in k and 'freq2phase' not in k]
        freq_keys = [k for k in obj.ps.keys() if 'freq' in k and 'freq2phase' not in k]
        
        assert len(phase_keys) > 0
        assert len(freq_keys) > 0


class TestParseHeader:
    """Tests for parse_header function."""

    def test_parse_header_with_rate_hint(self):
        """Test parsing header with rate hint."""
        header = [
            "% Some header line",
            "% Acquisition rate: 1000.0 Hz",
            "% Acquired 2024-01-01 12:00:00",
        ]
        
        fs, date = parse_header(header)
        
        assert fs == 1000.0
        assert date is not None

    def test_parse_header_with_custom_hints(self):
        """Test parsing header with custom hints."""
        header = [
            "% Some header line",
            "% Sample rate: 500.0 Hz",
            "% Started 2024-01-01 12:00:00",
        ]
        
        fs, date = parse_header(header, fs_hint="Sample rate", t0_hint="Started")
        
        assert fs == 500.0
        assert date is not None

    def test_parse_header_with_row_numbers(self):
        """Test parsing header with explicit row numbers."""
        header = [
            "% Some header line",
            "% Acquisition rate: 2000.0 Hz",
            "% Acquired 2024-01-01 12:00:00",
        ]
        
        fs, date = parse_header(header, row_fs=2, row_t0=3)
        
        assert fs == 2000.0
        assert date is not None

    def test_parse_header_missing_rate(self):
        """Test parsing header when rate is missing."""
        header = [
            "% Some header line",
            "% Acquired 2024-01-01 12:00:00",
        ]
        
        fs, date = parse_header(header)
        
        assert fs is None
        assert date is not None

    def test_parse_header_missing_date(self):
        """Test parsing header when date is missing."""
        header = [
            "% Some header line",
            "% Acquisition rate: 1000.0 Hz",
        ]
        
        fs, date = parse_header(header)
        
        assert fs == 1000.0
        assert date is None

    def test_parse_header_with_logger(self):
        """Test parse_header with custom logger."""
        header = [
            "% Acquisition rate: 1000.0 Hz",
            "% Acquired 2024-01-01 12:00:00",
        ]
        
        logger = logging.getLogger("test_logger")
        fs, date = parse_header(header, logger=logger)
        
        assert fs == 1000.0
        assert date is not None

    def test_parse_header_empty(self):
        """Test parsing empty header."""
        header = []
        
        fs, date = parse_header(header)
        
        assert fs is None
        assert date is None

    def test_parse_header_malformed_rate(self):
        """Test parsing header with malformed rate line."""
        header = [
            "% Acquisition rate: invalid Hz",
            "% Acquired 2024-01-01 12:00:00",
        ]
        
        fs, date = parse_header(header)
        
        # Should handle gracefully and return None
        assert fs is None or isinstance(fs, float)
        assert date is not None

