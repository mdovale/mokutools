"""
Unit tests for mokutools.filetools module.
"""
import pytest
import requests
import zipfile
import tarfile
import gzip
from io import BytesIO
from unittest.mock import Mock, patch, mock_open, MagicMock
import pandas as pd
import numpy as np
from py7zr import SevenZipFile

from mokutools.filetools import (
    get_file_list,
    download_files,
    parse_csv_file,
    get_columns_with_nans,
)


class TestGetFileList:
    """Tests for get_file_list function."""

    @patch('mokutools.filetools.requests.get')
    def test_get_file_list_success(self, mock_get):
        """Test successful file list retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": ["file1.li", "file2.li", "file3.li"]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_file_list("10.128.100.198")

        assert result == ["file1.li", "file2.li", "file3.li"]
        mock_get.assert_called_once_with("http://10.128.100.198/api/ssd/list")
        mock_response.raise_for_status.assert_called_once()

    @patch('mokutools.filetools.requests.get')
    def test_get_file_list_with_filter(self, mock_get):
        """Test file list retrieval with filter."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": ["test_file1.li", "test_file2.li", "other_file.li", "TEST_file3.li"]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_file_list("10.128.100.198", filter=["test"])

        assert result == ["test_file1.li", "test_file2.li", "TEST_file3.li"]
        assert "other_file.li" not in result

    @patch('mokutools.filetools.requests.get')
    def test_get_file_list_with_multiple_filters(self, mock_get):
        """Test file list retrieval with multiple filters (AND logic)."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": ["test_file1.li", "test_file2.li", "other_test.li", "file1.li"]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_file_list("10.128.100.198", filter=["test", "file"])

        assert result == ["test_file1.li", "test_file2.li"]
        assert "other_test.li" not in result
        assert "file1.li" not in result

    @patch('mokutools.filetools.requests.get')
    def test_get_file_list_empty_data(self, mock_get):
        """Test file list retrieval with empty data."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_file_list("10.128.100.198")

        assert result == []

    @patch('mokutools.filetools.requests.get')
    def test_get_file_list_missing_data_key(self, mock_get):
        """Test file list retrieval when data key is missing."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_file_list("10.128.100.198")

        assert result == []

    @patch('mokutools.filetools.requests.get')
    def test_get_file_list_http_error(self, mock_get):
        """Test file list retrieval with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            get_file_list("10.128.100.198")


class TestDownloadFiles:
    """Tests for download_files function."""

    @patch('mokutools.filetools.shutil.which')
    @patch('mokutools.filetools.get_file_list')
    @patch('mokutools.filetools.requests.get')
    @patch('mokutools.filetools.subprocess.run')
    @patch('mokutools.filetools.os.makedirs')
    @patch('mokutools.filetools.os.remove')
    @patch('mokutools.filetools.shutil.move')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_files_basic(
        self,
        mock_file,
        mock_move,
        mock_remove,
        mock_makedirs,
        mock_subprocess,
        mock_get,
        mock_get_file_list,
        mock_which,
    ):
        """Test basic file download without conversion."""
        mock_which.return_value = None  # mokucli not found, but convert=False
        mock_get_file_list.return_value = ["test_file.li"]
        
        # Mock download response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content.return_value = [b"file content"]
        mock_get.return_value.__enter__.return_value = mock_response
        mock_get.return_value.__exit__ = Mock(return_value=None)

        download_files("10.128.100.198", file_names="test", convert=False, archive=False)

        mock_get_file_list.assert_called_once_with("10.128.100.198")
        mock_makedirs.assert_called_once()
        mock_get.assert_called_once_with("http://10.128.100.198/api/ssd/download/test_file.li", stream=True)

    @patch('mokutools.filetools.shutil.which')
    @patch('mokutools.filetools.get_file_list')
    @patch('mokutools.filetools.requests.get')
    @patch('mokutools.filetools.subprocess.run')
    @patch('mokutools.filetools.zipfile.ZipFile')
    @patch('mokutools.filetools.os.makedirs')
    @patch('mokutools.filetools.os.remove')
    @patch('mokutools.filetools.shutil.move')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_files_with_conversion_and_archive(
        self,
        mock_file,
        mock_move,
        mock_remove,
        mock_makedirs,
        mock_zipfile,
        mock_subprocess,
        mock_get,
        mock_get_file_list,
        mock_which,
    ):
        """Test file download with conversion and archiving."""
        mock_which.return_value = "/usr/bin/mokucli"
        mock_get_file_list.return_value = ["test_file.li"]
        
        # Mock download response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content.return_value = [b"file content"]
        mock_get.return_value.__enter__.return_value = mock_response
        mock_get.return_value.__exit__ = Mock(return_value=None)

        # Mock zipfile
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        mock_zipfile.return_value.__exit__ = Mock(return_value=None)

        download_files("10.128.100.198", file_names="test", convert=True, archive=True)

        mock_subprocess.run.assert_called_once()
        mock_zipfile.assert_called_once()
        mock_remove.assert_called()

    @patch('mokutools.filetools.shutil.which')
    @patch('mokutools.filetools.get_file_list')
    def test_download_files_no_mokucli_when_convert_true(self, mock_get_file_list, mock_which):
        """Test that download_files returns early if mokucli not found and convert=True."""
        mock_which.return_value = None
        mock_get_file_list.return_value = ["test_file.li"]

        result = download_files("10.128.100.198", file_names="test", convert=True)

        assert result is None
        mock_get_file_list.assert_not_called()

    @patch('mokutools.filetools.get_file_list')
    def test_download_files_no_matching_files(self, mock_get_file_list):
        """Test download_files when no matching files are found."""
        mock_get_file_list.return_value = ["other_file.li"]

        with patch('builtins.print'):  # Suppress print output
            result = download_files("10.128.100.198", file_names="test", convert=False)

        assert result is None

    @patch('mokutools.filetools.get_file_list')
    def test_download_files_with_date_filter(self, mock_get_file_list):
        """Test download_files with date filter."""
        mock_get_file_list.return_value = [
            "data_20240101_file1.li",
            "data_20240102_file2.li",
            "data_20240101_file3.li"
        ]

        with patch('mokutools.filetools.requests.get') as mock_get, \
             patch('builtins.open', new_callable=mock_open), \
             patch('mokutools.filetools.os.makedirs'), \
             patch('mokutools.filetools.shutil.move'), \
             patch('builtins.print'):
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.iter_content.return_value = [b"content"]
            mock_get.return_value.__enter__.return_value = mock_response
            mock_get.return_value.__exit__ = Mock(return_value=None)

            download_files("10.128.100.198", date="20240101", convert=False)

            # Should download 2 files matching the date
            assert mock_get.call_count == 2

    @patch('mokutools.filetools.get_file_list')
    def test_download_files_raises_error_when_no_file_names_or_date(self, mock_get_file_list):
        """Test that download_files raises ValueError when neither file_names nor date provided."""
        with pytest.raises(ValueError, match="You must provide either"):
            download_files("10.128.100.198", convert=False)

    @patch('mokutools.filetools.shutil.which')
    @patch('mokutools.filetools.get_file_list')
    @patch('mokutools.filetools.requests.get')
    @patch('mokutools.filetools.requests.delete')
    @patch('mokutools.filetools.subprocess.run')
    @patch('mokutools.filetools.os.makedirs')
    @patch('mokutools.filetools.os.remove')
    @patch('mokutools.filetools.shutil.move')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_files_remove_from_server(
        self,
        mock_file,
        mock_move,
        mock_remove,
        mock_makedirs,
        mock_subprocess,
        mock_delete,
        mock_get,
        mock_get_file_list,
        mock_which,
    ):
        """Test download_files with remove_from_server=True."""
        mock_which.return_value = "/usr/bin/mokucli"
        mock_get_file_list.return_value = ["test_file.li"]
        
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content.return_value = [b"content"]
        mock_response.status_code = 200
        mock_get.return_value.__enter__.return_value = mock_response
        mock_get.return_value.__exit__ = Mock(return_value=None)
        
        mock_delete_response = Mock()
        mock_delete_response.status_code = 200
        mock_delete.return_value = mock_delete_response

        with patch('builtins.print'):
            download_files(
                "10.128.100.198",
                file_names="test",
                convert=True,
                remove_from_server=True
            )

        mock_delete.assert_called_once_with("http://10.128.100.198/api/ssd/delete/test_file.li")


class TestParseCsvFile:
    """Tests for parse_csv_file function."""

    def test_parse_csv_file_plain_text(self, tmp_path):
        """Test parsing a plain text CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_content = """# Header line 1
# Header line 2
col1,col2,col3
1,2,3
4,5,6
"""
        csv_file.write_text(csv_content)

        num_cols, num_rows, num_header_rows, header = parse_csv_file(str(csv_file))

        assert num_cols == 3
        assert num_rows == 5  # 2 header + 3 data lines
        assert num_header_rows == 2
        assert len(header) == 2

    def test_parse_csv_file_zip(self, tmp_path):
        """Test parsing a CSV file inside a ZIP archive."""
        csv_content = """# Header
col1,col2
1,2
3,4
"""
        zip_file = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_file, 'w') as zf:
            zf.writestr("data.csv", csv_content)

        num_cols, num_rows, num_header_rows, header = parse_csv_file(str(zip_file))

        assert num_cols == 2
        assert num_rows == 4  # 1 header + 3 data lines
        assert num_header_rows == 1

    def test_parse_csv_file_tar(self, tmp_path):
        """Test parsing a CSV file inside a TAR archive."""
        csv_content = """# Header line
col1,col2,col3,col4
1,2,3,4
5,6,7,8
"""
        tar_file = tmp_path / "test.tar"
        with tarfile.open(tar_file, 'w') as tf:
            info = tarfile.TarInfo(name="data.csv")
            info.size = len(csv_content.encode('utf-8'))
            tf.addfile(info, BytesIO(csv_content.encode('utf-8')))

        num_cols, num_rows, num_header_rows, header = parse_csv_file(str(tar_file))

        assert num_cols == 4
        assert num_rows == 4  # 1 header + 3 data lines
        assert num_header_rows == 1

    def test_parse_csv_file_gz(self, tmp_path):
        """Test parsing a gzipped CSV file."""
        csv_content = """# Header
col1,col2
1,2
3,4
5,6
"""
        gz_file = tmp_path / "test.csv.gz"
        with gzip.open(gz_file, 'wb') as gf:
            gf.write(csv_content.encode('utf-8'))

        num_cols, num_rows, num_header_rows, header = parse_csv_file(str(gz_file))

        assert num_cols == 2
        assert num_rows == 5  # 1 header + 4 data lines
        assert num_header_rows == 1

    def test_parse_csv_file_7z(self, tmp_path):
        """Test parsing a CSV file inside a 7z archive."""
        csv_content = """# Header
col1,col2,col3
1,2,3
4,5,6
7,8,9
"""
        seven_zip_file = tmp_path / "test.7z"
        # Create a temporary CSV file first
        temp_csv = tmp_path / "data.csv"
        temp_csv.write_text(csv_content)
        
        # Create 7z archive
        with SevenZipFile(seven_zip_file, 'w') as szf:
            szf.write(str(temp_csv), "data.csv")
        
        # Remove temp file
        temp_csv.unlink()

        num_cols, num_rows, num_header_rows, header = parse_csv_file(str(seven_zip_file))

        assert num_cols == 3
        assert num_rows == 5  # 1 header + 4 data lines
        assert num_header_rows == 1

    def test_parse_csv_file_different_header_symbols(self, tmp_path):
        """Test parsing CSV with different header symbols."""
        csv_file = tmp_path / "test.csv"
        csv_content = """% Header 1
! Header 2
@ Header 3
col1,col2
1,2
"""
        csv_file.write_text(csv_content)

        num_cols, num_rows, num_header_rows, header = parse_csv_file(str(csv_file))

        assert num_cols == 2
        assert num_header_rows == 3
        assert len(header) == 3

    def test_parse_csv_file_custom_delimiter(self, tmp_path):
        """Test parsing CSV with custom delimiter."""
        csv_file = tmp_path / "test.csv"
        csv_content = """# Header
col1;col2;col3
1;2;3
4;5;6
"""
        csv_file.write_text(csv_content)

        num_cols, num_rows, num_header_rows, header = parse_csv_file(str(csv_file), delimiter=';')

        assert num_cols == 3

    def test_parse_csv_file_no_header_raises_error(self, tmp_path):
        """Test that parse_csv_file raises error when no header is found."""
        csv_file = tmp_path / "test.csv"
        csv_content = """col1,col2
1,2
3,4
"""
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="No header lines detected"):
            parse_csv_file(str(csv_file))

    def test_parse_csv_file_no_data_raises_error(self, tmp_path):
        """Test that parse_csv_file raises error when no data lines are found."""
        csv_file = tmp_path / "test.csv"
        csv_content = """# Header line 1
# Header line 2
"""
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="No valid data lines found"):
            parse_csv_file(str(csv_file))

    def test_parse_csv_file_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("")

        with pytest.raises(ValueError):
            parse_csv_file(str(csv_file))

    def test_parse_csv_file_with_logger(self, tmp_path):
        """Test parse_csv_file with custom logger."""
        import logging
        logger = logging.getLogger("test_logger")
        
        csv_file = tmp_path / "test.csv"
        csv_content = """# Header
col1,col2
1,2
"""
        csv_file.write_text(csv_content)

        with patch.object(logger, 'debug') as mock_debug:
            num_cols, num_rows, num_header_rows, header = parse_csv_file(str(csv_file), logger=logger)
            assert mock_debug.called


class TestGetColumnsWithNans:
    """Tests for get_columns_with_nans function."""

    def test_get_columns_with_nans_no_nans(self):
        """Test get_columns_with_nans with DataFrame containing no NaNs."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })

        result = get_columns_with_nans(df)

        assert result == {}

    def test_get_columns_with_nans_some_nans(self):
        """Test get_columns_with_nans with DataFrame containing some NaNs."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan],
            'col2': [4, 5, 6],
            'col3': [np.nan, 8, 9]
        })

        result = get_columns_with_nans(df)

        assert 'col1' in result
        assert 'col3' in result
        assert 'col2' not in result
        assert result['col1'] == 0
        assert result['col3'] == 2

    def test_get_columns_with_nans_all_nans(self):
        """Test get_columns_with_nans with DataFrame where all columns have NaNs."""
        df = pd.DataFrame({
            'col1': [np.nan, np.nan],
            'col2': [np.nan, np.nan]
        })

        result = get_columns_with_nans(df)

        assert len(result) == 2
        assert 'col1' in result
        assert 'col2' in result

    def test_get_columns_with_nans_mixed_types(self):
        """Test get_columns_with_nans with DataFrame containing mixed data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, np.nan],
            'float_col': [1.1, np.nan, 3.3],
            'str_col': ['a', 'b', 'c']
        })

        result = get_columns_with_nans(df)

        assert 'int_col' in result
        assert 'float_col' in result
        assert 'str_col' not in result

    def test_get_columns_with_nans_empty_dataframe(self):
        """Test get_columns_with_nans with empty DataFrame."""
        df = pd.DataFrame()

        result = get_columns_with_nans(df)

        assert result == {}

    def test_get_columns_with_nans_single_column(self):
        """Test get_columns_with_nans with single column DataFrame."""
        df = pd.DataFrame({
            'col1': [1, np.nan, 3]
        })

        result = get_columns_with_nans(df)

        assert 'col1' in result
        assert result['col1'] == 0
