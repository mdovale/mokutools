"""
Unit tests for mokutools.moku_io.core module.
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
import os
import tempfile
import scipy.io

from mokutools.moku_io.core import (
    list_files,
    download,
    upload,
    delete,
    parse_csv_file,
    is_mat_file,
    is_li_file,
    read_lines,
    get_columns_with_nans,
)


class TestListFiles:
    """Tests for list_files function."""

    @patch('mokutools.moku_io.core.requests.get')
    def test_list_files_success(self, mock_get):
        """Test successful file list retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": ["file1.li", "file2.li", "file3.li"]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = list_files("10.128.100.198")

        assert result == ["file1.li", "file2.li", "file3.li"]
        mock_get.assert_called_once_with("http://10.128.100.198/api/ssd/list")
        mock_response.raise_for_status.assert_called_once()

    @patch('mokutools.moku_io.core.requests.get')
    def test_list_files_with_filters(self, mock_get):
        """Test file list retrieval with filters."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": ["test_file1.li", "test_file2.li", "other_file.li", "TEST_file3.li"]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = list_files("10.128.100.198", filters=["test"])

        assert result == ["test_file1.li", "test_file2.li", "TEST_file3.li"]
        assert "other_file.li" not in result

    @patch('mokutools.moku_io.core.requests.get')
    def test_list_files_with_multiple_filters(self, mock_get):
        """Test file list retrieval with multiple filters (AND logic)."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": ["test_file1.li", "test_file2.li", "other_test.li", "file1.li"]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = list_files("10.128.100.198", filters=["test", "file"])

        assert result == ["test_file1.li", "test_file2.li"]
        assert "other_test.li" not in result
        assert "file1.li" not in result

    @patch('mokutools.moku_io.core.requests.get')
    def test_list_files_empty_data(self, mock_get):
        """Test file list retrieval with empty data."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = list_files("10.128.100.198")

        assert result == []

    @patch('mokutools.moku_io.core.requests.get')
    def test_list_files_missing_data_key(self, mock_get):
        """Test file list retrieval when data key is missing."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = list_files("10.128.100.198")

        assert result == []

    @patch('mokutools.moku_io.core.requests.get')
    def test_list_files_http_error(self, mock_get):
        """Test file list retrieval with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            list_files("10.128.100.198")


class TestDownload:
    """Tests for download function."""

    @patch('mokutools.moku_io.core.shutil.which')
    @patch('mokutools.moku_io.core.list_files')
    @patch('mokutools.moku_io.core.requests.get')
    @patch('mokutools.moku_io.core.subprocess.run')
    @patch('mokutools.moku_io.core.os.makedirs')
    @patch('mokutools.moku_io.core.os.remove')
    @patch('mokutools.moku_io.core.shutil.move')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_basic(
        self,
        mock_file,
        mock_move,
        mock_remove,
        mock_makedirs,
        mock_subprocess,
        mock_get,
        mock_list_files,
        mock_which,
    ):
        """Test basic file download without conversion."""
        mock_which.return_value = None  # mokucli not found, but convert=False
        mock_list_files.return_value = ["test_file.li"]
        
        # Mock download response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content.return_value = [b"file content"]
        mock_get.return_value.__enter__.return_value = mock_response
        mock_get.return_value.__exit__ = Mock(return_value=None)

        result = download("10.128.100.198", patterns="test", convert=False, archive=False)

        assert result == ["test_file.li"]
        mock_list_files.assert_called_once_with("10.128.100.198")
        mock_makedirs.assert_called_once()
        mock_get.assert_called_once_with("http://10.128.100.198/api/ssd/download/test_file.li", stream=True)

    @patch('mokutools.moku_io.core.shutil.which')
    @patch('mokutools.moku_io.core.list_files')
    @patch('mokutools.moku_io.core.requests.get')
    @patch('mokutools.moku_io.core.subprocess.run')
    @patch('mokutools.moku_io.core.zipfile.ZipFile')
    @patch('mokutools.moku_io.core.os.makedirs')
    @patch('mokutools.moku_io.core.os.remove')
    @patch('mokutools.moku_io.core.shutil.move')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_with_conversion_and_archive(
        self,
        mock_file,
        mock_move,
        mock_remove,
        mock_makedirs,
        mock_zipfile,
        mock_subprocess,
        mock_get,
        mock_list_files,
        mock_which,
    ):
        """Test file download with conversion and archiving."""
        mock_which.return_value = "/usr/bin/mokucli"
        mock_list_files.return_value = ["test_file.li"]
        
        # Mock download response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content.return_value = [b"file content"]
        mock_get.return_value.__enter__.return_value = mock_response
        mock_get.return_value.__exit__ = Mock(return_value=None)

        # Create CSV file after subprocess.run is called (simulating mokucli conversion)
        from pathlib import Path
        def create_csv_file(*args, **kwargs):
            # Create the CSV file that would be created by mokucli
            Path("test_file.csv").write_text("col1,col2\n1,2\n")
        
        mock_subprocess.side_effect = create_csv_file

        # Mock zipfile
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        mock_zipfile.return_value.__exit__ = Mock(return_value=None)

        result = download("10.128.100.198", patterns="test", convert=True, archive=True)

        assert result == ["test_file.li"]
        mock_subprocess.assert_called_once()
        mock_zipfile.assert_called_once()
        mock_remove.assert_called()
        
        # Clean up
        if os.path.exists("test_file.csv"):
            os.remove("test_file.csv")

    @patch('mokutools.moku_io.core.shutil.which')
    @patch('mokutools.moku_io.core.list_files')
    def test_download_no_mokucli_when_convert_true(self, mock_list_files, mock_which):
        """Test that download raises error if mokucli not found and convert=True."""
        mock_which.return_value = None
        mock_list_files.return_value = ["test_file.li"]

        with pytest.raises(ValueError, match="mokucli not found"):
            download("10.128.100.198", patterns="test", convert=True)

    @patch('mokutools.moku_io.core.list_files')
    def test_download_no_matching_files(self, mock_list_files):
        """Test download when no matching files are found."""
        mock_list_files.return_value = ["other_file.li"]

        result = download("10.128.100.198", patterns="test", convert=False)

        assert result == []

    @patch('mokutools.moku_io.core.list_files')
    def test_download_with_date_filter(self, mock_list_files):
        """Test download with date filter."""
        mock_list_files.return_value = [
            "data_20240101_file1.li",
            "data_20240102_file2.li",
            "data_20240101_file3.li"
        ]

        with patch('mokutools.moku_io.core.requests.get') as mock_get, \
             patch('builtins.open', new_callable=mock_open), \
             patch('mokutools.moku_io.core.os.makedirs'), \
             patch('mokutools.moku_io.core.shutil.move'):
            
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.iter_content.return_value = [b"content"]
            mock_get.return_value.__enter__.return_value = mock_response
            mock_get.return_value.__exit__ = Mock(return_value=None)

            result = download("10.128.100.198", date="20240101", convert=False)

            # Should download 2 files matching the date
            assert len(result) == 2
            assert mock_get.call_count == 2

    @patch('mokutools.moku_io.core.list_files')
    def test_download_raises_error_when_no_patterns_or_date(self, mock_list_files):
        """Test that download raises ValueError when neither patterns nor date provided."""
        with pytest.raises(ValueError, match="You must provide either"):
            download("10.128.100.198", convert=False)

    @patch('mokutools.moku_io.core.shutil.which')
    @patch('mokutools.moku_io.core.list_files')
    @patch('mokutools.moku_io.core.requests.get')
    @patch('mokutools.moku_io.core.requests.delete')
    @patch('mokutools.moku_io.core.subprocess.run')
    @patch('mokutools.moku_io.core.zipfile.ZipFile')
    @patch('mokutools.moku_io.core.os.makedirs')
    @patch('mokutools.moku_io.core.os.remove')
    @patch('mokutools.moku_io.core.shutil.move')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_remove_from_server(
        self,
        mock_file,
        mock_move,
        mock_remove,
        mock_makedirs,
        mock_zipfile,
        mock_subprocess,
        mock_delete,
        mock_get,
        mock_list_files,
        mock_which,
    ):
        """Test download with remove_from_server=True."""
        mock_which.return_value = "/usr/bin/mokucli"
        mock_list_files.return_value = ["test_file.li"]
        
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content.return_value = [b"content"]
        mock_response.status_code = 200
        mock_get.return_value.__enter__.return_value = mock_response
        mock_get.return_value.__exit__ = Mock(return_value=None)
        
        mock_delete_response = Mock()
        mock_delete_response.status_code = 200
        mock_delete_response.raise_for_status = Mock()
        mock_delete.return_value = mock_delete_response

        # Create CSV file after subprocess.run is called (simulating mokucli conversion)
        from pathlib import Path
        def create_csv_file(*args, **kwargs):
            # Create the CSV file that would be created by mokucli
            Path("test_file.csv").write_text("col1,col2\n1,2\n")
        
        mock_subprocess.side_effect = create_csv_file

        # Mock zipfile
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        mock_zipfile.return_value.__exit__ = Mock(return_value=None)

        result = download(
            "10.128.100.198",
            patterns="test",
            convert=True,
            archive=True,
            remove_from_server=True
        )

        assert result == ["test_file.li"]
        mock_delete.assert_called_once_with("http://10.128.100.198/api/ssd/delete/test_file.li")
        
        # Clean up
        if os.path.exists("test_file.csv"):
            os.remove("test_file.csv")


class TestUpload:
    """Tests for upload function."""

    @patch('mokutools.moku_io.core.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"file content")
    def test_upload_single_file(self, mock_file, mock_post):
        """Test uploading a single file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name

        try:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = upload("10.128.100.198", tmp_path)

            assert result[os.path.basename(tmp_path)] is True
            mock_post.assert_called_once()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @patch('mokutools.moku_io.core.requests.post')
    def test_upload_multiple_files(self, mock_post):
        """Test uploading multiple files."""
        files = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{i}.txt') as tmp:
                    tmp.write(f"content {i}".encode())
                    files.append(tmp.name)

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = upload("10.128.100.198", files)

            assert len(result) == 3
            assert all(result.values())
            assert mock_post.call_count == 3
        finally:
            for f in files:
                if os.path.exists(f):
                    os.remove(f)

    def test_upload_nonexistent_file(self):
        """Test uploading a file that doesn't exist."""
        result = upload("10.128.100.198", "/nonexistent/file.txt")

        assert result["file.txt"] is False

    @patch('mokutools.moku_io.core.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"file content")
    def test_upload_http_error(self, mock_file, mock_post):
        """Test upload with HTTP error."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name

        try:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
            mock_post.return_value = mock_response

            with pytest.raises(requests.HTTPError):
                upload("10.128.100.198", tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestDelete:
    """Tests for delete function."""

    @patch('mokutools.moku_io.core.list_files')
    @patch('mokutools.moku_io.core.requests.delete')
    def test_delete_with_patterns(self, mock_delete, mock_list_files):
        """Test deleting files with pattern matching."""
        mock_list_files.return_value = ["test_file1.li", "test_file2.li", "other_file.li"]
        
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_delete.return_value = mock_response

        result = delete("10.128.100.198", patterns="test", confirm=True)

        assert len(result) == 2
        assert "test_file1.li" in result
        assert "test_file2.li" in result
        assert mock_delete.call_count == 2

    @patch('mokutools.moku_io.core.list_files')
    @patch('mokutools.moku_io.core.requests.delete')
    def test_delete_all_files(self, mock_delete, mock_list_files):
        """Test deleting all files."""
        mock_list_files.return_value = ["file1.li", "file2.li", "file3.li"]
        
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_delete.return_value = mock_response

        result = delete("10.128.100.198", delete_all=True, confirm=True)

        assert len(result) == 3
        assert mock_delete.call_count == 3

    @patch('mokutools.moku_io.core.list_files')
    def test_delete_requires_confirmation(self, mock_list_files):
        """Test that delete requires confirmation."""
        mock_list_files.return_value = ["test_file.li"]

        with pytest.raises(ValueError, match="Deletion requires confirmation"):
            delete("10.128.100.198", patterns="test", confirm=False)

    @patch('mokutools.moku_io.core.list_files')
    def test_delete_no_files_found(self, mock_list_files):
        """Test delete when no matching files are found."""
        mock_list_files.return_value = ["other_file.li"]

        result = delete("10.128.100.198", patterns="test", confirm=True)

        assert result == []

    @patch('mokutools.moku_io.core.list_files')
    def test_delete_raises_error_when_no_patterns_or_delete_all(self, mock_list_files):
        """Test that delete raises ValueError when neither patterns nor delete_all provided."""
        with pytest.raises(ValueError, match="Must specify"):
            delete("10.128.100.198", confirm=True)


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


class TestIsMatFile:
    """Tests for is_mat_file function."""

    def test_is_mat_file_valid(self, tmp_path):
        """Test is_mat_file with a valid .mat file."""
        mat_file = tmp_path / "test.mat"
        
        # Create a minimal valid .mat file
        data = {'test_var': np.array([1, 2, 3])}
        scipy.io.savemat(str(mat_file), data)

        assert is_mat_file(str(mat_file)) is True

    def test_is_mat_file_invalid(self, tmp_path):
        """Test is_mat_file with an invalid file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a mat file")

        assert is_mat_file(str(txt_file)) is False

    def test_is_mat_file_nonexistent(self):
        """Test is_mat_file with a nonexistent file."""
        assert is_mat_file("/nonexistent/file.mat") is False


class TestIsLiFile:
    """Tests for is_li_file function."""

    def test_is_li_file_valid(self, tmp_path):
        """Test is_li_file with a file that starts with 'LI'."""
        li_file = tmp_path / "test.li"
        with open(li_file, 'wb') as f:
            f.write(b'LI' + b'\x00' * 6)  # Write 'LI' followed by zeros

        assert is_li_file(str(li_file)) is True

    def test_is_li_file_invalid(self, tmp_path):
        """Test is_li_file with an invalid file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an li file")

        assert is_li_file(str(txt_file)) is False

    def test_is_li_file_nonexistent(self):
        """Test is_li_file with a nonexistent file."""
        assert is_li_file("/nonexistent/file.li") is False


class TestReadLines:
    """Tests for read_lines function."""

    def test_read_lines_success(self, tmp_path):
        """Test reading lines from a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        result = read_lines(str(test_file), 3)

        assert len(result) == 3
        assert result == ["line1", "line2", "line3"]

    def test_read_lines_fewer_lines_than_requested(self, tmp_path):
        """Test reading when file has fewer lines than requested."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2")

        result = read_lines(str(test_file), 5)

        assert len(result) == 2
        assert result == ["line1", "line2"]

    def test_read_lines_empty_file(self, tmp_path):
        """Test reading from an empty file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("")

        result = read_lines(str(test_file), 5)

        assert result == []

    def test_read_lines_nonexistent_file(self):
        """Test reading from a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            read_lines("/nonexistent/file.txt", 5)


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

