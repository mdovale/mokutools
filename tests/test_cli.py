"""
Unit tests for mokutools.moku_io.cli module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from mokutools.moku_io.cli import app, dl, up, rm, download_cli, upload_cli, delete_cli


class TestCLIImports:
    """Test that CLI imports successfully and has expected structure."""

    def test_app_exists(self):
        """Test that the Typer app exists."""
        assert app is not None
        assert hasattr(app, 'command')

    def test_commands_exist(self):
        """Test that expected commands exist."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ls" in result.stdout
        assert "dl" in result.stdout
        assert "up" in result.stdout
        assert "rm" in result.stdout


class TestCLICommands:
    """Test CLI commands with mocked core functions."""

    @patch('mokutools.moku_io.cli.list_files')
    def test_cmd_ls_success(self, mock_list_files):
        """Test ls command successfully lists files."""
        mock_list_files.return_value = ["file1.li", "file2.li"]
        runner = CliRunner()
        result = runner.invoke(app, ["ls", "10.128.100.188"])
        
        assert result.exit_code == 0
        assert "file1.li" in result.stdout
        assert "file2.li" in result.stdout
        mock_list_files.assert_called_once_with("10.128.100.188")

    @patch('mokutools.moku_io.cli.list_files')
    def test_cmd_ls_empty(self, mock_list_files):
        """Test ls command with no files."""
        mock_list_files.return_value = []
        runner = CliRunner()
        result = runner.invoke(app, ["ls", "10.128.100.188"])
        
        assert result.exit_code == 0
        assert "No files found" in result.stdout

    @patch('mokutools.moku_io.cli.list_files')
    def test_cmd_ls_error(self, mock_list_files):
        """Test ls command handles errors."""
        mock_list_files.side_effect = Exception("Connection error")
        runner = CliRunner()
        result = runner.invoke(app, ["ls", "10.128.100.188"])
        
        assert result.exit_code == 1
        assert "Error" in result.stderr

    @patch('mokutools.moku_io.cli.download')
    def test_cmd_dl_with_pattern(self, mock_download):
        """Test dl command with pattern."""
        mock_download.return_value = ["file1.li"]
        runner = CliRunner()
        result = runner.invoke(app, ["dl", "10.128.100.188", "--pattern", "test"])
        
        assert result.exit_code == 0
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["ip"] == "10.128.100.188"
        assert call_kwargs["patterns"] == ["test"]

    @patch('mokutools.moku_io.cli.download')
    def test_cmd_dl_with_date(self, mock_download):
        """Test dl command with date."""
        mock_download.return_value = ["file1.li"]
        runner = CliRunner()
        result = runner.invoke(app, ["dl", "10.128.100.188", "--date", "20250101"])
        
        assert result.exit_code == 0
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["date"] == "20250101"

    @patch('mokutools.moku_io.cli.download')
    def test_cmd_dl_missing_pattern_or_date(self, mock_download):
        """Test dl command fails without pattern or date."""
        runner = CliRunner()
        result = runner.invoke(app, ["dl", "10.128.100.188"])
        
        assert result.exit_code == 1
        assert "Must provide" in result.stderr

    @patch('mokutools.moku_io.cli.upload')
    def test_cmd_up_success(self, mock_upload):
        """Test up command successfully uploads files."""
        mock_upload.return_value = {"file1.csv": True, "file2.csv": True}
        runner = CliRunner()
        result = runner.invoke(app, ["up", "10.128.100.188", "file1.csv", "file2.csv"])
        
        assert result.exit_code == 0
        assert "Uploaded" in result.stdout
        mock_upload.assert_called_once_with("10.128.100.188", ["file1.csv", "file2.csv"])

    @patch('mokutools.moku_io.cli.delete')
    @patch('mokutools.moku_io.cli.list_files')
    def test_cmd_rm_with_pattern(self, mock_list_files, mock_delete):
        """Test rm command with pattern and --yes flag."""
        mock_list_files.return_value = ["temp_file1.li", "temp_file2.li"]
        mock_delete.return_value = ["temp_file1.li", "temp_file2.li"]
        runner = CliRunner()
        result = runner.invoke(app, ["rm", "10.128.100.188", "--pattern", "temp", "--yes"])
        
        assert result.exit_code == 0
        mock_delete.assert_called_once()
        call_kwargs = mock_delete.call_args[1]
        assert call_kwargs["ip"] == "10.128.100.188"
        assert call_kwargs["patterns"] == ["temp"]
        assert call_kwargs["confirm"] is True

    @patch('mokutools.moku_io.cli.delete')
    @patch('mokutools.moku_io.cli.list_files')
    def test_cmd_rm_with_all(self, mock_list_files, mock_delete):
        """Test rm command with --all flag."""
        mock_list_files.return_value = ["file1.li", "file2.li"]
        mock_delete.return_value = ["file1.li", "file2.li"]
        runner = CliRunner()
        result = runner.invoke(app, ["rm", "10.128.100.188", "--all", "--yes"])
        
        assert result.exit_code == 0
        mock_delete.assert_called_once()
        call_kwargs = mock_delete.call_args[1]
        assert call_kwargs["delete_all"] is True
        assert call_kwargs["confirm"] is True

    @patch('mokutools.moku_io.cli.delete')
    @patch('mokutools.moku_io.cli.list_files')
    def test_cmd_rm_missing_pattern_or_all(self, mock_list_files, mock_delete):
        """Test rm command fails without pattern or --all."""
        runner = CliRunner()
        result = runner.invoke(app, ["rm", "10.128.100.188", "--yes"])
        
        assert result.exit_code == 1
        assert "Must provide" in result.stderr


class TestHelperFunctions:
    """Test the renamed helper functions and backward compatibility."""

    @patch('mokutools.moku_io.cli.download')
    def test_dl_function(self, mock_download):
        """Test the dl helper function."""
        mock_download.return_value = ["file1.li"]
        with patch('builtins.print'):
            dl("10.128.100.188", file_names=["test"])
        
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["ip"] == "10.128.100.188"
        assert call_kwargs["patterns"] == ["test"]

    @patch('mokutools.moku_io.cli.upload')
    def test_up_function(self, mock_upload):
        """Test the up helper function."""
        mock_upload.return_value = {"file1.csv": True}
        with patch('builtins.print'):
            up("10.128.100.188", ["file1.csv"])
        
        mock_upload.assert_called_once_with("10.128.100.188", ["file1.csv"])

    @patch('mokutools.moku_io.cli.delete')
    @patch('mokutools.moku_io.cli.list_files')
    @patch('builtins.input')
    def test_rm_function_with_confirmation(self, mock_input, mock_list_files, mock_delete):
        """Test the rm helper function with confirmation."""
        mock_list_files.return_value = ["temp_file.li"]
        mock_input.return_value = "yes"
        mock_delete.return_value = ["temp_file.li"]
        with patch('builtins.print'):
            rm("10.128.100.188", file_names=["temp"])
        
        mock_delete.assert_called_once()
        call_kwargs = mock_delete.call_args[1]
        assert call_kwargs["confirm"] is True

    @patch('mokutools.moku_io.cli.dl')
    def test_download_cli_deprecation(self, mock_dl):
        """Test that download_cli calls dl and shows deprecation warning."""
        with pytest.warns(DeprecationWarning, match="download_cli is deprecated"):
            download_cli("10.128.100.188", file_names=["test"])
        
        mock_dl.assert_called_once()

    @patch('mokutools.moku_io.cli.up')
    def test_upload_cli_deprecation(self, mock_up):
        """Test that upload_cli calls up and shows deprecation warning."""
        with pytest.warns(DeprecationWarning, match="upload_cli is deprecated"):
            upload_cli("10.128.100.188", ["file1.csv"])
        
        mock_up.assert_called_once()

    @patch('mokutools.moku_io.cli.rm')
    def test_delete_cli_deprecation(self, mock_rm):
        """Test that delete_cli calls rm and shows deprecation warning."""
        with pytest.warns(DeprecationWarning, match="delete_cli is deprecated"):
            delete_cli("10.128.100.188", file_names=["temp"])
        
        mock_rm.assert_called_once()

