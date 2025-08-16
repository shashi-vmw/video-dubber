import pytest
from unittest.mock import patch, Mock
from click.testing import CliRunner
from cli import main, logger


@pytest.mark.unit
class TestMainCLIFunction:
    """Test the main CLI function behavior and flow."""

    def test_video_processor_initialization(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that VideoProcessor is initialized with correct working directory."""
        runner = CliRunner()
        
        with patch('cli.VideoProcessor') as mock_processor_class:
            mock_instance = Mock()
            mock_processor_class.return_value = mock_instance
            mock_instance.process_video_dubbing.return_value = True
            
            result = runner.invoke(main, [
                '--input-video', temp_video_file,
                '--output-video', 'output.mp4',
                '--output-language', 'Spanish',
                '--gemini-api-key', 'test-key',
                '--working-dir', 'custom-work-dir'
            ])
            
            assert result.exit_code == 0
            mock_processor_class.assert_called_once_with('custom-work-dir')

    def test_process_video_dubbing_called_with_correct_params(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that process_video_dubbing is called with correct parameters."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'custom-output.mp4',
            '--output-language', 'German',
            '--gemini-api-key', 'test-key-123'
        ])
        
        assert result.exit_code == 0
        
        # Verify the method was called once
        mock_video_processor.process_video_dubbing.assert_called_once()
        
        # Get call arguments
        call_args = mock_video_processor.process_video_dubbing.call_args[0]
        input_video, output_video, config, logger_func = call_args
        
        assert input_video == temp_video_file
        assert output_video == 'custom-output.mp4'
        assert config['OUTPUT_LANGUAGE'] == 'German'
        assert config['GEMINI_API_KEY'] == 'test-key-123'
        assert callable(logger_func)

    def test_success_flow_logging(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test logging output for successful execution."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code == 0
        assert "🚀 Starting video dubbing process..." in result.output
        assert "✅ Video dubbing process completed successfully!" in result.output

    def test_failure_flow_logging_and_exit_code(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test logging and exit code for failed execution."""
        runner = CliRunner()
        
        # Mock VideoProcessor to return False (failure)
        mock_video_processor.process_video_dubbing.return_value = False
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code == 1
        assert "🚀 Starting video dubbing process..." in result.output
        assert "❌ Video dubbing process failed!" in result.output

    def test_exception_handling(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that exceptions from VideoProcessor are handled gracefully."""
        runner = CliRunner()
        
        # Mock VideoProcessor to raise an exception
        mock_video_processor.process_video_dubbing.side_effect = Exception("Test exception")
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code == 1
        assert "🚀 Starting video dubbing process..." in result.output
        assert "❌ Video dubbing process failed!" in result.output

    def test_logger_function_works(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that the logger function passed to VideoProcessor works correctly."""
        runner = CliRunner()
        
        # Capture the logger function passed to VideoProcessor
        captured_logger = None
        
        def capture_logger(*args):
            nonlocal captured_logger
            captured_logger = args[3]  # Logger is the 4th argument
            return True
        
        mock_video_processor.process_video_dubbing.side_effect = capture_logger
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code == 0
        assert captured_logger is not None
        assert callable(captured_logger)
        
        # Test that the logger function works (it should be the same as our logger function)
        assert captured_logger == logger

    def test_extraction_only_success_path(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test successful extraction-only workflow."""
        runner = CliRunner()
        
        # Mock to return a directory path for extraction-only mode
        mock_video_processor.process_video_dubbing.return_value = "/path/to/extraction/dir"
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--extraction-only'
        ])
        
        assert result.exit_code == 0
        assert "🚀 Starting video dubbing process..." in result.output
        assert "✅ Video dubbing process completed successfully!" in result.output

    def test_video_processor_return_none(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test handling when VideoProcessor returns None."""
        runner = CliRunner()
        
        mock_video_processor.process_video_dubbing.return_value = None
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code == 1
        assert "🚀 Starting video dubbing process..." in result.output
        assert "❌ Video dubbing process failed!" in result.output

    def test_video_processor_return_truthy_string(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test handling when VideoProcessor returns a truthy string (file path)."""
        runner = CliRunner()
        
        mock_video_processor.process_video_dubbing.return_value = "/path/to/output.mp4"
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code == 0
        assert "🚀 Starting video dubbing process..." in result.output
        assert "✅ Video dubbing process completed successfully!" in result.output

    def test_video_processor_return_empty_string(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test handling when VideoProcessor returns empty string (falsy)."""
        runner = CliRunner()
        
        mock_video_processor.process_video_dubbing.return_value = ""
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code == 1
        assert "🚀 Starting video dubbing process..." in result.output
        assert "❌ Video dubbing process failed!" in result.output

    @patch('cli.exit')
    def test_exit_called_on_failure(self, mock_exit, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that exit(1) is called when processing fails."""
        runner = CliRunner()
        
        mock_video_processor.process_video_dubbing.return_value = False
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        # In CLI testing, the exit doesn't actually happen, but we can verify the call
        mock_exit.assert_called_once_with(1)

    def test_working_directory_default(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that default working directory is used when not specified."""
        runner = CliRunner()
        
        with patch('cli.VideoProcessor') as mock_processor_class:
            mock_instance = Mock()
            mock_processor_class.return_value = mock_instance
            mock_instance.process_video_dubbing.return_value = True
            
            result = runner.invoke(main, [
                '--input-video', temp_video_file,
                '--output-video', 'output.mp4',
                '--output-language', 'Spanish',
                '--gemini-api-key', 'test-key'
            ])
            
            assert result.exit_code == 0
            mock_processor_class.assert_called_once_with('working-dir')

    def test_multiple_consecutive_calls(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that multiple consecutive CLI calls work independently."""
        runner = CliRunner()
        
        # First call
        result1 = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output1.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key1'
        ])
        
        assert result1.exit_code == 0
        first_call_count = mock_video_processor.process_video_dubbing.call_count
        
        # Second call
        result2 = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output2.mp4',
            '--output-language', 'French',
            '--gemini-api-key', 'test-key2'
        ])
        
        assert result2.exit_code == 0
        assert mock_video_processor.process_video_dubbing.call_count == first_call_count + 1