import pytest
import os
from unittest.mock import patch, Mock
from click.testing import CliRunner
from cli import main


@pytest.mark.integration
class TestFullDubbingWorkflow:
    """Test the complete dubbing workflow with mocked dependencies."""

    def test_successful_full_dubbing_with_api_key(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test successful full dubbing workflow using API key authentication."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-api-key-123',
            '--input-language', 'English',
            '--llm-model', 'gemini-2.5-pro',
            '--tts-model', 'gemini-2.5-pro-preview-tts'
        ])
        
        assert result.exit_code == 0
        assert "🚀 Starting video dubbing process..." in result.output
        assert "✅ Video dubbing process completed successfully!" in result.output
        
        # Verify VideoProcessor was called with correct config
        mock_video_processor.process_video_dubbing.assert_called_once()
        call_args = mock_video_processor.process_video_dubbing.call_args
        
        video_path, output_path, config, logger_func = call_args[0]
        assert video_path == temp_video_file
        assert output_path == 'output.mp4'
        assert config['GEMINI_API_KEY'] == 'test-api-key-123'
        assert config['OUTPUT_LANGUAGE'] == 'Spanish'
        assert config['INPUT_LANGUAGE'] == 'English'
        assert config['MODEL_NAME'] == 'models/gemini-2.5-pro'
        assert config['TTS_MODEL'] == 'gemini-2.5-pro-preview-tts'
        assert config['USE_VERTEX_AI'] is False
        assert config['EXTRACTION_ONLY'] is False

    def test_successful_vertex_ai_workflow(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test successful dubbing workflow using Vertex AI authentication."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Hindi',
            '--use-vertex-ai',
            '--project-id', 'test-project-123',
            '--location', 'us-central1',
            '--compress', '720p'
        ])
        
        assert result.exit_code == 0
        assert "🚀 Starting video dubbing process..." in result.output
        assert "✅ Video dubbing process completed successfully!" in result.output
        
        # Verify VideoProcessor was called with correct Vertex AI config
        call_args = mock_video_processor.process_video_dubbing.call_args
        video_path, output_path, config, logger_func = call_args[0]
        
        assert config['USE_VERTEX_AI'] is True
        assert config['PROJECT_ID'] == 'test-project-123'
        assert config['LOCATION'] == 'us-central1'
        assert config['COMPRESSION_PROFILE'] == '720p'
        assert config['OUTPUT_LANGUAGE'] == 'Hindi'

    def test_extraction_only_workflow(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test extraction-only workflow that doesn't require authentication."""
        runner = CliRunner()
        
        # Mock VideoProcessor to return a directory path for extraction-only mode
        mock_video_processor.process_video_dubbing.return_value = "/path/to/working/dir"
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--extraction-only',
            '--compress', '360p'
        ])
        
        assert result.exit_code == 0
        assert "🚀 Starting video dubbing process..." in result.output
        assert "✅ Video dubbing process completed successfully!" in result.output
        
        # Verify config for extraction-only mode
        call_args = mock_video_processor.process_video_dubbing.call_args
        video_path, output_path, config, logger_func = call_args[0]
        
        assert config['EXTRACTION_ONLY'] is True
        assert config['COMPRESSION_PROFILE'] == '360p'
        assert config['GEMINI_API_KEY'] is None
        assert config['OUTPUT_LANGUAGE'] is None

    def test_workflow_with_environment_variables(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test workflow using environment variables for authentication."""
        runner = CliRunner()
        
        env = {
            'GEMINI_API_KEY': 'env-api-key-456',
            'GOOGLE_CLOUD_PROJECT': 'env-project-789',
            'GOOGLE_CLOUD_LOCATION': 'us-west1'
        }
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'French',
            '--use-vertex-ai'
        ], env=env)
        
        assert result.exit_code == 0
        
        # Verify environment variables were used
        call_args = mock_video_processor.process_video_dubbing.call_args
        video_path, output_path, config, logger_func = call_args[0]
        
        assert config['USE_VERTEX_AI'] is True
        assert config['PROJECT_ID'] == 'env-project-789'
        assert config['LOCATION'] == 'us-west1'

    def test_workflow_with_reuse_mode(self, mock_video_processor, mock_file_system, temp_video_file, temp_working_dir, clean_env):
        """Test workflow with reuse mode using previous working directory."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'German',
            '--gemini-api-key', 'test-api-key',
            '--reuse', temp_working_dir
        ])
        
        assert result.exit_code == 0
        
        # Verify reuse configuration
        call_args = mock_video_processor.process_video_dubbing.call_args
        video_path, output_path, config, logger_func = call_args[0]
        
        assert config['REUSE_PATH'] == temp_working_dir
        assert config['OUTPUT_LANGUAGE'] == 'German'

    def test_workflow_with_custom_models(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test workflow with custom model specifications."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Japanese',
            '--gemini-api-key', 'test-api-key',
            '--llm-model', 'gemini-1.5-pro',
            '--tts-model', 'gemini-1.5-pro-preview-tts',
            '--input-language', 'Korean',
            '--working-dir', 'custom-work-dir',
            '--strict'
        ])
        
        assert result.exit_code == 0
        
        # Verify custom model configuration
        call_args = mock_video_processor.process_video_dubbing.call_args
        video_path, output_path, config, logger_func = call_args[0]
        
        assert config['MODEL_NAME'] == 'models/gemini-1.5-pro'
        assert config['TTS_MODEL'] == 'gemini-1.5-pro-preview-tts'
        assert config['INPUT_LANGUAGE'] == 'Korean'
        assert config['OUTPUT_LANGUAGE'] == 'Japanese'
        assert config['WORKING_DIR'] == 'custom-work-dir'
        assert config['STRICT'] is True

    def test_failed_video_processing(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test handling when VideoProcessor fails."""
        runner = CliRunner()
        
        # Mock VideoProcessor to return False (failure)
        mock_video_processor.process_video_dubbing.return_value = False
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-api-key'
        ])
        
        assert result.exit_code == 1
        assert "🚀 Starting video dubbing process..." in result.output
        assert "❌ Video dubbing process failed!" in result.output

    def test_video_processor_exception(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test handling when VideoProcessor raises an exception."""
        runner = CliRunner()
        
        # Mock VideoProcessor to raise an exception
        mock_video_processor.process_video_dubbing.side_effect = Exception("Processing failed")
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-api-key'
        ])
        
        assert result.exit_code == 1
        assert "🚀 Starting video dubbing process..." in result.output
        assert "❌ Video dubbing process failed!" in result.output

    def test_compression_profiles(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test all compression profile options."""
        runner = CliRunner()
        
        for profile in ['360p', '720p', '1080p', '360P', '720P', '1080P']:  # Test case insensitivity
            result = runner.invoke(main, [
                '--input-video', temp_video_file,
                '--output-video', 'output.mp4',
                '--output-language', 'Spanish',
                '--gemini-api-key', 'test-api-key',
                '--compress', profile
            ])
            
            assert result.exit_code == 0
            
            call_args = mock_video_processor.process_video_dubbing.call_args
            config = call_args[0][2]
            assert config['COMPRESSION_PROFILE'] == profile.lower()

    @pytest.mark.parametrize("input_lang,output_lang", [
        ("English", "Spanish"),
        ("Telugu", "Hindi"),
        ("Chinese", "English"),
        ("French", "German"),
    ])
    def test_multiple_language_combinations(self, mock_video_processor, mock_file_system, temp_video_file, clean_env, input_lang, output_lang):
        """Test various input/output language combinations."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-video', 'output.mp4',
            '--input-language', input_lang,
            '--output-language', output_lang,
            '--gemini-api-key', 'test-api-key'
        ])
        
        assert result.exit_code == 0
        
        call_args = mock_video_processor.process_video_dubbing.call_args
        config = call_args[0][2]
        assert config['INPUT_LANGUAGE'] == input_lang
        assert config['OUTPUT_LANGUAGE'] == output_lang