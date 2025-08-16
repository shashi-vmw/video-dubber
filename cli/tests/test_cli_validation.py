import pytest
import os
from click.testing import CliRunner
from cli import main


@pytest.mark.unit
class TestCLIValidation:
    """Test CLI argument validation and error handling."""

    def test_missing_input_video(self, clean_env):
        """Test that missing input video fails validation."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code != 0
        assert "Missing option '--input-video'" in result.output

    def test_missing_output_path_for_full_dubbing(self, temp_video_file, clean_env):
        """Test that missing output path fails validation for full dubbing."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code != 0
        assert "Missing option '--output-path'" in result.output

    def test_missing_output_language_for_full_dubbing(self, mock_file_system, temp_video_file, clean_env):
        """Test that missing output language fails validation for full dubbing."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code != 0
        assert "Missing option '--output-language'" in result.output

    def test_output_language_not_required_for_extraction_only(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that output language is not required for extraction-only mode."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--extraction-only'
        ])
        
        assert result.exit_code == 0

    def test_missing_api_key_and_vertex_ai(self, mock_file_system, temp_video_file, clean_env):
        """Test that missing authentication method fails validation."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish'
        ])
        
        assert result.exit_code != 0
        assert "Please provide a Gemini API Key" in result.output

    def test_vertex_ai_missing_project_id(self, mock_file_system, temp_video_file, clean_env):
        """Test that Vertex AI without project ID fails validation."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--use-vertex-ai',
            '--location', 'us-central1'
        ])
        
        assert result.exit_code != 0
        assert "you must provide a Project ID and Location" in result.output

    def test_vertex_ai_missing_location(self, mock_file_system, temp_video_file, clean_env):
        """Test that Vertex AI without location fails validation."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--use-vertex-ai',
            '--project-id', 'test-project'
        ])
        
        assert result.exit_code != 0
        assert "you must provide a Project ID and Location" in result.output

    def test_reuse_and_strict_conflict(self, mock_file_system, temp_video_file, temp_working_dir, clean_env):
        """Test that reuse and strict flags cannot be used together."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key',
            '--reuse', temp_working_dir,
            '--strict'
        ])
        
        assert result.exit_code != 0
        assert "--reuse and --strict flags cannot be used together" in result.output

    def test_reuse_and_extraction_only_conflict(self, mock_file_system, temp_video_file, temp_working_dir, clean_env):
        """Test that reuse and extraction-only flags cannot be used together."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--reuse', temp_working_dir,
            '--extraction-only'
        ])
        
        assert result.exit_code != 0
        assert "--reuse and --extraction-only flags cannot be used together" in result.output

    def test_nonexistent_input_video(self, clean_env):
        """Test that non-existent input video fails validation."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', '/nonexistent/video.mp4',
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_nonexistent_reuse_path(self, mock_file_system, temp_video_file, clean_env):
        """Test that non-existent reuse path fails validation."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key',
            '--reuse', '/nonexistent/path'
        ])
        
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_invalid_compression_profile(self, mock_file_system, temp_video_file, clean_env):
        """Test that invalid compression profile fails validation."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key',
            '--compress', 'invalid-profile'
        ])
        
        assert result.exit_code != 0
        assert "Invalid value for '--compress'" in result.output

    def test_valid_compression_profiles(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that all valid compression profiles are accepted."""
        runner = CliRunner()
        
        valid_profiles = ['360p', '720p', '1080p', '360P', '720P', '1080P']
        
        for profile in valid_profiles:
            result = runner.invoke(main, [
                '--input-video', temp_video_file,
                '--output-path', 'output',
                '--output-language', 'Spanish',
                '--gemini-api-key', 'test-key',
                '--compress', profile
            ])
            
            assert result.exit_code == 0, f"Profile {profile} should be valid"

    def test_environment_variable_api_key(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that API key from environment variable is used."""
        runner = CliRunner()
        
        env = {'GEMINI_API_KEY': 'env-api-key'}
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish'
        ], env=env)
        
        assert result.exit_code == 0

    def test_environment_variable_vertex_ai_config(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that Vertex AI config from environment variables is used."""
        runner = CliRunner()
        
        env = {
            'GOOGLE_CLOUD_PROJECT': 'env-project',
            'GOOGLE_CLOUD_LOCATION': 'env-location'
        }
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--use-vertex-ai'
        ], env=env)
        
        assert result.exit_code == 0

    def test_cli_option_overrides_env_var(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that CLI options override environment variables."""
        runner = CliRunner()
        
        env = {
            'GEMINI_API_KEY': 'env-api-key',
            'GOOGLE_CLOUD_PROJECT': 'env-project',
            'GOOGLE_CLOUD_LOCATION': 'env-location'
        }
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'cli-api-key',
            '--project-id', 'cli-project',
            '--location', 'cli-location'
        ], env=env)
        
        assert result.exit_code == 0
        
        # Verify CLI values were used, not environment values
        call_args = mock_video_processor.process_video_dubbing.call_args
        config = call_args[0][2]
        assert config['GEMINI_API_KEY'] == 'cli-api-key'
        assert config['PROJECT_ID'] == 'cli-project'
        assert config['LOCATION'] == 'cli-location'

    def test_no_auth_required_for_extraction_only(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that no authentication is required for extraction-only mode."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--extraction-only'
        ])
        
        assert result.exit_code == 0
        
        # Verify that authentication fields are None/False in config
        call_args = mock_video_processor.process_video_dubbing.call_args
        config = call_args[0][2]
        assert config['GEMINI_API_KEY'] is None
        assert config['USE_VERTEX_AI'] is False
        assert config['EXTRACTION_ONLY'] is True